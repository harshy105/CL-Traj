import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict
from torch import Tensor 

from network.layers.embedding import Embedding
from network.layers.attention_layer import AttentionLayer
from network.layers.mlp_layer import MLPLayer
from utilities.transformation import recurr_to_vcs_to_local_tensor
from utilities.weight_init import weight_init
from utilities.create_edges import (get_prev2mode_edges, get_map2mode_edges, 
                    get_surr2mode_edges, get_goal2mode_edges,
                    purge_far_away_edges, compute_rel_pos_time_emb)


class Decoder(nn.Module):
    def __init__(
        self,
        num_modes: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        head_dim: int,
        use_goal: bool,
        max_num_input_frames: int,
        num_dec_recurr_steps: int,
        num_layers: int,
        num_dec_future_steps: int,
        output_dim: int,
        rel_dis_norm_factor: float,
        dis_threshold: float,
        modes_noise_factor: float,
    ):
        super(Decoder, self).__init__()
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_goal = use_goal
        self.max_num_input_frames = max_num_input_frames
        self.num_dec_recurr_steps = num_dec_recurr_steps
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_dec_future_steps = num_dec_future_steps
        self.num_prediction_steps = num_dec_future_steps // num_dec_recurr_steps
        self.rel_dis_norm_factor = rel_dis_norm_factor
        self.dis_threshold = dis_threshold
        self.modes_noise_factor = modes_noise_factor

        input_dim_r_t2m = 6
        self.r_t2m_emb = Embedding(input_dim=input_dim_r_t2m, hidden_dim=hidden_dim)
        self.t2m_attn_layers = nn.ModuleList([AttentionLayer(hidden_dim=hidden_dim,
                    num_heads=num_heads, head_dim=head_dim, dropout=dropout, bipartite=True,
                    has_pos_emb=True) for _ in range(num_layers)])
        
        if use_goal:
            input_dim_r_g2m = 6
            self.r_g2m_emb = Embedding(input_dim=input_dim_r_g2m, hidden_dim=hidden_dim)
            self.g2m_attn_layers = nn.ModuleList([AttentionLayer(hidden_dim=hidden_dim,
                    num_heads=num_heads, head_dim=head_dim, dropout=dropout, bipartite=True,
                    has_pos_emb=True) for _ in range(num_layers)])
        
        input_dim_r_pt2m = 6
        self.r_pt2m_emb = Embedding(input_dim=input_dim_r_pt2m, hidden_dim=hidden_dim)
        self.pt2m_attn_layers = nn.ModuleList([AttentionLayer(hidden_dim=hidden_dim,
                    num_heads=num_heads, head_dim=head_dim, dropout=dropout, bipartite=True,
                    has_pos_emb=True) for _ in range(num_layers)])
        
        input_dim_r_a2m = 6
        self.r_a2m_emb = Embedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim)
        self.a2m_attn_layers = nn.ModuleList([AttentionLayer(hidden_dim=hidden_dim,
                    num_heads=num_heads, head_dim=head_dim, dropout=dropout, bipartite=True,
                    has_pos_emb=True) for _ in range(num_layers)])

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)

        self.to_delta_pos_mu = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, 
                                    output_dim=self.num_prediction_steps * output_dim)
        self.to_delta_pos_scale = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, 
                                    output_dim=self.num_prediction_steps * output_dim)
        if self.num_modes > 1:
            self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        
        self.apply(weight_init)

    def forward(self, current_frame: int, sim_mask: Tensor, data: Dict[str, Tensor], scene_enc: Dict[str, Tensor]) -> Dict[str, Tensor]:
        DEVICE = data['agents_position'].device
        dtype = data['agents_position'].dtype
        num_agents = data["agents_position"].shape[0]
        
        # set the max num input frames
        if (self.max_num_input_frames is not None) and (current_frame + 1 > self.max_num_input_frames):
            max_num_input_frames = self.max_num_input_frames
        else:
            max_num_input_frames = current_frame + 1

        # create the edges
        mask_am = sim_mask[:, None].repeat(1, self.num_modes) # (A, M)
        mask_at = data["agents_input_reg_mask"][:, current_frame + 1 - max_num_input_frames : current_frame + 1].bool()  # (A, T_prev)
        edge_at2am = get_prev2mode_edges(mask_at, mask_am)
        if self.use_goal:
            mask_ag = data["agents_goal_mask"][:, current_frame + 1 :].bool()
            edge_ag2am = get_goal2mode_edges(mask_ag, mask_am)
        mask_ac = data["agents_input_reg_mask"][:, current_frame].bool() # (A,)
        edge_ma2ma = get_surr2mode_edges(self.num_modes, mask_am, mask_ac, data["agents_batch"])
        edge_mpt2ma = get_map2mode_edges(self.num_modes, mask_am, data["agents_batch"], data["lanes_batch"])
            
        # get the positions, heading and timestep information for creating the rel_embeddings
        pos_at = data["agents_position"][:, current_frame + 1 - max_num_input_frames : current_frame + 1].reshape(-1, self.output_dim)  # (A*T_prev, 2)
        head_at = data["agents_heading_rad"][:, current_frame + 1 - max_num_input_frames : current_frame + 1].reshape(-1)  # (A*T_prev,)
        time_at = torch.arange(-max_num_input_frames+1, 1, device=DEVICE, dtype=dtype).repeat(num_agents) # (A*T_prev,)
        assert head_at.shape == time_at.shape
        
        if self.use_goal:
            pos_ag = data["agents_position"][:, current_frame + 1 :].reshape(-1, self.output_dim) # (A*T, 2)
            head_ag = data["agents_heading_rad"][:, current_frame + 1 :].reshape(-1) # (A*T,)
            time_ag = torch.arange(0, data["agents_heading_rad"].shape[-1] - current_frame - 1, 
                                   device=DEVICE, dtype=dtype).repeat(num_agents) # (A*T,)
            assert head_ag.shape == time_ag.shape

        pos_pt = (data["lanes_end_locs"] + data["lanes_start_locs"]) / 2
        vector_pt = data["lanes_end_locs"] - data["lanes_start_locs"]
        head_pt = torch.atan2(vector_pt[:, 1], vector_pt[:, 0])
        pos_mpt = pos_pt.repeat(self.num_modes, 1) # (M*N_l, 2)
        head_mpt = head_pt.repeat(self.num_modes) # (M*N_l,)
        time_mpt = torch.zeros_like(head_mpt) # (M*N_l,)
        assert head_mpt.shape == time_mpt.shape
        
        # init embeddings
        query_m = self.mode_emb.weight
        if self.training and self.modes_noise_factor > 0.0:
            query_m = query_m + torch.randn_like(query_m) * self.modes_noise_factor  # add noise to the modes
        query_am = query_m.repeat(scene_enc["x_a"].size(0), 1)
        x_at = scene_enc["x_a"].reshape(-1, self.hidden_dim)
        if self.use_goal:
            x_ag = torch.zeros(*(*mask_ag.shape, self.hidden_dim), device=DEVICE, dtype=dtype).reshape(-1, self.hidden_dim)
        x_mpt = scene_enc["x_pt"].repeat(self.num_modes, 1)
        x_ma = scene_enc["x_a"][:, -1].repeat(self.num_modes, 1)

        traj_pos_mu_local = torch.zeros(num_agents, self.num_modes, self.num_layers, 
                                self.num_dec_future_steps, self.output_dim, device=DEVICE)
        traj_pos_scale_local = torch.zeros(num_agents, self.num_modes, self.num_layers, 
                                self.num_dec_future_steps, self.output_dim, self.output_dim, device=DEVICE)

        for i in range(self.num_layers):
            pos_am = data["agents_position"][:, current_frame, :2].repeat_interleave(self.num_modes, dim=0)  # (A*M, 2)
            head_am = data["agents_heading_rad"][:, current_frame].repeat_interleave(self.num_modes, dim=0)  # (A*M)
            time_am = torch.zeros_like(head_am) # (A*M,)
            
            pos_ma = data["agents_position"][:, current_frame, :2].repeat(self.num_modes, 1)  # (M*A, 2)
            head_ma = data["agents_heading_rad"][:, current_frame].repeat(self.num_modes)  # (M*A,) 
            time_ma = torch.zeros_like(head_ma) # (M*A,) 
            
            for t in range(self.num_dec_recurr_steps):
                # create position embs across input and modes of each agent
                rel_at2am = compute_rel_pos_time_emb(edge_at2am, pos_at, pos_am, head_at, head_am, time_at, time_am,
                                                          self.rel_dis_norm_factor, self.max_num_input_frames)
                r_at2am = self.r_t2m_emb(continuous_inputs=rel_at2am, categorical_embs=None)
                # create position embs across goal and modes of each agent
                if self.use_goal:
                    rel_ag2am = compute_rel_pos_time_emb(edge_ag2am, pos_ag, pos_am, head_ag, head_am, time_ag, time_am,
                                                          self.rel_dis_norm_factor, self.num_dec_future_steps)
                    r_ag2am = self.r_g2m_emb(continuous_inputs=rel_ag2am, categorical_embs=None)
                # create position embs across agents and modes at last time step
                edge_ma2ma_purged = purge_far_away_edges(edge_ma2ma, self.dis_threshold, pos_ma, pos_ma)
                rel_ma2ma_purged = compute_rel_pos_time_emb(edge_ma2ma_purged, pos_ma, pos_ma, head_ma, head_ma, time_ma, time_ma,
                                                          self.rel_dis_norm_factor, self.max_num_input_frames)
                r_ma2ma_purged = self.r_a2m_emb(continuous_inputs=rel_ma2ma_purged, categorical_embs=None)
                # create position embs across lanes and modes
                edge_mpt2ma_purged = purge_far_away_edges(edge_mpt2ma, self.dis_threshold, pos_mpt, pos_ma)
                rel_mpt2ma_purged = compute_rel_pos_time_emb(edge_mpt2ma_purged, pos_mpt, pos_ma, head_mpt, head_ma, time_mpt, time_ma,
                                                          self.rel_dis_norm_factor, self.max_num_input_frames)
                r_mpt2ma_purged = self.r_pt2m_emb(continuous_inputs=rel_mpt2ma_purged, categorical_embs=None)

                # perform cross attentions
                query_am, attn_t = self.t2m_attn_layers[i]((x_at, query_am), r_at2am, edge_at2am)
                if self.use_goal:
                    query_am, attn_g = self.g2m_attn_layers[i]((x_ag, query_am), r_ag2am, edge_ag2am)
                query_ma = query_am.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                query_ma, attn_pt = self.pt2m_attn_layers[i]((x_mpt, query_ma), r_mpt2ma_purged, edge_mpt2ma_purged)
                query_ma, attn_a = self.a2m_attn_layers[i]((x_ma, query_ma), r_ma2ma_purged, edge_ma2ma_purged)
                query_am = query_ma.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)

                # get traj in recurr coordindate system
                delta_pos_mu_recurr_itr = self.to_delta_pos_mu(query_am).view(-1, self.num_modes, self.num_prediction_steps, self.output_dim) # (A, M, T_pred, 2)
                delta_pos_scale_recurr_itr = self.to_delta_pos_scale(query_am).view(-1, self.num_modes, self.num_prediction_steps, self.output_dim) # (A, M, T_pred, 2)
                
                pos_mu_recurr_itr = torch.cumsum(delta_pos_mu_recurr_itr, dim=-2) # (A, M, T_pred, 2)
                pos_scale_recurr_itr = torch.cumsum(
                        F.elu_(delta_pos_scale_recurr_itr, alpha=1.0) + 1.0, # exp(x) for x < 0; x+1 for x > 0
                        dim=-2) + 0.1 # (A, M, T_pred, 2)
                
                pos_mu_recurr_itr[~sim_mask] = 0 # setting the positions for non sim agents to zero (following Heading computation for surr_agent will be unusable)
                pos_scale_recurr_itr[~sim_mask] = 1.0 # setting the scales for non sim agents to 1
                pos_scale_recurr_itr = torch.diag_embed(pos_scale_recurr_itr) # (A, M, T_pred, 2, 2)
                
                # transform the trajectories to vcs and local coordinate system
                (pos_mu_vcs_itr, head_mu_vcs_itr, _, pos_mu_local_itr, 
                 pos_scale_local_itr) = recurr_to_vcs_to_local_tensor(
                     recurr_vcs_pos=pos_am, # (A*M, 2)
                     recurr_vcs_head_rad=head_am, # (A*M,)
                     local_vcs_pos=data['agents_position'][:, current_frame, :2].repeat_interleave(self.num_modes, dim=0), # (A*M, 2)
                     local_vcs_head_rad=data['agents_heading_rad'][:, current_frame].repeat_interleave(self.num_modes, dim=0), # (A*M,)
                     traj_recurr_pos=pos_mu_recurr_itr.reshape(-1, self.num_prediction_steps, self.output_dim), # (A*M, T_pred, 2)
                     scale_recurr_pos=pos_scale_recurr_itr.reshape(-1, self.num_prediction_steps, self.output_dim, self.output_dim), # (A*M, T_pred, 2, 2)
                 ) # (A*M, T_pred, 2), (A*M, T_pred), (A*M, T_pred, 2), (A*M, T_pred, 2, 2)
                
                # store the traj and scale in local coordinate system for training                
                traj_pos_mu_local[:, :, i, self.num_prediction_steps * t : self.num_prediction_steps * (t + 1), 
                            :] = pos_mu_local_itr.reshape(num_agents, self.num_modes, self.num_prediction_steps, self.output_dim)
                traj_pos_scale_local[:, :, i, self.num_prediction_steps * t : self.num_prediction_steps * (t + 1), 
                            :, :] = pos_scale_local_itr.reshape(num_agents, self.num_modes, self.num_prediction_steps, self.output_dim, self.output_dim)
                
                # update the current position of agents in vcs
                if t < self.num_dec_recurr_steps - 1:
                    # update the position and heading of the simulated agents
                    pos_mu_vcs_itr_detached = pos_mu_vcs_itr.detach() # (A*M, T_pred, 2)
                    head_mu_vcs_itr_detached = head_mu_vcs_itr.detach() # (A*M, T_pred)
                    current_pos = pos_mu_vcs_itr_detached[:, -1, :].reshape(
                                                    num_agents, self.num_modes, self.output_dim) # (A, M, 2)
                    current_head = head_mu_vcs_itr_detached[:, -1].reshape(
                                                    num_agents, self.num_modes) # (A, M)
                    
                    current_pos[~sim_mask] = data["agents_position"][~sim_mask, current_frame, None, :2].repeat(1, self.num_modes, 1)
                    current_head[~sim_mask] = data["agents_heading_rad"][~sim_mask, current_frame, None].repeat(1, self.num_modes)
                    
                    pos_am = current_pos.reshape(-1, self.output_dim) # (A*M, 2)
                    head_am = current_head.reshape(-1) # (A*M,)
                    pos_ma = current_pos.transpose(0, 1).reshape(-1, self.output_dim) # (M*A, 2)
                    head_ma = current_head.transpose(0, 1).reshape(-1) # (M*A,)
                    
                    # update time of the simulated agents
                    next_time = torch.ones(num_agents, self.num_modes, device=DEVICE, dtype=dtype) * (self.num_prediction_steps*(t+1)) # (A, M)
                    next_time[~sim_mask] = 0 # Not updating the non sim agents
                    time_am = next_time.reshape(-1) # (A*M)
                    time_ma = next_time.transpose(0, 1).reshape(-1) # (M*A)
                    time_mpt = torch.ones_like(head_mpt) * (self.num_prediction_steps*(t+1))
        
        if not sim_mask.all():
            assert traj_pos_mu_local[~sim_mask].abs().max() < 1e-2, "Non sim agents must have zero mean prediction in local CS"
            assert (
                traj_pos_scale_local[~sim_mask] - torch.eye(self.output_dim, device=DEVICE, dtype=dtype)
            ).abs().max() < 1e-2, "Non sim agents must have Identity Covariance Matrix in local CS"
        
        if self.num_modes > 1:
            pi = self.to_pi(query_am.reshape(num_agents, self.num_modes, self.hidden_dim)).squeeze(-1) 
        else:
            pi = torch.zeros(num_agents, 1, device=DEVICE, dtype=dtype)

        return {
            "traj_pos_mu": traj_pos_mu_local,
            "traj_pos_scale": traj_pos_scale_local,
            "pi": pi,
        }