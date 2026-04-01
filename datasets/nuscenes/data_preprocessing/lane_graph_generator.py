import torch
import networkx as nx
import warnings
import numpy as np
import pickle
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from datasets.nuscenes.nuscenes_devkit.map_expansion.map_api import NuScenesMap
from config.nuscenes_config import NuScenesPreprocessConfig, CityNodesFeatures, CityCenterlinesGraph
from config.config import NUSCENES_PATH, EXTRACTED_MAP

'''
Generate lane graph for the cities
'''

def get_map_records(data_root: str, cities_names:List[str]) -> Dict[str, NuScenesMap]:
    cities_maps_records = {}
    for city_name in cities_names:
        city_map = NuScenesMap(dataroot=data_root, map_name=city_name)
        cities_maps_records[city_name] = city_map
    return cities_maps_records

def get_centerlines_graph_features(city_map:NuScenesMap, resolution:float) -> Tuple[Dict[str, int], CityNodesFeatures]:
    centerlines = city_map.get_records_in_radius(0, 0, 1e5, ['lane', 'lane_connector']) # gets the entire city lanes
    centerlines = centerlines['lane'] + centerlines['lane_connector']
    centerlines_points_pairs = city_map.discretize_lanes(centerlines, resolution)
    centerlines_nodes_pairs = dict()
    city_nodes_features = CityNodesFeatures(
        nodes_id = torch.zeros((1), dtype=torch.int32),
        start_locs = torch.zeros((1,2), dtype=torch.float),
        end_locs = torch.zeros((1,2), dtype=torch.float)
        )

    centerline_id = 0
    num_nodes = 0
    for k in centerlines_points_pairs:
        # compute the node features
        points_locs = torch.tensor(centerlines_points_pairs[k], dtype=torch.float)[:,:2]
        start_locs = points_locs[:-1] # (N, 2)
        end_locs = points_locs[1:] # # (N, 2)
        nodes_id = torch.arange(num_nodes, num_nodes+len(start_locs), dtype=torch.int32)

        # store the centerline nodes pairs
        centerlines_nodes_pairs[k] = nodes_id

        # append the centerline nodes features to city nodes 
        city_nodes_features.nodes_id = torch.concat((city_nodes_features.nodes_id, nodes_id), dim=0)
        city_nodes_features.start_locs = torch.concat((city_nodes_features.start_locs, start_locs), dim=0)
        city_nodes_features.end_locs = torch.concat((city_nodes_features.end_locs, end_locs), dim=0)

        # update indexs
        centerline_id += 1
        num_nodes += len(start_locs)
    
    # remove the init zero features
    city_nodes_features.nodes_id = city_nodes_features.nodes_id[1:]
    city_nodes_features.start_locs = city_nodes_features.start_locs[1:]
    city_nodes_features.end_locs = city_nodes_features.end_locs[1:]

    return centerlines_nodes_pairs, city_nodes_features

def get_centerlines_graph_edges(city_map:NuScenesMap, centerlines_nodes_pairs:Dict[str, int]) -> torch.Tensor:    
    city_edges_indices = torch.zeros((2,1), dtype=torch.int32)
    for k in centerlines_nodes_pairs:
        centerline_nodes_id = centerlines_nodes_pairs[k]
        # intra centerline connection
        source_nodes_id = centerline_nodes_id[1:]
        target_nodes_id = centerline_nodes_id[:-1]
        new_edges_indices = torch.stack((source_nodes_id, target_nodes_id), dim=0) # (2, N)
        city_edges_indices = torch.concat((city_edges_indices, new_edges_indices), dim=-1)   
        # get the connection among the incoming and outgoing nodes
        outgoing_centerlines = city_map.get_outgoing_lane_ids(lane_token=k)
        target_node_id = centerline_nodes_id[-1]
        for out_k in outgoing_centerlines:
            try:
                outgoing_centerline_nodes_id = centerlines_nodes_pairs[out_k]
                source_node_id = outgoing_centerline_nodes_id[0]
                new_edge_index = torch.stack((source_node_id, target_node_id), dim=0).unsqueeze(-1) # (2, 1)
                city_edges_indices = torch.concat((city_edges_indices, new_edge_index), dim=-1)
            except:
                warnings.warn(f'Warning: Outgoing lane {out_k} was not rasterised', stacklevel=2)

    # remove the init edge indices
    city_edges_indices = city_edges_indices[:,1:]

    return city_edges_indices   

def fix_centerline_in_nuScenes(city_name:str, city_map:NuScenesMap) -> None:
    # fix centerlines
    if city_name == 'singapore-onenorth':
        # 1st issue at location (524, 1269)
        buggy_token = '2c1f1364-131c-46f5-aa1f-67d546a1377b'
        buggy_arcline = city_map.get_arcline_path(buggy_token)
        outgoing_token = city_map.get_outgoing_lane_ids(buggy_token)[0] # only 1 outgoing
        outgoing_arcline = city_map.get_arcline_path(outgoing_token)
        #fix
        buggy_arcline[0]['end_pose'] = outgoing_arcline[0]['start_pose'].copy() 

def visualize_city_centerlines_graph(city_name:str, city_centerlines_graph:CityCenterlinesGraph, 
                                     origin:Tuple, radius:float) -> None:
    sampled_edges_indices = sample_edges_within_radius(city_centerlines_graph, origin, radius)
    nodes_id = city_centerlines_graph.nodes_features.nodes_id
    nodes_start_locs = city_centerlines_graph.nodes_features.start_locs
    nodes_end_locs = city_centerlines_graph.nodes_features.end_locs
    nodes_avg_locs = (nodes_start_locs + nodes_end_locs)/2
    source_nodes_avg_locs = nodes_avg_locs[nodes_id[sampled_edges_indices[0,:]]]
    target_nodes_avg_locs = nodes_avg_locs[nodes_id[sampled_edges_indices[1,:]]]
    # plot the graph
    graph_edges = target_nodes_avg_locs - source_nodes_avg_locs
    plt.scatter(origin[0], origin[1], color='orange', marker='o', alpha=0.8)
    for i in range(len(graph_edges)): # best for resolution=2
        plt.arrow(source_nodes_avg_locs[i,0], source_nodes_avg_locs[i,1], 
                  graph_edges[i,0], graph_edges[i,1], width=0.08, length_includes_head=True)
    plt.axis('square')
    plt.title(f'{city_name}')
    plt.show()
    plt.close('all')


def sample_edges_within_radius(city_centerlines_graph:CityCenterlinesGraph, origin:Tuple, radius:float) ->  torch.Tensor:
    # find the nodes with start location inside the area of interest
    #TODO introduce the angle of target agent
    start_locs = city_centerlines_graph.nodes_features.start_locs
    nodes_id = city_centerlines_graph.nodes_features.nodes_id
    sampled_nodes = ((start_locs[:,0] > origin[0]-radius) *
                    (start_locs[:,1] > origin[1]-radius) * 
                    (start_locs[:,0] < origin[0]+radius) * 
                    (start_locs[:,1] < origin[1]+radius))
    sampled_nodes_id = nodes_id[sampled_nodes]

    # get the edges for the corresponding nodes
    edges_indices = city_centerlines_graph.edges_indices
    sampled_edges_indices = sample_edges_for_nodes(edges_indices, sampled_nodes_id)
    return sampled_edges_indices


def sample_edges_for_nodes(edges_indices:torch.Tensor, sampled_nodes_id:torch.Tensor) -> torch.Tensor:
    # sample the edge with target node == sampled_nodes_id
    sampled_edges_indices = torch.zeros((2,1), dtype=torch.int32)
    for sampled_node_id in sampled_nodes_id:
        delta=1
        index = sampled_node_id.clone()
        while delta != 0 and delta != -1 and delta != -2: # -1 or -2 to avoid the edge case if no node exists with sampled node id in target
            delta = edges_indices[1, index] - sampled_node_id # target edge in the 2nd row
            index -= delta.item()
        #see if there are duplicate around the index
        indices = []
        for idx in range(index-4, index+4):
            try:
                if edges_indices[1, idx] == sampled_node_id:
                    indices.append(idx)
            except:
                pass
        indices = torch.tensor(indices)
        if len(indices) == 0:
            new_edge_index = torch.tensor([[0], [sampled_node_id]], dtype=torch.int32) # make a connection from 0 as source node target node for higlighing in visualiation
            # sampled_edges_indices = torch.concat((sampled_edges_indices, new_edge_index), dim=-1)
        else:
            sampled_edges_indices = torch.concat((sampled_edges_indices, edges_indices[:,indices]), dim=-1)
    # remove init indices
    sampled_edges_indices = sampled_edges_indices[:,1:]
    return sampled_edges_indices

def test_city_centerlines_graph(city_name:str, city_centerlines_graph:CityCenterlinesGraph):
    edges_indices = city_centerlines_graph.edges_indices
    nodes_id = city_centerlines_graph.nodes_features.nodes_id
    nodes_start_locs = city_centerlines_graph.nodes_features.start_locs
    nodes_end_locs = city_centerlines_graph.nodes_features.end_locs
    source_nodes_start_locs = nodes_start_locs[nodes_id[edges_indices[0,:]]]
    target_nodes_end_locs = nodes_end_locs[nodes_id[edges_indices[1,:]]]
    edges_delta_distance = ((source_nodes_start_locs - target_nodes_end_locs)**2).sum(dim=-1)
    edges_with_issue = torch.nonzero(edges_delta_distance != 0.0).squeeze(1)
    edges_indices_with_issue = edges_indices[:, edges_with_issue]
    edges_delta_distance_with_issue = edges_delta_distance[edges_with_issue]
    if len(edges_with_issue) != 0:
        print('---------')
        print(f'centerline graph edges for {city_name} has problem edges:', edges_indices_with_issue.T.tolist())
        print(f'with corresponding edges deltas in source start and target end locs:', edges_delta_distance_with_issue)
    issues_locs = nodes_end_locs[nodes_id[edges_indices_with_issue[1,:]]]   
    for i, issue_locs in enumerate(issues_locs):
        visualize_city_centerlines_graph(city_name+'_issue_loc_'+str(i), city_centerlines_graph, origin=issue_locs.tolist(), radius=15)

def save_cities_centerlines_graph(cities_centerlines_graph:Dict):
    with open(EXTRACTED_MAP, 'wb') as outfile:
        pickle.dump(cities_centerlines_graph, outfile)
        
if __name__ == '__main__':
    cities_names = ['singapore-onenorth', 'boston-seaport', 'singapore-hollandvillage', 'singapore-queenstown']
    cities_maps_records = get_map_records(NUSCENES_PATH, cities_names)
    config = NuScenesPreprocessConfig
    cities_centerlines_graph = {}
    for city_name in cities_maps_records:
        city_map = cities_maps_records[city_name]
        # fix_centerline_in_nuScenes(city_name, city_map)
        centerlines_nodes_pairs, city_nodes_features = get_centerlines_graph_features(city_map, resolution=config.resolution)
        city_edges_indices = get_centerlines_graph_edges(city_map, centerlines_nodes_pairs)
        city_centerlines_graph = CityCenterlinesGraph(nodes_features = city_nodes_features,
                                                    edges_indices = city_edges_indices)
        # visualize_city_centerlines_graph(city_name, city_centerlines_graph, origin=(660,1105), radius=50)
        # test_city_centerlines_graph(city_name, city_centerlines_graph)
        cities_centerlines_graph[city_name] = city_centerlines_graph 
    save_cities_centerlines_graph(cities_centerlines_graph)

        