import os
import numpy as np

from config.config import NUSCENES_PATH, TARGET_PATH
from config.nuscenes_config import NuScenesPreprocessConfig
from datasets.nuscenes.nuscenes_devkit import NuScenes
from datasets.nuscenes.nuscenes_devkit.eval.prediction.splits import get_prediction_challenge_split
from datasets.utilities.lmdb_database_creator import LMDBDatabaseCreator
from datasets.nuscenes.data_preprocessing.sample_creation import SampleCreator

class TargetDataCreator:
    def __init__(
            self,
            data_set: str,
            preprocess_config: NuScenesPreprocessConfig,
    ) -> None:
        """
        Create data structure for data generator from the prediction challenge tokens
         - token as input -> find the logs from the extracted scenes
                             and create the trajectory data structure
         - extract map not translated to allow more efficient roation

        @param data_set (str): nuScenes dataset (e.g. 'v1.0-trainval' or 'v1.0-mini')
        @param preprocess_config (NuScenesPreprocessConfig): Preprocessing config to
                                create TrainingSamples
        """
        # load data
        self.data_set = data_set
        self.data_root = NUSCENES_PATH
        self.nuscence = NuScenes(version=data_set, dataroot=self.data_root, verbose=True)
        self.preprocess_config = preprocess_config

    def create_training_data_challenge_split_lmdb(self, data_split: str, save_path: str) -> None:
        """
        Sample data creation and saving
        :param data_split:  e.g. 'val' (test_dat),  'train_val'
        :param (str) save_path:  Path to save samples
        :return:
        """
        data_tokens = get_prediction_challenge_split(data_split, dataroot=self.data_root)

        data_split_save_path = os.path.join(save_path, data_split)
        if not os.path.exists(data_split_save_path):
            os.makedirs(data_split_save_path)

        sample_creator = self._create_sample_creator(data_split=data_split)

        max_size = len(data_tokens) * 300_000
        lmdb_db_creator = LMDBDatabaseCreator(data_split_save_path, max_size=max_size, use_compressor=True)

        for num, token in enumerate(data_tokens):
            print(num)
            sample_scene = sample_creator.create_sample_from_token(token)
            # sample_creator.visualize_input(sample_scene)
            lmdb_db_creator.write_sample(num, sample_scene)
        lmdb_db_creator.write_meta_data(len(data_tokens))
    
    def _create_sample_creator(self, data_split: str) -> SampleCreator:
        sample_creator = SampleCreator(
            nuscenes_preprocess_config=self.preprocess_config,
            nuscenes=self.nuscence,
            data_split=data_split,
            data_set=self.data_set)
        return sample_creator


if __name__ == "__main__":
    test_preprocess_config = NuScenesPreprocessConfig()
    data_split_map = {'mini_train': 'v1.0-mini',
                      'mini_val': 'v1.0-mini',
                      'val': 'v1.0-trainval', 
                      'train_val': 'v1.0-trainval', 
                      'train': 'v1.0-trainval'}
    for data_split, data_set in data_split_map.items():
        data_preparation = TargetDataCreator(
            data_set=data_set, preprocess_config=test_preprocess_config)
        data_preparation.create_training_data_challenge_split_lmdb(
            data_split=data_split, save_path=TARGET_PATH)
        
        del data_preparation