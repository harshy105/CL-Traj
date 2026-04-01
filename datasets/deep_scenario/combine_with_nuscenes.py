import os
from tqdm import tqdm

from datasets.utilities.lmdb_database_creator import LMDBDatabaseCreator
from datasets.utilities.lmdb_loader import LMDBLoader

if __name__ == '__main__':
    splits = ['train',] # get the train_val from nuscenes
    dataset_ds_path = "" # deepScenario processed folder
    dataset_nu_path = "" # nuScenes processed folder
    new_datset_path = "" # combined processed folder
    for split in splits:
        dataset_ds_split_path = os.path.join(dataset_ds_path, split)
        dataset_nu_split_path = os.path.join(dataset_nu_path, split)
        new_datset_split_path = os.path.join(new_datset_path, split)
        lmdb_database_ds = LMDBLoader(data_path=dataset_ds_split_path)
        lmdb_database_nu = LMDBLoader(data_path=dataset_nu_split_path)
        new_lmdb_database_split = LMDBDatabaseCreator(save_path=new_datset_split_path, max_size=int(5e10), write_frequency=20)

        num_samples = 0
        print(f'----Saving {split} from DeepScenario-----')
        for sample in tqdm(lmdb_database_ds):
            new_lmdb_database_split.write_sample(num_samples, sample) 
            num_samples += 1

        print(f'----Saving {split} from nuScenes-----')
        for sample in tqdm(lmdb_database_nu):
            new_lmdb_database_split.write_sample(num_samples, sample) 
            num_samples += 1

        new_lmdb_database_split.write_meta_data(num_samples=num_samples)
        print('----Done----')