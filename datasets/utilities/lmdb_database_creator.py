from copy import deepcopy
from pathlib import Path
from typing import Tuple, List
import pickle

import lmdb
import zstandard


class LMDBDatabaseCreator:
    def __init__(
            self,
            save_path,
            max_size=5 * 1e9,
            write_frequency=5000,
            use_compressor: bool = True
    ) -> None:
        self.write_frequency = write_frequency

        self.use_compressor = use_compressor

        if self.use_compressor:
            self.compressor = zstandard.ZstdCompressor()
            self.decompressor = zstandard.ZstdDecompressor()

        lmdb_path = Path(save_path)
        lmdb_path.mkdir(parents=True, exist_ok=True)
        lmdb_path = lmdb_path.as_posix()

        self.db = lmdb.open(
            lmdb_path, subdir=True, map_size=max_size,
            readonly=False, meminit=False, map_async=True)

        with self.db.begin(write=False) as txn:
            pickled_log_ids = txn.get(b'__log_ids__')
            pickled_num_samples = txn.get(b'__num_samples__')

        if pickled_log_ids is None:
            self.log_ids = []
        else:
            self.log_ids = self.loads(pickled_log_ids)

        if pickled_num_samples is None:
            self.num_samples = 0
        else:
            self.num_samples = self.loads(pickled_num_samples)

    def write_sample(self, index, sample, key: str = None):
        # Write data to LMDB
        key = index if key is None else key
        data = self.dumps(sample)
        if self.use_compressor:
            data = self.compressor.compress(data)

        with self.db.begin(write=True) as txn:
            txn.put(u'{}'.format(key).encode('ascii'), data)

            if index % self.write_frequency == 0:
                self.db.sync()

    def write_meta_data(self, num_samples):
        # Write meta data (length & keys)
        keys = [u'{}'.format(k).encode('ascii') for k in range(num_samples)]
        with self.db.begin(write=True) as txn:
            txn.put(b'__keys__', self.dumps(keys))
            txn.put(b'__len__', self.dumps(len(keys)))

        self.db.sync()
        self.db.close()
        print("Writing meta data!")

    @staticmethod
    def loads(obj):
        return pickle.loads(obj)

    @staticmethod
    def dumps(obj):
        return pickle.dumps(obj, protocol=5)

    def append_log_id_n_num_samples(self, log_id: str, num_samples: int):
        self.log_ids.append(log_id)
        self.num_samples = num_samples
        with self.db.begin(write=True) as txn:
            txn.put(b'__log_ids__', self.dumps(self.log_ids))
            txn.put(b'__num_samples__', self.dumps(self.num_samples))
        # self.db.sync()

    def get_log_id_n_num_samples(self) -> Tuple[List, int]:
        return deepcopy(self.log_ids), deepcopy(self.num_samples)

    def get_sample(self, key: str):
        with self.db.begin(write=False) as txn:
            data = txn.get(key.encode('ascii'))

        if self.use_compressor:
            data = self.decompressor.decompress(data)

        return self.loads(data)


if __name__ == '__main__':
    lmdb = LMDBDatabaseCreator(
        save_path="/testing/",
        max_size=int(1e6),
        write_frequency=10)

    for i in range(5):
        log_ids_test_1, num_samples_test_1 = lmdb.get_log_id_n_num_samples()
        lmdb.append_log_id_n_num_samples('641eede2d22f429217a40ddf', i+1)
        log_ids_test_2, num_samples_test_2 = lmdb.get_log_id_n_num_samples()
