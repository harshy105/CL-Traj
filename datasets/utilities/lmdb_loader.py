import os
import lmdb
import pickle
import zstandard


class LMDBLoader:
    def __init__(
            self,
            data_path: str,
            use_compressor: bool = True
    ) -> None:
        self.db_path = data_path
        self.use_compressor = use_compressor

        if use_compressor:
            self.decompressor = zstandard.ZstdDecompressor()

        self.env = lmdb.open(
            self.db_path, subdir=os.path.isdir(self.db_path),
            readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = self.loads(txn.get(b'__len__'))
            self.keys = self.loads(txn.get(b'__keys__'))

    def __getitem__(self, index: int):
        with self.env.begin(write=False) as txn:
            data = txn.get(self.keys[index])

        if self.use_compressor:
            data = self.decompressor.decompress(data)

        return self.loads(data)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    @staticmethod
    def loads(obj):
        return pickle.loads(obj)

    @staticmethod
    def dumps(obj):
        return pickle.dumps(obj, protocol=5)
