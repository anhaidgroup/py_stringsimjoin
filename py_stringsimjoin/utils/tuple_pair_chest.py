DEFAULT_AVAILABLE_MEMORY = 1e9
import pandas as pd
import sys


class TuplePairChest(object):
    def __init__(self,
                 file_name=None,
                 header=None,
                 mem_size=DEFAULT_AVAILABLE_MEMORY,
                 append_to_file=False):
        self.file_name = file_name
        self.header = header
        self.mem_size = mem_size
        self.append_to_file = append_to_file
        self.store = []

    def preprocess(self):
        if not self.append_to_file:
            if self.file_name is not None:
                self.create_file()
            if self.header is not None:
                self.write_header()

    def create_file(self):
        with open(self.file_name, 'w'):
            pass

    def write_header(self):
        df = pd.DataFrame(columns=self.header)
        df.to_csv(self.file_name, mode='a', index=False)

    def flush_chest_to_disk(self):
        df = pd.DataFrame(self.store)
        df.to_csv(self.file_name, mode='a', index=False, header=False)

    def get_dataframe_from_chest(self):
        df = pd.DataFrame(self.store, columns=self.header)
        return df

    def append(self, obj):
        self.store.append(obj)
        if self.file_name != None:
            if self.nbytes(self.store) > self.mem_size:
                self.flush_chest_to_disk()
                self.store = []

    def postprocess(self):
        if self.file_name != None:
            self.flush_chest_to_disk()
            self.store = []
            return True
        else:
            df = self.get_dataframe_from_chest()
            return df

    def nbytes(self, o):
        """ Number of bytes of an object
        >>> nbytes(123)  # doctest: +SKIP
        24
        >>> nbytes('Hello, world!')  # doctest: +SKIP
        50
        >>> import numpy as np
        >>> nbytes(np.ones(1000, dtype='i4'))
        4000
        """
        if hasattr(o, 'nbytes'):
            return o.nbytes
        n = str(type(o))
        if 'pandas' in n and ('DataFrame' in n or 'Series' in n):
            return sum(b.values.nbytes * (10 if b.values.dtype == 'O' else 1)
                       for b in o._data.blocks)  # pragma: no cover
        else:
            return sys.getsizeof(o)
