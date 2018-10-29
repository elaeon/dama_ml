import hashlib
import numpy as np

class Hash:
    def __init__(self, hash_fn:str='sha1'):
        self.hash_fn = hash_fn
        self.hash = getattr(hashlib, hash_fn)()

    def chunks(self, it):
        for chunk in it:
            self.hash.update(chunk)

    def __str__(self):
        return "${hash_fn}${digest}".format(hash_fn=self.hash_fn, digest=self.hash.hexdigest())


def unique_dtypes(dtypes):
    return np.unique([dtype.name for _, dtype in dtypes])