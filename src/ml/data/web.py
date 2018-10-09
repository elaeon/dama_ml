import tqdm

class HttpDataset(object):
    def __init__(self, url, sess=None):
        self.url = url
        self.sess = sess

    def download(self, filepath, chunksize):
        response = self.sess.get(self.url)
        with open(filepath, "wb") as f:
            for chunk in tqdm.tqdm(response.iter_content(chunksize)):
                if chunk:
                    f.write(chunk)
                    f.flush()

    def from_data(self, dataset, chunksize=258):        
        self.download(dataset.filepath, chunksize=chunksize)
