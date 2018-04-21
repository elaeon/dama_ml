import unittest
import numpy as np

from ml.processing import Transforms, Fit
from ml.layers import IterLayer


def linear(x):
    return x + 1


def linear_2(x):
    x0 = np.empty((x.shape[0], 2), dtype=x.dtype)
    x0[:, :1] = x + 1
    return x0


def linear_3(x):
    x0 = np.empty((x.shape[0], 3), dtype=x.dtype)
    x0[:, :2] = x + 1
    return x0


def linear_1(x):
    x0 = np.empty((x.shape[0], 1), dtype=x.dtype)
    x0[:, 0] = x[:, 0] + 1
    return x0


def linear_p(x, b):
    return x + b


def parabole(x, chunks_size=10):
    if x.shape[0] > chunks_size:
        raise Exception("Array size must be {}, but {} founded".format(chunks_size, x.shape[0]))
    return x*x


def categorical(x):
    for i, e in enumerate(x[:, 0]):
        x[i, 0] = int(e.replace("x", ""))
    return x


def categorical2(x):
    xo = np.empty(x.shape, dtype=np.dtype("float"))
    xo[:,0] = x[:,0]
    for i, e in enumerate(x[:, 1]):
        xo[i, 1] = int(e.replace("x", ""))
    for i, row in enumerate(x[:, 2:]):
        xo[i, 2:] = row.astype(int)
    return xo


def counter_group(data):
    counter = {}
    for row in data:
        key = "-".join(map(str, row))
        if key in counter:
            counter[key] += 1
        else:
            counter[key] = 1
    return counter

    
def add_counter_group(data, counter):
    ndata = np.empty((data.shape[0], data.shape[1] + 1), dtype=data.dtype)
    for i, row in enumerate(data):
        key = "-".join(map(str, row))
        ndata[i] = np.append(row, counter[key])
    return ndata


class TestTransforms(unittest.TestCase):
    def setUp(self):
        pass

    def test_compact(self):
        from ml.processing import FitStandardScaler
        transforms_0 = Transforms()
        transforms_0.add(linear_p, b=0)
        transforms_1 = Transforms()
        transforms_1.add(linear_p, b=1)
        transforms = transforms_0 + transforms_1
        self.assertEqual(transforms.transforms[0].type(), "fn")
        self.assertEqual(len(transforms.transforms), 1)
        self.assertEqual(len(transforms.transforms[0].transforms), 2)

        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=1)
        transforms.add(linear)
        transforms.add(FitStandardScaler, b=1)
        self.assertEqual(len(transforms.transforms), 2)
        self.assertEqual(transforms.transforms[0].type(), "fn")
        self.assertEqual(transforms.transforms[1].type(), "class")
        self.assertEqual(len(transforms.transforms[0].transforms), 3)
        self.assertEqual(len(transforms.transforms[1].transforms), 1)

        transforms.clean()
        transforms.add(linear)
        transforms.add(FitStandardScaler)
        transforms.add(FitStandardScaler)
        transforms.add(linear_p, b=1)
        self.assertEqual(len(transforms.transforms), 3)
        self.assertEqual(transforms.transforms[0].type(), "fn")
        self.assertEqual(len(transforms.transforms[0].transforms), 1)
        self.assertEqual(transforms.transforms[1].type(), "class")
        self.assertEqual(len(transforms.transforms[1].transforms), 2)
        self.assertEqual(transforms.transforms[2].type(), "fn")
        self.assertEqual(len(transforms.transforms[2].transforms), 1)

    def test_json(self):
        from ml.processing import FitTsne
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=1)
        transforms.add(linear_p, b=1)
        transforms.add(FitTsne, name="tsne")
        result = transforms.to_json()
        txt = '[{"fn": {"transforms": [["tests.test_transforms.linear", {}], ["tests.test_transforms.linear_p", {"b": 1}], ["tests.test_transforms.linear_p", {"b": 1}]]}}, {"class": {"transforms": [["ml.processing.FitTsne", {"name_00_ml": "tsne"}]]}}]'
        self.assertEqual(result, txt)

        transforms.clean()
        transforms.add(linear)
        transforms.add(linear_p, b=1)        
        transforms.add(linear)
        result = transforms.to_json()
        txt = '[{"fn": {"transforms": [["tests.test_transforms.linear", {}], ["tests.test_transforms.linear_p", {"b": 1}], ["tests.test_transforms.linear", {}]]}}]'
        self.assertEqual(result, txt)

    def test_from_json(self):
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=1)
        result = transforms.to_json()
        t = Transforms.from_json(result)
        self.assertEqual(t.to_json(), result)

    def test_add_transforms(self):
        t0 = Transforms()
        t0.add(linear)
        t1 = Transforms()
        t1.add(linear_p, b=1)
        nt = t0 + t1
        txt = '[{"fn": {"transforms": [["tests.test_transforms.linear", {}], ["tests.test_transforms.linear_p", {"b": 1}]]}}]'
        self.assertEqual(nt.to_json(), txt)

        t0 = Transforms()
        t0.add(linear)
        t1 = Transforms()
        t1.add(linear_p, b=1)
        nt = t0 + t1
        txt = '[{"fn": {"transforms": [["tests.test_transforms.linear", {}], ["tests.test_transforms.linear_p", {"b": 1}]]}}]'
        self.assertEqual(nt.to_json(), txt)

    def test_apply_row(self):
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=10)
        numbers = np.ones((10, 1))
        result = transforms.apply(numbers)
        self.assertItemsEqual(result.to_narray().reshape(-1), np.ones((10, 1)) + 11) # result [12, ..., 12]

    def test_apply_many_dim(self):
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_2)
        transforms.add(linear_3)
        transforms.add(linear_1)
        numbers = np.ones((10, 1))
        result = transforms.apply(numbers)
        self.assertItemsEqual(result.to_narray().reshape(-1), np.zeros((10, 1)) + 5)

    def test_apply_row_iterlayer(self):
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=10)
        numbers = IterLayer((e for e in np.ones((10,))), shape=(10,))
        result = transforms.apply(numbers)
        self.assertItemsEqual(result.to_memory().reshape(-1), np.ones((10, 1)) + 11) # result [12, ..., 12]

    def test_apply_col(self):
        from ml.processing import FitStandardScaler, FitTruncatedSVD
        transforms = Transforms()
        transforms.add(FitStandardScaler)
        transforms.add(FitTruncatedSVD, n_components=2)
        numbers = np.random.rand(1000, 3)
        result = transforms.apply(numbers).to_narray()
        self.assertEqual(-.1 <= result.mean() < .1, True)
        self.assertEqual(0 <= result.std() <= 1.1, True)
        self.assertEqual(result.shape, (1000, 2))

    def test_apply(self):
        from ml.processing import FitStandardScaler
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=10)
        transforms.add(FitStandardScaler)
        numbers = np.random.rand(1000, 2)        
        result =  transforms.apply(numbers).to_narray()
        self.assertEqual(.95 <= result.std() <= 1.05, True)
        self.assertEqual(-0.1 <= result.mean() <= 0.1, True)

    def test_apply_to_clf(self):
        from ml.ds import DataLabel
        from ml.clf.extended.w_sklearn import RandomForest
        from ml.processing import FitStandardScaler

        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=10)
        transforms.add(FitStandardScaler)
        X = np.random.rand(100, 2)
        Y = (X[:,0] > .5).astype(float)
        dataset = DataLabel(name="test", dataset_path="/tmp/", 
            transforms=transforms, rewrite=True)
        with dataset:
            dataset.build_dataset(X, Y)
        classif = RandomForest( 
            model_name="test", 
            check_point_path="/tmp/")
        classif.set_dataset(dataset)
        classif.train(num_steps=1)
        classif.save(model_version="1")
        with classif.test_ds:
            self.assertEqual(classif.test_ds.apply_transforms, True)
        dataset.destroy()
        classif.destroy()

    def test_bad_input_array(self):
        from ml.ds import DataLabel
        from ml.clf.extended.w_sklearn import RandomForest
    
        X = np.asarray([1, 0]*10)
        Y = X*1
        dataset = DataLabel(name="test", dataset_path="/tmp/", rewrite=True)
        with dataset:
            try:
                dataset.build_dataset(X, Y)
                self.assertEqual(False, True)
            except Exception:
                print("OK")
            dataset.destroy()

    def test_transform_col_model(self):
        from ml.ds import DataLabel
        from ml.processing import FitTsne

        transforms = Transforms()
        transforms.add(FitTsne, name="tsne")
        X = np.random.rand(100, 4)
        Y = X[:,0] > .5
        dataset = DataLabel(name="test", dataset_path="/tmp/", 
            transforms=transforms, rewrite=True, apply_transforms=True)
        with dataset:
            dataset.build_dataset(X, Y)
            shape = dataset.shape
            dataset.info()
        dataset.destroy()
        transforms.destroy()
        self.assertEqual(shape, (100, 6))

    def test_transforms_clf(self):
        from ml.ds import DataLabel
        from ml.processing import FitTsne
        from ml.clf.extended.w_sklearn import RandomForest

        transforms = Transforms()
        transforms.add(FitTsne, name="tsne")
        X = np.random.rand(1000, 4)
        Y = np.append(np.zeros(500), np.ones(500), axis=0)
        dataset = DataLabel(name="test", dataset_path="/tmp/", 
            transforms=transforms, rewrite=True, apply_transforms=True)
        with dataset:
            dataset.build_dataset(X, Y)
            dataset.info()
        
        classif = RandomForest(
            model_name="test", 
            check_point_path="/tmp/")
        classif.set_dataset(dataset)
        classif.train()
        classif.save(model_version="1")
        classif.scores().print_scores()
        transforms.destroy()
        classif.destroy()
        dataset.destroy()

    def test_transforms_convert(self):
        from ml.ds import DataLabel
        from ml.processing import FitTsne

        transforms = Transforms()
        transforms.add(FitTsne, name="tsne")
        X = np.random.rand(1000, 4)
        Y = np.append(np.zeros(500), np.ones(500), axis=0)
        dataset = DataLabel(name="test", dataset_path="/tmp/", 
            transforms=transforms, rewrite=True, apply_transforms=False)
        with dataset:
            dataset.build_dataset(X, Y)
            dsb = dataset.convert(name="test2", apply_transforms=True)
        with dsb:
            shape = dsb.shape
        transforms.destroy()
        dataset.destroy()
        dsb.destroy()
        self.assertEqual(shape, (1000, 6))

    def test_nan_transforms(self):
        from ml.processing import FitReplaceNan
        data = [
            [1,   2,   3,5,   None],
            [0,   None,3,5,   9],
            [0,   2.5, 3,None,9],
            [None,2,   3,5,   None]
        ]
        data = np.array(data, dtype=float)
        ft = FitReplaceNan(data, name="test_nan_fit", path='/tmp/')
        data_nonan = ft.transform(data).to_memory()
        self.assertItemsEqual(data_nonan[:,0], [1,0,0,-1])
        self.assertItemsEqual(data_nonan[:,1], [2,2,2.5,2])
        self.assertItemsEqual(data_nonan[:,2], [3,3,3,3])
        self.assertItemsEqual(data_nonan[:,3], [5,5,-1,5])
        self.assertItemsEqual(data_nonan[:,4], [-1,9,9,-1])
        ft.destroy()

    def test_transforms_convert_apply(self):
        from ml.ds import DataLabel
        from ml.processing import FitStandardScaler, FitRobustScaler

        transforms = Transforms()
        transforms.add(FitStandardScaler, name="scaler")
        X = np.random.rand(1000, 4)
        Y = np.append(np.zeros(500), np.ones(500), axis=0)
        dataset = DataLabel(name="test", dataset_path="/tmp/", 
            transforms=transforms, rewrite=True, apply_transforms=True)
        with dataset:
            dataset.build_dataset(X, Y)
            self.assertEqual(dataset._applied_transforms, True)
        transforms = Transforms()
        transforms.add(FitRobustScaler, name="scaler")
        with dataset:
            dsb = dataset.convert(name="test2", apply_transforms=True, 
                transforms=transforms, dataset_path="/tmp")
        with dsb:
            self.assertEqual(dsb._applied_transforms, True)
        dataset.destroy()
        dsb.destroy()

        dataset = DataLabel(name="test", dataset_path="/tmp/", 
            transforms=transforms, rewrite=True, apply_transforms=False)
        with dataset:
            dataset.build_dataset(X, Y)
            self.assertEqual(dataset._applied_transforms, False)
        transforms = Transforms()
        transforms.add(FitRobustScaler, name="scaler")
        with dataset:
            dsb = dataset.convert(name="test2", apply_transforms=True, 
                transforms=transforms, dataset_path="/tmp")
        with dsb:
            self.assertEqual(dsb._applied_transforms, True)
        dataset.destroy()
        dsb.destroy()

    def test_transforms_drop_cols(self):
        from ml.processing import drop_columns
        transforms = Transforms()
        transforms.add(drop_columns, include_cols=[1,2])
        X = np.random.rand(1000, 4)
        result = transforms.apply(X).to_narray()
        self.assertEqual(result.shape, (1000, 2))

        transforms = Transforms()
        transforms.add(drop_columns, exclude_cols=[3])
        result = transforms.apply(X).to_narray()
        self.assertEqual(result.shape, (1000, 3))
        
    def test_transforms_row_col(self):
        from ml.processing import drop_columns
        from ml.processing import FitStandardScaler

        transforms = Transforms()
        transforms.add(drop_columns, include_cols=[1,2])
        transforms.add(FitStandardScaler, name="scaler")
        transforms.add(linear_p, b=10)
        X = np.random.rand(10, 4)
        result = transforms.apply(X).to_narray()
        self.assertEqual(result.shape, (10, 2))

    def test_transforms_shape(self):
        from ml.processing import drop_columns
        from ml.processing import FitStandardScaler

        transforms = Transforms()
        transforms.add(drop_columns, include_cols=[1,2])
        transforms.add(FitStandardScaler, name="scaler")
        transforms.add(linear_p, b=10)
        X = np.random.rand(10, 4)
        result = transforms.apply(X)
        self.assertEqual(result.shape, (10, 2))

    def test_batch_transforms_row(self):
        X = np.random.rand(100, 4)
        transforms = Transforms()
        transforms.add(parabole, chunks_size=10)
        result = transforms.apply(X, chunks_size=10)

    def test_transform_dtype(self):
        X = np.asarray([
            ["1x", "2", "3", "4"], 
            ["5x", "6x", "7", "8"]], dtype=np.dtype("O"))
        transforms = Transforms()
        transforms.add(categorical)
        transforms.add(categorical2)
        transforms.add(linear_p, b=10)
        transforms.add(parabole)
        result = transforms.apply(X, chunks_size=10)
        data = result.to_narray()
        self.assertItemsEqual(data[0], [121, 144, 169, 196])

    def test_transforms_apply(self):
        from ml.processing import Process
        class P(Process):
            def load(self):
                return self.counter

            def save(self, result):
                self.counter = result

        X = np.asarray([
            [1, 2], 
            [2, 3],
            [3, 4],
            [5, 6],
            [1, 2],
            [2, 3],
            [1, 1],
            [1, 2],
            [1, 2]], dtype=np.dtype("int"))
        it = IterLayer(X, shape=X.shape)
        cg = P(counter_group, name="test")
        cg.apply(it)
        it = IterLayer(X, shape=X.shape)
        self.assertItemsEqual(cg.map(add_counter_group, it)[0], [1,2,4])


if __name__ == '__main__':
    unittest.main()
