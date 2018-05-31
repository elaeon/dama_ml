import unittest
import numpy as np
import csv

from ml.ds import DataLabelSetFile, DataLabelFold, DataLabel
from ml.ds import Data
from ml.processing import Transforms


def build_csv_file(path, vector, sep=","):
    with open(path, 'w') as f:
        header = ["id"] + map(str, list(range(vector.shape[-1])))
        f.write(sep.join(header))
        f.write("\n")
        for i, row in enumerate(vector):
            f.write(str(i)+sep)
            f.write(sep.join(map(str, row)))
            f.write("\n")


def linear(x, o_features=None, b=0):
    return x + b


def parabole(x, o_features=None, b=0):
    return x**2 + b


def to_int(x, col=None, o_features=None):
    x[:, col] = x[:, col].astype(np.int)
    return x


def label_t(x, o_features=None):
    return np.log1p(x)


class TestDataset(unittest.TestCase):
    def setUp(self):
        NUM_FEATURES = 10
        self.X = np.append(np.zeros((5, NUM_FEATURES)), np.ones((5, NUM_FEATURES)), axis=0).astype(float)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)

    def tearDown(self):
        pass

    def test_from_data_dim_7_1_2(self):
        dataset = DataLabel(
            name="test_ds_0",
            dataset_path="/tmp/",
            rewrite=True)
        with dataset:
            dataset.from_data(self.X, self.Y)
            X_train, X_validation, X_test, y_train, y_validation, y_test = dataset.cv()
            self.assertEqual(y_train.shape, (7,))
            self.assertEqual(y_validation.shape, (1,))
            self.assertEqual(y_test.shape, (2,))
        dataset.destroy()

    def test_from_data_dim_5_2_3(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            rewrite=True)
        with dataset:
            dataset.from_data(self.X, self.Y)
            X_train, X_validation, X_test, y_train, y_validation, y_test = dataset.cv(train_size=.5, valid_size=.2)
            self.assertEqual(y_train.shape, (5,))
            self.assertEqual(y_validation.shape, (2,))
            self.assertEqual(y_test.shape, (3,))
        dataset.destroy()

    def test_only_labels(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            rewrite=True)
        with dataset:
            dataset.from_data(self.X, self.Y)
            dataset0, label0 = dataset.only_labels([0])
            self.assertItemsEqual(label0, np.zeros(5))
            dataset1, label1 = dataset.only_labels([1])
            self.assertItemsEqual(label1, np.ones(5))
        dataset.destroy()

    def test_labels_info(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            rewrite=True)
        with dataset:
            dataset.from_data(self.X, self.Y)
            labels_counter = dataset.labels_info()
            self.assertEqual(labels_counter[0]+labels_counter[1], 10)
        dataset.destroy()

    def test_distinct_data(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            rewrite=True)
        with dataset:
            dataset.from_data(self.X, self.Y)
            self.assertEqual(dataset.distinct_data() > 0, True)
        dataset.destroy()

    def test_sparcity(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            rewrite=True)
        with dataset:
            dataset.from_data(self.X, self.Y)
            self.assertEqual(dataset.sparcity() > .3, True)
        dataset.destroy()

    def test_copy(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            rewrite=True)
        with dataset:
            dataset.from_data(self.X, self.Y)
            ds = dataset.convert("test_convert", percentaje=.5, dataset_path="/tmp")

        with ds:
            self.assertEqual(ds.data.shape[0], 5)

        ds.destroy()
        dataset.destroy()

    def test_apply_transforms_flag(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            rewrite=True)
        with dataset:
            dataset.from_data(self.X, self.Y)
            dataset.apply_transforms = True
            copy = dataset.convert("test_2", apply_transforms=False, dataset_path="/tmp/")
        with copy:
            self.assertEqual(copy.apply_transforms, False)
        copy.destroy()
        dataset.destroy()

    def test_convert_percentaje(self):
        with DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            rewrite=True) as dataset:
            dataset.from_data(self.X, self.Y)
            dsb = dataset.convert("convert_test", dataset_path="/tmp/", percentaje=.5)
        with dsb:
            self.assertEqual(round(self.X.shape[0]/2,0), dsb.data.shape[0])
            self.assertEqual(round(self.Y.shape[0]/2,0), dsb.labels.shape[0])
        dsb.destroy()

    def test_convert_transforms_true(self):
        o_features = self.X.shape[1]
        transforms = Transforms()
        transforms.add(linear, b=1, o_features=o_features)
        dataset = DataLabel(name="test_ds", dataset_path="/tmp/",
            rewrite=True, transforms=transforms, apply_transforms=True)
        
        with dataset:
            dataset.from_data(self.X, self.Y)
            transforms = Transforms()
            transforms.add(parabole, o_features=o_features)
            dsb = dataset.convert("convert_test", dataset_path="/tmp/",
                                transforms=transforms, apply_transforms=True)

        with dsb:
            self.assertEqual(len(dsb.transforms.transforms), 1)
            self.assertEqual(len(dsb.transforms.transforms[0].transforms), 2)

        dsb.destroy()
        dataset.destroy()

    def test_convert_data_transforms_true(self):
        o_features = self.X.shape[1]
        transforms = Transforms()
        transforms.add(linear, b=1, o_features=o_features)
        dataset = Data(name="test_ds", dataset_path="/tmp/",
            rewrite=True, transforms=transforms, apply_transforms=True)
        
        with dataset:
            dataset.from_data(self.X)
            transforms = Transforms()
            transforms.add(parabole, o_features=o_features)
            dsb = dataset.convert("convert_test", dataset_path="/tmp/",
                                transforms=transforms, apply_transforms=True)

        with dsb:
            self.assertEqual(len(dsb.transforms.transforms), 1)
            self.assertEqual(len(dsb.transforms.transforms[0].transforms), 2)

        dsb.destroy()
        dataset.destroy()

    def test_convert_transforms_false(self):
        o_features = self.X.shape[1]
        transforms = Transforms()
        transforms.add(linear, b=1, o_features=o_features)
        dataset = DataLabel(name="test_ds", dataset_path="/tmp/",
            rewrite=True, transforms=transforms, apply_transforms=False)
        
        transforms = Transforms()
        transforms.add(parabole, o_features=o_features)
        with dataset:
            dataset.from_data(self.X, self.Y)
            dsb = dataset.convert("convert_test", dataset_path="/tmp/",
                                transforms=transforms, apply_transforms=False)

        with dsb:
            self.assertEqual(len(dsb.transforms.transforms), 1)
            self.assertEqual(len(dsb.transforms.transforms[0].transforms), 2)
        dsb.destroy()
        dataset.destroy()


    def test_add_transform(self):
        transforms = Transforms()
        transforms.add(linear, b=1, o_features=self.X.shape[1])
        with DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            rewrite=True,
            apply_transforms=False,
            transforms=transforms) as dataset:
            dataset.from_data(self.X, self.Y)
            transforms = Transforms()
            transforms.add(linear, b=2, o_features=self.X.shape[1])
            dsb = dataset.convert("add_transform", transforms=transforms, 
                                apply_transforms=True)
        with dsb:
            #self.assertItemsEqual(self.X[0]+3, dsb.data[0])
            self.assertEqual(dsb.name == "add_transform", True)
        dsb.destroy()
        dataset.destroy()

    def test_to_df(self):
        with DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            rewrite=True) as dataset:
            dataset.from_data(self.X, self.Y, 10)
            df = dataset.to_df()
        self.assertEqual(list(df.columns), ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'target'])
        dataset.destroy()
        #with Data(
        #    name="test_ds",
        #    dataset_path="/tmp/",
        #    rewrite=True) as dataset:
        #    dataset.from_data(self.X)
        #    df = dataset.to_df()
        #self.assertEqual(list(df.columns), ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
        #dataset.destroy()

    def test_ds_build(self):
        X = np.asarray([
            [1,2,3,4,5,6],
            [6,5,4,3,2,1],
            [0,0,0,0,0,0],
            [-1,0,-1,0,-1,0]], dtype=np.float)
        dl = Data(name="test", dataset_path="/tmp", rewrite=True)
        with dl:
            dl.from_data(X)
            self.assertItemsEqual(dl.data[0], X[0])
            self.assertItemsEqual(dl.data[1], X[1])
            self.assertItemsEqual(dl.data[2], X[2])
            self.assertItemsEqual(dl.data[3], X[3])
        dl.destroy()

    def test_ds_dtype(self):
        X = np.asarray([[1,2,3,4,5]], dtype=np.int)
        Y = np.asarray(['1','2','3','4','5'], dtype=str)
        dl = DataLabel(name="test", dataset_path="/tmp", rewrite=True)
        with dl:
            dl.from_data(X, Y)
            self.assertEqual(dl.dtype, X.dtype)
            self.assertEqual(dl.ltype, Y.dtype)
        dl.destroy()

    def test_get_set(self):
        from ml.processing import rgb2gray
        transforms = Transforms()
        transforms.add(rgb2gray)
        with DataLabel(name="test", dataset_path="/tmp",
            author="AGMR", rewrite=True, transforms=transforms,
            description="description text", compression_level=5, 
            apply_transforms=False) as dsb:
            self.assertEqual(dsb.author, "AGMR")
            self.assertEqual(dsb.transforms.to_json(), transforms.to_json())
            self.assertEqual(dsb.description, "description text")
            self.assertEqual(dsb.compression_level, 5)
            self.assertEqual(dsb.dataset_class, 'ml.ds.DataLabel')
            self.assertEqual(type(dsb.timestamp), type(''))
            self.assertEqual(dsb.apply_transforms, False)
            self.assertEqual(dsb.hash_header is not None, True)

            dsb.from_data(self.X.astype('float32'), self.Y)
            self.assertEqual(dsb.md5 is not None, True)
            dsb.destroy()

    def test_to_libsvm(self):
        from ml.utils.files import rm
        def check(path):
            with open(path, "r") as f:
                row = f.readline()
                row = row.split(" ")
                self.assertEqual(row[0] in ["0", "1", "2"], True)
                self.assertEqual(len(row), 3)
                elem1 = row[1].split(":")
                elem2 = row[2].split(":")
                self.assertEqual(int(elem1[0]), 1)
                self.assertEqual(int(elem2[0]), 2)
                self.assertEqual(2 == len(elem1) == len(elem2), True)

        X = np.random.rand(100, 2)
        Y = np.asarray([0 if .5 < sum(e) <= 1 else -1 if 0 < sum(e) < .5 else 1 for e in X])
        with DataLabel(
            name="test_ds_1",
            dataset_path="/tmp/",
            rewrite=True) as dataset:
            dataset.from_data(X, Y)
            dataset.to_libsvm(name="test.txt", save_to="/tmp")
            check("/tmp/test.txt")
            dataset.destroy()
        rm("/tmp/test.txt")

    def test_no_data(self):
        from ml.processing import rgb2gray
        transforms = Transforms()
        transforms.add(rgb2gray)
        dsb = DataLabel(name="test", dataset_path="/tmp",
            author="AGMR", rewrite=True, transforms=transforms,
            description="description text", compression_level=5,
            apply_transforms=False)
        dsb.md5 = ""
        timestamp = dsb.timestamp

        dsb2 = DataLabel(name="test", dataset_path="/tmp", rewrite=False)
        self.assertEqual(dsb2.author, "AGMR")
        self.assertEqual(dsb2.hash_header is not None, True)
        self.assertEqual(dsb2.description, "description text")
        self.assertEqual(dsb2.timestamp, timestamp)
        self.assertEqual(dsb2.compression_level, 5)
        self.assertEqual(dsb2.dataset_class, "ml.ds.DataLabel")
        dsb.destroy()

    def test_to_data(self):
        with DataLabel(
            name="test_ds_1",
            dataset_path="/tmp/",
            rewrite=True) as dataset:
            dataset.from_data(self.X, self.Y)
            data = dataset.to_data()
        with data:
            self.assertEqual(data.shape, (10, 10))
        dataset.destroy()
        data.destroy()

    def test_datalabel_to_data(self):
        with DataLabel(name="test_ds_1", dataset_path="/tmp/", rewrite=True) as dataset:
            dataset.from_data(self.X, self.Y)
            data = dataset.to_data()
        with data:
            self.assertEqual(data.shape, (10, 10))
        dataset.destroy()
        data.destroy()

    def test_text_ds(self):
        X = np.asarray([(str(line)*10, "1") for line in range(100)], dtype=np.dtype("O"))
        with Data(name="test", dataset_path="/tmp/", rewrite=True) as ds:
            ds.from_data(X)
            self.assertEqual(ds.shape, (100, 2))
            self.assertEqual(ds.dtype, X.dtype)
            ds.destroy()

    def test_file_merge(self):
        build_csv_file('/tmp/test_X.csv', self.X, sep=",")
        build_csv_file('/tmp/test_Y.csv', self.Y.reshape(-1, 1), sep="|")

        with DataLabelSetFile(
            training_data_path=['/tmp/test_X.csv', '/tmp/test_Y.csv'], 
                sep=[",", "|"], merge_field="id", dataset_path="/tmp/") as dbf:
            dbf.from_data(labels_column="0_y")
            self.assertEqual(dbf.shape, self.X.shape)
            self.assertEqual(dbf.labels.shape, self.Y.shape)
            dbf.destroy()

    def test_fmtypes(self):
        with Data(name="test", dataset_path="/tmp/") as data:
            data.from_data(self.X)
            columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            data.columns = columns
            self.assertEqual(data.columns.shape[0], self.X.shape[1])
            self.assertItemsEqual(data.columns[:], columns)
            data.destroy()

    def test_fmtypes_set_columns(self):
        with Data(name="test", dataset_path="/tmp/") as data:
            data.from_data(self.X)
            columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            data.columns = columns
            data.columns[2] = 'X'
            self.assertEqual(data.columns[2], 'X')
            data.destroy()

    def test_fmtypes_empty(self):
        with Data(name="test", dataset_path="/tmp/") as data:
            data.from_data(self.X)
            self.assertItemsEqual(data.columns, [u'c0', u'c1', u'c2', u'c3', 
                u'c4', u'c5', u'c6', u'c7', u'c8', u'c9'])
            data.destroy()

    def test_features_fmtype(self):
        from ml import fmtypes
        with Data(name="test", dataset_path="/tmp/") as data:
            array = np.asarray([
                [0, 1, -1, 3, 4, 0],
                [1, -1, 0, 2 ,5, 1],
                [0, 0, 1, 2, 2, 1],
                [0, 1, 1, 3, 6, 0],
                [1, 1, 0, 7, 7, 1]
            ], dtype=int)
            data.from_data(array)
            data.columns = ['a', 'b', 'c', 'd', 'e', 'f']
            #print(data.features_fmtype(fmtypes.BOOLEAN))
            #self.assertEqual(data.features_fmtype(fmtypes.BOOLEAN), [0, 5])
            #self.assertEqual(data.features_fmtype(fmtypes.NANBOOLEAN), [1, 2, 3])
            #self.assertEqual(data.features_fmtype(fmtypes.ORDINAL), [4])
            data.destroy()

    def test_features_fmtype_set(self):
        from ml import fmtypes
        from ml.processing import Transforms
        from ml.layers import IterLayer

        array = np.asarray([
            [0, 1, -1, 1, '4', 0],
            [1, -1, 0, 2, '5', 1],
            [0, 0, 1, 4, '2', 1],
            [0, 1, 1, 3, '6', 0],
            [1, 1, 0, 7, '7', 1]
        ], dtype=np.dtype("int"))
        t = Transforms()
        t.add(to_int, col=4)
        fmtypes_t = fmtypes.FmtypesT()
        fmtypes_t.add(0, fmtypes.BOOLEAN)
        fmtypes_t.add(2, fmtypes.NANBOOLEAN)
        fmtypes_t.add(1, fmtypes.NANBOOLEAN)
        fmtypes_t.add(5, fmtypes.BOOLEAN)
        fmtypes_t.add(4, fmtypes.ORDINAL)
        fmtypes_t.fmtypes_fill(6)
        with Data(name="test", dataset_path="/tmp/",
                    transforms=t, apply_transforms=True, rewrite=True) as data:
            data.from_data(array, chunks_size=2)
            #self.assertEqual(data.features_fmtype(fmtypes.BOOLEAN), [0, 5])
            #self.assertEqual(data.features_fmtype(fmtypes.NANBOOLEAN), [1, 2])
            #self.assertEqual(data.features_fmtype(fmtypes.ORDINAL), [3, 4])
        data.destroy()

    def test_features_fmtype_edit(self):
        from ml import fmtypes
        with Data(name="test", dataset_path="/tmp/") as data:
            array = np.asarray([
                [0, 1, -1, 3, 4, 0],
                [1, -1, 0, 2 ,5, 1],
                [0, 0, 1, 2, 2, 1],
                [0, 1, 1, 3, 6, 0],
                [1, 1, 0, 7, 7, 1]
            ], dtype=int)
            data.from_data(array)
            #data.set_fmtypes(3, fmtypes.DENSE)
            #data.set_fmtypes(4, fmtypes.DENSE)
            #self.assertItemsEqual(data.fmtypes[:], 
            #    [fmtypes.BOOLEAN.id, fmtypes.NANBOOLEAN.id, fmtypes.NANBOOLEAN.id, 
            #    fmtypes.DENSE.id, fmtypes.DENSE.id, fmtypes.BOOLEAN.id])
            data.destroy()

    def test_rewrite_data(self):
        with Data(name="test", dataset_path="/tmp/") as data:
            array = np.zeros((10, 2))
            data.from_data(array)
            data.data[:, 1] = np.ones((10))
            self.assertItemsEqual(data.data[:, 1], np.ones((10)))
            self.assertItemsEqual(data.data[:, 0], np.zeros((10)))

        with Data(name="test", dataset_path="/tmp/", rewrite=False) as data:
            data.info()
            data.destroy()

    def test_cv_ds(self):
        dl = DataLabel(name="test", dataset_path="/tmp/")
        with dl:
            dl.from_data(self.X, self.Y)
            train_ds, validation_ds, test_ds = dl.cv_ds(train_size=.6, valid_size=.2)
        with train_ds:            
            self.assertEqual(train_ds.shape, (6, 10))
        with validation_ds:
            self.assertEqual(validation_ds.shape, (2, 10))
        with test_ds:
            self.assertEqual(test_ds.shape, (2, 10))

        dl.destroy()
        train_ds.destroy()
        validation_ds.destroy()
        test_ds.destroy()

    def test_cv_unbalanced(self):
        X = np.random.rand(1000, 2)
        Y = np.asarray([str(e) for e in (X[:, 1] < .5)], dtype="|O")
        ds = DataLabel(name="test", dataset_path="/tmp/")
        with ds:
            ds.from_data(X, Y)
            X_train, X_validation, X_test, y_train, y_validation, y_test = ds.cv(train_size=.7, valid_size=0, unbalanced={u'True': .2, u'False': 350})
        counter = np.unique(Y, return_counts=True)
        un = np.unique(y_test, return_counts=True)
        self.assertEqual(np.unique(y_train, return_counts=True)[1][1] - 4 <= round(counter[1][1]*.7*.2, 0), True)
        ds.destroy()

    def test_labels_transforms(self):
        transforms = Transforms()
        transforms.add(label_t, o_features=1)
        dl = DataLabel(name="test", dataset_path="/tmp/")
        X = np.random.rand(10, 1)
        Y_0 = np.random.randint(1, 10, size=(10, 1))
        Y = transforms.apply(Y_0, chunks_size=0).to_narray()
        with dl:
            dl.from_data(X, Y)
            self.assertEqual(dl.labels[0], np.log1p(Y_0[0]))
        dl.destroy()


class TestDataSetFile(unittest.TestCase):
    def setUp(self):
        NUM_FEATURES = 10
        self.X = np.append(np.zeros((5, NUM_FEATURES)), np.ones((5, NUM_FEATURES)), axis=0)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)
        dataset = np.c_[self.X, self.Y]
        with open('/tmp/test.csv', 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(map(str, range(10)) + ['target']) 
            for row in dataset:
                csv_writer.writerow(row)

    def test_load(self):
        dataset = DataLabelSetFile(
            name="test",
            dataset_path="/tmp/")
        with dataset:
            data, labels = dataset.from_csv('/tmp/test.csv', 'target')
            self.assertItemsEqual(self.Y, labels.astype(int))
        dataset.destroy()


class TestDataSetFold(unittest.TestCase):
    def setUp(self):
        NUM_FEATURES = 10
        self.X = np.append(np.zeros((5, NUM_FEATURES)), np.ones((5, NUM_FEATURES)), axis=0)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)
        self.dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            rewrite=True)
        with self.dataset:
            self.dataset.from_data(self.X, self.Y)

    def tearDown(self):
        self.dataset.destroy()

    def test_fold(self):
        n_splits = 5
        dsbf = DataLabelFold(n_splits=n_splits, dataset_path="/tmp")
        dsbf.from_data(self.dataset)
        for dsb in dsbf.get_splits():
            with dsb:
                self.assertEqual(dsb.shape[0], 8)
                self.assertEqual(dsb.shape[1], 10)
        self.assertEqual(len(dsbf.splits), n_splits)
        dsbf.destroy()


if __name__ == '__main__':
    unittest.main()
