import unittest
import numpy as np
import csv

from ml.ds import DataSetBuilder, DataSetBuilderFile, DataSetBuilderFold, DataLabel
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

def linear(x, b=0):
    return x + b

class TestDataset(unittest.TestCase):
    def setUp(self):
        NUM_FEATURES = 10
        self.X = np.append(np.zeros((5, NUM_FEATURES)), np.ones((5, NUM_FEATURES)), axis=0)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)

    def tearDown(self):
        pass

    def test_build_dataset_dim_7_1_2(self):
        dataset = DataSetBuilder(
            name="test_ds_0",
            dataset_path="/tmp/",
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        self.assertEqual(dataset.train_labels.shape, (7,))
        self.assertEqual(dataset.validation_labels.shape, (1,))
        self.assertEqual(dataset.test_labels.shape, (2,))
        dataset.destroy()

    def test_build_dataset_dim_5_2_3(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        self.assertEqual(dataset.train_labels.shape, (5,))
        self.assertEqual(dataset.validation_labels.shape, (2,))
        self.assertEqual(dataset.test_labels.shape, (3,))
        dataset.destroy()

    def test_only_labels(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        dataset0, label0 = dataset.only_labels([0])
        self.assertItemsEqual(label0, np.zeros(5))
        dataset1, label1 = dataset.only_labels([1])
        self.assertItemsEqual(label1, np.ones(5))
        dataset.destroy()

    def test_labels_info(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        labels_counter = dataset.labels_info()
        self.assertEqual(labels_counter[0]+labels_counter[1], 10)
        dataset.destroy()

    def test_distinct_data(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        self.assertEqual(dataset.distinct_data() > 0, True)
        dataset.destroy()

    def test_sparcity(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        self.assertEqual(dataset.sparcity() > .3, True)
        dataset.destroy()

    def test_copy(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        ds = dataset.copy(percentaje=.5)
        self.assertEqual(dataset.copy(percentaje=.5).train_data.shape[0], 3)
        dl = dataset.desfragment()
        self.assertEqual(dl.copy(percentaje=.5).data.shape[0], 5)
        ds.destroy()
        dl.destroy()
        dataset.destroy()

    def test_apply_transforms_flag(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        dataset.apply_transforms = True
        copy = dataset.copy()
        #transforms_to_apply = copy.transforms_to_apply
        #self.assertEqual(transforms_to_apply, False)
        self.assertEqual(copy.apply_transforms, dataset.apply_transforms)
        copy.destroy()

        dataset.apply_transforms = False
        copy = dataset.copy()
        #transforms_to_apply = copy.transforms_to_apply
        #self.assertEqual(transforms_to_apply, False)        
        self.assertEqual(copy.apply_transforms, dataset.apply_transforms)
        copy.destroy()
        dataset.destroy()

    def test_apply_tranforms(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True,
            apply_transforms=True)
        

    def test_convert(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        dsb = dataset.convert("convert_test", dtype='float32', ltype='|S1')
        #apply_transforms=False, percentaje=1, applied_transforms=False):
        self.assertEqual(dsb.train_data.dtype, np.dtype('float32'))
        self.assertEqual(dsb.train_labels.dtype, np.dtype('|S1'))
        dsb.destroy()

        dsb = dataset.convert("convert_test", dtype='auto', ltype='auto')
        self.assertEqual(dsb.train_data.dtype, dataset.train_data.dtype)
        self.assertEqual(dsb.train_labels.dtype, dataset.train_labels.dtype)
        dsb.destroy()
        dataset.destroy()

    def test_add_transform(self):
        transforms = Transforms()
        transforms.add(linear, b=1)
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator=None,
            rewrite=True,
            apply_transforms=True,
            transforms=transforms)
        dataset.build_dataset(self.X, self.Y)
        self.assertItemsEqual(self.X[0]+1, dataset.train_data[0])
        transforms = Transforms()
        transforms.add(linear, b=2)
        dsb = dataset.add_transforms(transforms, name="add_transform")
        self.assertItemsEqual(self.X[0]+3, dsb.train_data[0])
        self.assertEqual(dsb.name == "add_transform", True)
        dsb.destroy()
        dataset.destroy()
        
        dataset = DataSetBuilder(
            name="test_ds_0",
            dataset_path="/tmp/",
            ltype='int',
            apply_transforms=False,
            validator=None)
        dataset.build_dataset(self.X, self.Y)
        transforms = Transforms()
        transforms.add(linear, b=1)
        dsb = dataset.add_transforms(transforms)
        self.assertItemsEqual(self.X[0], dsb.train_data[0])
        self.assertEqual(dsb.name != "add_transform", True)
        dsb.destroy()
        dataset.destroy()

    def test_to_df(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        df = dataset.to_df()
        self.assertEqual(list(df.columns), ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'target'])
        dataset.destroy()

    def test_outlayer(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        outlayers = dataset.outlayers()
        dataset.remove_outlayers(list(outlayers))
        dataset.destroy()

    def test_plot(self):
        X = np.random.rand(100, 2)
        Y = np.asarray([0 if .5 < sum(e) <= 1 else -1 if 0 < sum(e) < .5 else 1 for e in X])
        dataset = DataSetBuilder(
            name="test_ds_1",
            dataset_path="/tmp/",
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(X, Y)
        #dataset.plot(view="columns", type_g="violin")
        #dataset.plot(view="rows", type_g="pairplot")
        dataset.plot(view="rows", type_g="scatter")
        dataset.destroy()

    def test_dsb_build_iter(self):
        dsb = DataSetBuilder(name="test", dataset_path="/tmp", chunks=100, dtype="int64")
        shape = (10000, 2)
        step = 0
        range_list = range(0, 15000, 50)
        for init, end in zip(range_list, range_list[1:]):
            iter_ = ((i, i) for i in xrange(init, end))
            step = dsb.build_dataset_from_iter(iter_, shape, "train_data", 
                init=step, type_t=dsb.dtype_t)
        self.assertEqual(dsb.train_data.shape, shape)
        self.assertItemsEqual(dsb.train_data[9999], [9999, 9999])
        dsb.destroy()

    def test_get_set(self):
        from ml.processing import rgb2gray
        transforms = Transforms()
        transforms.add(rgb2gray)
        dsb = DataSetBuilder(name="test", dataset_path="/tmp", chunks=100, 
            author="AGMR", rewrite=True, dtype='float32', transforms=transforms,
            description="description text", train_size=.7, valid_size=.1, 
            validator="cross", compression_level=5, ltype='int',
            apply_transforms = False)
        self.assertEqual(dsb.author, "AGMR")
        self.assertEqual(dsb.dtype, 'float32')
        self.assertEqual(dsb.transforms.to_json(), transforms.to_json())
        self.assertEqual(dsb.description, "description text")
        self.assertEqual(dsb.valid_size, .1)
        self.assertEqual(dsb.train_size, .7)
        self.assertEqual(dsb.test_size, .2)
        self.assertEqual(dsb.validator, 'cross')
        self.assertEqual(dsb.compression_level, 5)
        self.assertEqual(dsb.ltype, 'int')
        self.assertEqual(dsb.dataset_class, 'ml.ds.DataSetBuilder')
        self.assertEqual(type(dsb.timestamp), type(''))
        self.assertEqual(dsb.apply_transforms, False)

        dsb.build_dataset(self.X, self.Y)
        self.assertEqual(type(dsb.md5), type(''))
        dsb.destroy()

    def test_to_libsvm(self):
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
        dataset = DataLabel(
            name="test_ds_1",
            dataset_path="/tmp/",
            ltype='int',
            rewrite=True)
        dataset.build_dataset(X, Y)
        dataset.to_libsvm(name="test.txt", save_to="/tmp")
        check("/tmp/test.txt")
        dataset.destroy()

        dataset = DataSetBuilder(
            name="test_ds_1",
            dataset_path="/tmp/",
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(X, Y)
        dataset.to_libsvm(name="test", save_to="/tmp")
        check("/tmp/test.train.txt")
        check("/tmp/test.test.txt")
        check("/tmp/test.validation.txt")
        dataset.destroy()

    def test_no_data(self):
        from ml.processing import rgb2gray
        transforms = Transforms()
        transforms.add(rgb2gray)
        dsb = DataSetBuilder(name="test", dataset_path="/tmp", chunks=100, 
            author="AGMR", rewrite=True, dtype='float32', transforms=transforms,
            description="description text", train_size=.7, valid_size=.1, 
            validator="cross", compression_level=5, ltype='int',
            apply_transforms=False)

        dsb2 = DataSetBuilder(name="test", dataset_path="/tmp", rewrite=True)
        dsb2.info()
        dsb.destroy()

    def test_to_data(self):
        dataset = DataSetBuilder(
            name="test_ds_1",
            dataset_path="/tmp/",
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        data = dataset.to_data()
        dataset.destroy()
        data.destroy()

    def test_to_datalabel(self):
        dataset = DataSetBuilder(
            name="test_ds_1",
            dataset_path="/tmp/",
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        data = dataset.to_datalabel()
        dataset.destroy()
        data.destroy()

    def test_datalabel_to_data(self):
        dataset = DataLabel(
            name="test_ds_1",
            dataset_path="/tmp/",
            ltype='int',
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        data = dataset.to_data()
        dataset.destroy()
        data.destroy()

    def test_rewrite(self):
        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        self.assertEqual(dataset.validation_labels.shape[0], 2)

        dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.4,
            valid_size=.1,
            ltype='int',
            validator="cross",
            rewrite=True)
        dataset.build_dataset(self.X, self.Y)
        self.assertEqual(dataset.validation_labels.shape[0], 1)

    def test_text_ds(self):
        import h5py
        X = np.asarray([str(line)*10 for line in range(100)])
        ds = Data(name="test", dtype='object')
        ds.build_dataset(X)
        self.assertEqual(ds.shape[0], 100)
        ds.destroy()

        X = np.asarray([(str(line)*10, "1") for line in range(100)])
        ds = Data(name="test", dtype='object')
        ds.build_dataset(X)
        self.assertEqual(ds.shape, (100, 2))
        ds.destroy()

    def test_file_merge(self):
        build_csv_file('/tmp/test_X.csv', self.X, sep=",")
        build_csv_file('/tmp/test_Y.csv', self.Y.reshape(-1, 1), sep="|")

        dbf = DataSetBuilderFile(name="test", 
            training_data_path=['/tmp/test_X.csv', '/tmp/test_Y.csv'], 
                sep=[",", "|"], merge_field="id")
        dbf.build_dataset(labels_column="0_y")
        self.assertEqual(dbf.shape, self.X.shape)
        self.assertEqual(dbf.labels.shape, self.Y.shape)
        dbf.destroy()

    def delete_columns(self):
        X = np.random.rand(100, 10)
        dsb = Data(name="test")
        dsb.build_dataset(X)
        d = dsb.delete_columns([1,5,8])
        print(d.shape)
        dsb.destroy()

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
        dataset = DataSetBuilderFile(
            name="test",
            dataset_path="/tmp/",
            validator="cross")
        data, labels = dataset.from_csv('/tmp/test.csv', 'target')
        self.assertItemsEqual(self.Y, labels.astype(int))


class TestDataSetFold(unittest.TestCase):
    def setUp(self):
        NUM_FEATURES = 10
        self.X = np.append(np.zeros((5, NUM_FEATURES)), np.ones((5, NUM_FEATURES)), axis=0)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)
        self.dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            chunks=2,
            rewrite=True)
        self.dataset.build_dataset(self.X, self.Y)

    def tearDown(self):
        self.dataset.destroy()

    def test_fold(self):
        n_splits = 5
        dsbf = DataSetBuilderFold(n_splits=n_splits)
        dsbf.build_dataset(self.dataset)
        for dsb in dsbf.get_splits():
            self.assertEqual(dsb.shape[0], 10)
            self.assertEqual(dsb.shape[1], 10)
        self.assertEqual(len(dsbf.splits), n_splits)
        dsbf.destroy()


if __name__ == '__main__':
    unittest.main()
