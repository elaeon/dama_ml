import unittest
import numpy as np

from ml.processing import Transforms, Fit


def linear(x, fmtypes=None):
    return x + 1


def linear_p(x, b, fmtypes=None):
    return x + b


def parabole(x, fmtypes=None):
    return x*x


class FitIdent(Fit):
    def fit(self, data, **params):
        def transform(data):
            from ml.utils.numeric_functions import features_fmtype
            from ml import fmtypes
            col_i = features_fmtype(self.fmtypes, fmtypes.NANBOOLEAN)
            data[:, col_i] = np.nan
            return data
        return transform


class TestTransforms(unittest.TestCase):
    def setUp(self):
        pass

    def test_compact(self):
        transforms_0 = Transforms()
        transforms_0.add(linear_p, b=0)
        transforms_1 = Transforms()
        transforms_1.add(linear_p, b=1)
        transforms = transforms_0 + transforms_1
        result = transforms.compact()
        self.assertEqual(result[0].type(), "row")
        self.assertEqual(len(result[0].transforms), 2)

        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=1)
        transforms.add(linear, type="column")
        transforms.add(linear_p, type="column", b=1)
        result = transforms.compact()
        self.assertEqual(result[0].type(), "row")
        self.assertEqual(len(result[0].transforms), 2)
        self.assertEqual(result[1].type(), "column")
        self.assertEqual(len(result[1].transforms), 2)

        transforms.clean()
        transforms.add(linear)
        transforms.add(linear_p, type="column", b=1)
        transforms.add(linear, type="column")
        transforms.add(linear_p, type="row", b=1)
        result = transforms.compact()
        self.assertEqual(result[0].type(), "row")
        self.assertEqual(len(result[0].transforms), 1)
        self.assertEqual(result[1].type(), "column")
        self.assertEqual(len(result[1].transforms), 2)
        self.assertEqual(result[2].type(), "row")
        self.assertEqual(len(result[2].transforms), 1)
        
        transforms.clean()
        transforms.add(linear)
        transforms.add(linear_p, b=1)
        transforms.add(parabole)
        transforms.add(linear_p, type="column", b=1)
        result = transforms.compact()
        self.assertEqual(result[0].type(), "row")
        self.assertEqual(len(result[0].transforms), 3)
        self.assertEqual(result[1].type(), "column")
        self.assertEqual(len(result[1].transforms), 1)

        transforms.clean()
        transforms.add(linear)
        transforms.add(linear_p, b=1)
        transforms.add(parabole)
        result = transforms.compact()
        self.assertEqual(result[0].type(), "row")
        self.assertEqual(len(result[0].transforms), 3)
        self.assertEqual(len(result), 1)

        transforms.clean()
        transforms.add(linear)
        transforms.add(linear_p, type="column", b=1)
        transforms.add(linear_p)
        transforms.add(linear, type="column", b=1)
        result = transforms.compact()
        self.assertEqual(result[0].type(), "row")
        self.assertEqual(len(result[0].transforms), 1)
        self.assertEqual(result[1].type(), "column")
        self.assertEqual(len(result[1].transforms), 1)
        self.assertEqual(result[2].type(), "row")
        self.assertEqual(len(result[2].transforms), 1)
        self.assertEqual(result[3].type(), "column")
        self.assertEqual(len(result[3].transforms), 1)

    def test_json(self):
        from ml.processing import FitTsne
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, type="column", b=1)
        transforms.add(linear_p, b=1)
        transforms.add(FitTsne, name="tsne", type="column")
        result = transforms.to_json()
        txt = '[{"row": {"o_features": null, "transforms": [["tests.test_transforms.linear", {}]]}}, {"column": {"o_features": null, "transforms": [["tests.test_transforms.linear_p", {"b": 1}]]}}, {"row": {"o_features": null, "transforms": [["tests.test_transforms.linear_p", {"b": 1}]]}}, {"column": {"o_features": null, "transforms": [["ml.processing.FitTsne", {"name_00_ml": "tsne"}]]}}]'
        self.assertEqual(result, txt)

        transforms.clean()
        transforms.add(linear)
        transforms.add(linear_p, b=1)        
        transforms.add(linear, type="column")
        result = transforms.to_json()
        txt = '[{"row": {"o_features": null, "transforms": [["tests.test_transforms.linear", {}], ["tests.test_transforms.linear_p", {"b": 1}]]}}, {"column": {"o_features": null, "transforms": [["tests.test_transforms.linear", {}]]}}]'
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
        txt = '[{"row": {"o_features": null, "transforms": [["tests.test_transforms.linear", {}], ["tests.test_transforms.linear_p", {"b": 1}]]}}]'
        self.assertEqual(nt.to_json(), txt)

    def test_apply_row(self):
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=10)
        numbers = np.ones((10,))
        result = transforms.apply(numbers)
        self.assertItemsEqual(result, np.ones((10,)) + 11) # result [12, ..., 12]

    def test_apply_row_iterlayer(self):
        from ml.layers import IterLayer
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=10)
        numbers = IterLayer((e for e in np.ones((10,))), shape=(10,))
        result = transforms.apply(numbers)
        self.assertItemsEqual(result, np.ones((10,)) + 11) # result [12, ..., 12]

    def test_apply_col(self):
        from ml.processing import FitStandardScaler, FitTruncatedSVD
        transforms = Transforms()
        transforms.add(FitStandardScaler, type="column")
        transforms.add(FitTruncatedSVD, type="column", n_components=2)
        numbers = np.random.rand(1000, 3)
        result = np.asarray(list(transforms.apply(numbers)))
        self.assertEqual(-.1 <= result.mean() < .1, True)
        self.assertEqual(.9 <= result.std() <= 1.1, True)
        self.assertEqual(result.shape, (1000, 2))

    def test_apply(self):
        from ml.processing import FitStandardScaler
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=10)
        transforms.add(FitStandardScaler, type="column")
        numbers = np.random.rand(1000, 2)        
        result =  np.asarray(list(transforms.apply(numbers)))
        self.assertEqual(.95 <= result.std() <= 1.05, True)
        self.assertEqual(-0.1 <= result.mean() <= 0.1, True)

    def test_apply_fmtypes(self):
        from ml import fmtypes
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=10)
        transforms.add(FitIdent, type="column")
        numbers = np.empty((1000, 2))
        fmtypes = [fmtypes.BOOLEAN.id, fmtypes.NANBOOLEAN.id]
        result =  np.asarray(list(transforms.apply(numbers, fmtypes=fmtypes)))
        self.assertEqual(all(np.isnan(result[:, 1])), True)

    def test_apply_to_clf(self):
        from ml.ds import DataSetBuilder
        from ml.clf.extended.w_sklearn import RandomForest
        from ml.processing import FitStandardScaler

        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=10)
        transforms.add(FitStandardScaler, type="column")
        X = np.asarray([1, 0]*10)
        Y = X*1
        dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            dtype='int', ltype='int', transforms=transforms, rewrite=True)
        dataset.build_dataset(X, Y)
        classif = RandomForest(dataset=dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/",
            dtype='float64',
            ltype='int')
        classif.train(num_steps=1)
        self.assertEqual(classif.dataset.apply_transforms, True)
        dataset.destroy()
        classif.dataset.destroy()

    def test_transform_col_model(self):
        from ml.ds import DataSetBuilder
        from ml.processing import FitTsne

        transforms = Transforms()
        transforms.add(FitTsne, name="tsne", type="column")
        X = np.random.rand(100, 4)
        Y = X*1
        dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='float64', transforms=transforms, rewrite=True, apply_transforms=True)
        dataset.build_dataset(X, Y)
        shape = dataset.shape
        dataset.info()
        dataset.destroy()
        transforms.destroy()
        self.assertEqual(shape, (100, 6))

    def test_transforms_clf(self):
        from ml.ds import DataSetBuilder
        from ml.processing import FitTsne
        from ml.clf.extended.w_sklearn import RandomForest

        transforms = Transforms()
        transforms.add(FitTsne, name="tsne", type="column")
        X = np.random.rand(1000, 4)
        Y = np.append(np.zeros(500), np.ones(500), axis=0)
        dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='float64', transforms=transforms, rewrite=True, apply_transforms=True)
        dataset.build_dataset(X, Y)
        dataset.info()
        
        classif = RandomForest(
            dataset=dataset,
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        classif.train()
        classif.scores().print_scores()
        classif.destroy()
        dataset.destroy()

    def test_transforms_convert(self):
        from ml.ds import DataSetBuilder
        from ml.processing import FitTsne

        transforms = Transforms()
        transforms.add(FitTsne, name="tsne", type="column")
        X = np.random.rand(1000, 4)
        Y = np.append(np.zeros(500), np.ones(500), axis=0)
        dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='float64', transforms=transforms, rewrite=True, apply_transforms=False)
        dataset.build_dataset(X, Y)
        dsb = dataset.convert(name="test2", apply_transforms=True, ltype="float64")
        shape = dsb.shape
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
        data_nonan = np.array(list(ft.transform(data)))
        self.assertItemsEqual(data_nonan[:,0], [1,0,0,-1])
        self.assertItemsEqual(data_nonan[:,1], [2,2,2.5,2])
        self.assertItemsEqual(data_nonan[:,2], [3,3,3,3])
        self.assertItemsEqual(data_nonan[:,3], [5,5,-1,5])
        self.assertItemsEqual(data_nonan[:,4], [-1,9,9,-1])
        ft.destroy()

    def test_transforms_convert_apply(self):
        from ml.ds import DataSetBuilder
        from ml.processing import FitStandardScaler, FitRobustScaler

        transforms = Transforms()
        transforms.add(FitStandardScaler, name="scaler", type="column")
        X = np.random.rand(1000, 4)
        Y = np.append(np.zeros(500), np.ones(500), axis=0)
        dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='float64', transforms=transforms, rewrite=True, apply_transforms=True)
        dataset.build_dataset(X, Y)
        self.assertEqual(dataset._applied_transforms, True)
        transforms = Transforms()
        transforms.add(FitRobustScaler, name="scaler", type="column")
        dsb = dataset.convert(name="test2", apply_transforms=True, 
            transforms=transforms, ltype="float64", dataset_path="/tmp")
        self.assertEqual(dsb._applied_transforms, True)
        dataset.destroy()
        dsb.destroy()

        dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='float64', transforms=transforms, rewrite=True, apply_transforms=False)
        dataset.build_dataset(X, Y)
        self.assertEqual(dataset._applied_transforms, False)
        transforms = Transforms()
        transforms.add(FitRobustScaler, name="scaler", type="column")
        dsb = dataset.convert(name="test2", apply_transforms=True, 
            transforms=transforms, ltype="float64", dataset_path="/tmp")
        self.assertEqual(dsb._applied_transforms, True)
        dataset.destroy()
        dsb.destroy()

        dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='float64', transforms=transforms, rewrite=True, apply_transforms=True)
        dataset.build_dataset(X, Y)
        dsb = dataset.copy(dataset_path="/tmp")
        self.assertEqual(dsb._applied_transforms, True)
        dataset.destroy()
        dsb.destroy()

        dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='float64', transforms=transforms, rewrite=True, apply_transforms=False)
        dataset.build_dataset(X, Y)
        dsb = dataset.copy(dataset_path="/tmp")
        self.assertEqual(dsb._applied_transforms, False)
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
        transforms.add(FitStandardScaler, name="scaler", type="column")
        transforms.add(linear_p, b=10)
        X = np.random.rand(10, 4)
        result = transforms.apply(X).to_narray()
        self.assertEqual(result.shape, (10, 2))

    def test_transforms_shape(self):
        from ml.processing import drop_columns
        from ml.processing import FitStandardScaler

        transforms = Transforms()
        transforms.add(drop_columns, include_cols=[1,2], o_features=2)
        transforms.add(FitStandardScaler, o_features=2, name="scaler", type="column")
        transforms.add(linear_p, b=10, o_features=2)
        X = np.random.rand(10, 4)
        result = transforms.apply(X)
        self.assertEqual(result.shape, (10, 2))


if __name__ == '__main__':
    unittest.main()
