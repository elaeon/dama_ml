import unittest
import numpy as np

from ml.processing import Transforms


def linear(x):
    return x + 1


def linear_p(x, b):
    return x + b


def parabole(x):
    return x*x


class TestTransforms(unittest.TestCase):
    def setUp(self):
        pass

    def test_compact(self):
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
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, type="column", b=1)
        transforms.add(linear_p, b=1)
        result = transforms.to_json()
        txt = '[{"row": {"tests.test_transforms.linear": {}}}, {"column": {"tests.test_transforms.linear_p": {"b": 1}}}, {"row": {"tests.test_transforms.linear_p": {"b": 1}}}]'
        self.assertEqual(result, txt)

        transforms.clean()
        transforms.add(linear)
        transforms.add(linear_p, b=1)        
        transforms.add(linear, type="column")
        result = transforms.to_json()
        txt = '[{"row": {"tests.test_transforms.linear": {}, "tests.test_transforms.linear_p": {"b": 1}}}, {"column": {"tests.test_transforms.linear": {}}}]'
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
        txt = '[{"row": {"tests.test_transforms.linear": {}, "tests.test_transforms.linear_p": {"b": 1}}}]'
        self.assertEqual(nt.to_json(), txt)

    def test_apply_row(self):
        transforms = Transforms()
        transforms.add(linear)
        transforms.add(linear_p, b=10)
        numbers = np.ones((10,))
        result = transforms.apply(numbers)
        self.assertItemsEqual(result, np.ones((10,)) + 11) # result [12, ..., 12]

    def test_apply_col(self):
        from ml.processing import FitStandardScaler
        transforms = Transforms()
        base_numbers = np.random.rand(1000, 2)
        transforms.add(FitStandardScaler, type="column")
        numbers = np.random.rand(1000, 2)
        result = transforms.apply(numbers, base_data=base_numbers)
        self.assertEqual(-.1 <= result.mean() < .1, True)
        self.assertEqual(.9 <= result.std() <= 1.1, True)

if __name__ == '__main__':
    unittest.main()
