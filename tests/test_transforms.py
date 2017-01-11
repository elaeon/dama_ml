import unittest
import numpy as np

from ml.processing import Transforms


def linear(x):
    return x + 1


def linear_p(x, b):
    return x + b


class TestTransforms(unittest.TestCase):
    def setUp(self):
        self.transforms = Transforms()

    def test_json(self):
        self.transforms.add(linear)
        self.transforms.add(linear_p, b=1)
        result = self.transforms.to_json()
        txt = '{"tests.test_transforms.linear": {}, "tests.test_transforms.linear_p": {"b": 1}}'
        self.assertEqual(result, txt)

    def test_from_json(self):
        self.transforms.add(linear)
        self.transforms.add(linear_p, b=1)
        result = self.transforms.to_json()
        t = Transforms.from_json(result)
        self.assertEqual(t.transforms, self.transforms.transforms)

    def test_add_transforms(self):
        self.transforms.add(linear)
        t = Transforms()
        t.add(linear_p, b=1)
        nt = self.transforms + t
        txt = '{"tests.test_transforms.linear": {}, "tests.test_transforms.linear_p": {"b": 1}}'
        self.assertEqual(nt.to_json(), txt)

    def test_apply(self):
        self.transforms.add(linear)
        self.transforms.add(linear_p, b=10)
        numbers = np.ones((10,))
        result = self.transforms.apply(numbers)
        self.assertItemsEqual(result, np.ones((10,)) + 11)


if __name__ == '__main__':
    unittest.main()
