import unittest
import numpy as np
from ml.utils.basic import Shape


# def test_add(self):
#    x = np.random.rand(10, 2)
#    y = np.random.rand(10)
#    x_train = StructArray([("x", x)])
#    y_train = StructArray([("y", y)])
#    xy_train = x_train + y_train
#    self.assertEqual((xy_train["x"].to_ndarray() == x).all(), True)
#    self.assertEqual((xy_train["y"].to_ndarray() == y).all(), True)


class TestShape(unittest.TestCase):

    def test_get_item_shape(self):
        shape = Shape({"x": (10,), "y": (5,), "z": (10, 8)})
        self.assertEqual(shape[0], 10)
        self.assertEqual(shape["y"], (5, ))
        self.assertEqual(shape["z"], (10, 8))

    def test_to_tuple(self):
        shape = Shape({"x": (10,), "y": (10,), "z": (10,)})
        self.assertEqual(shape.to_tuple(), (10, 3))
        shape = Shape({"x": (10, 2), "y": (10,)})
        self.assertEqual(shape.to_tuple(), (10, 3))
        shape = Shape({"x": (10, 2), "y": (10, 1), "z": (10, 3, 2)})
        self.assertEqual(shape.to_tuple(), (10, 5, 2))
        shape = Shape({"x": (10, 2), "y": (10, 1, 1), "z": (10, 3, 2, 2)})
        self.assertEqual(shape.to_tuple(), (10, 5, 2, 2))
        shape = Shape({"x": (10, 2), "y": (4, 2), "z": (11, 1, 1)})
        self.assertEqual(shape.to_tuple(), (11, 3, 1))
        shape = Shape({"x": (10,), "y": (5,), "z": (10, 8)})
        self.assertEqual(shape.to_tuple(), (10, 10))
        shape = Shape({"y": (10,)})
        self.assertEqual(shape.to_tuple(), (10,))
        shape = Shape({"y": (10, 1)})
        self.assertEqual(shape.to_tuple(), (10, 1))

    def test_length(self):
        shape = Shape({"x": (10, 2), "y": (3, 2), "z": (11, 1, 1)})
        self.assertEqual(shape.max_length, 11)
        shape = Shape({})
        self.assertEqual(shape.max_length, 0)

    def test_iterable(self):
        shape = Shape({"x": (10, 2)})
        for s0, s1 in zip(shape, (10 ,2)):
            self.assertEqual(s0, s1)
