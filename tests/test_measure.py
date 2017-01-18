import unittest
import numpy as np

from ml.clf.wrappers import ListMeasure


class TestListMeasure(unittest.TestCase):
    def setUp(self):
        self.list_measure1 = ListMeasure()
        self.list_measure1.add_measure("Name", "Row1")
        self.list_measure1.add_measure("M1", 1)

        self.list_measure2 = ListMeasure()
        self.list_measure2.add_measure("Name", "Row2")
        self.list_measure2.add_measure("M1", .5)


    def test_list_measure(self):
        m1 = self.list_measure1.get_measure("M0")
        self.assertEqual(m1, None)        
        m1 = self.list_measure1.get_measure("M1")
        self.assertEqual(m1, [1])

        list_measure_1_2 = self.list_measure1 + self.list_measure2
        #list_measure_1_2.print_scores()
        m1_m2 = list_measure_1_2.get_measure("M1")
        self.assertEqual(m1_m2, [1, 0.5])


if __name__ == '__main__':
    unittest.main()
