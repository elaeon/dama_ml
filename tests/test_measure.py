import unittest
import numpy as np

from ml.clf.measures import ListMeasure


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
        self.assertEqual(m1['values'], [1])
        self.assertEqual(m1['reverse'], False)

        list_measure_1_2 = self.list_measure1 + self.list_measure2
        m1_m2 = list_measure_1_2.get_measure("M1")
        self.assertEqual(m1_m2['values'], [1, 0.5])
        self.assertEqual(m1_m2['reverse'], False)

    def test_list_measure_to_dict(self):
        list_measure_1_2 = self.list_measure1 + self.list_measure2
        #list_measure_1_2.print_scores()
        self.assertEqual(list_measure_1_2.measures_to_dict(), 
            {'M1': {'values': [1, 0.5], 'reverse': False}, 
            'Name': {'values': ['Row1', 'Row2'], 'reverse': False}})

    def test_list_measure_empty(self):
        list_measure = ListMeasure(headers=["Name", "M1", "M2", "M3"], 
            measures=[["Row1", None, 0, ""], ["Row2", None, None, ""]])
        self.assertEqual(list_measure.empty_columns(), set([1, 3]))
        list_measure.drop_empty_columns()
        self.assertEqual(list_measure.headers, ["Name", "M2"])
        self.assertEqual(list_measure.measures, [["Row1", 0], ["Row2", None]]) 
        #list_measure.print_scores()


if __name__ == '__main__':
    unittest.main()
