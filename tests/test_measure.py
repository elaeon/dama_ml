import unittest
import numpy as np

from ml.clf.measures import ListMeasure, Measure


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


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.pred_l = np.asarray([0,0,0,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,1,0,1,0,1,0,1])
        self.labels = np.asarray([0,0,0,1,1,1,0,0,1,1,1,1,0,0,1,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,1])

    def test_metrics(self):
        from ml.clf.measures import accuracy, precision, f1
        measure = Measure(self.pred_l, self.labels, name="test")
        measure.add(accuracy)
        measure.add(precision)
        measure.add(f1)
        metrics = [round(v, 2) for v in list(measure.scores())]
        self.assertEqual(metrics, [0.84, 0.83, 0.83])

    def test_tolist(self):
        from ml.clf.measures import accuracy, precision, f1
        measure0 = Measure(self.pred_l, self.labels, name="test0")
        measure0.add(accuracy)
        measure0.add(precision)
        measure0.add(f1)
        measure1 = Measure(self.pred_l, self.labels, name="test1")
        measure1.add(accuracy)
        measure1.add(precision)
        measure1.add(f1)
        measure2 = Measure(self.pred_l, self.labels, name="test2")
        measure2.add(accuracy)
        measure2.add(precision)
        measure2.add(f1)
        list_measure = measure0.to_list() + measure1.to_list() + measure2.to_list()
        self.assertEqual(list_measure.measures[0], 
            ['test0', 0.83870967741935487, 0.829059829059829, 0.83243243243243248])
        self.assertEqual(list_measure.measures[1], 
            ['test1', 0.83870967741935487, 0.829059829059829, 0.83243243243243248])
        self.assertEqual(list_measure.measures[2], 
            ['test2', 0.83870967741935487, 0.829059829059829, 0.83243243243243248])

    def test_gini(self):
        from ml.utils.numeric_functions import gini_normalized
        measure = Measure(self.pred_l, self.labels, name="test")
        measure.add(gini_normalized)
        metrics = [round(v, 2) for v in list(measure.scores())]
        self.assertEqual(metrics, [0.59])


if __name__ == '__main__':
    unittest.main()
