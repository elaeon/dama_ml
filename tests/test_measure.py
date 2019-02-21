import unittest
import numpy as np

from ml.measures import ListMeasure, Measure, MeasureBatch
from ml.measures import accuracy, precision, f1, logloss
from ml.measures import gini_normalized
from ml.data.it import Iterator


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
        self.assertEqual(list_measure_1_2.measures_to_dict(),
                         {'M1': {'values': [1, 0.5], 'reverse': False},
                          'Name': {'values': ['Row1', 'Row2'], 'reverse': False}})

    def test_list_measure_empty(self):
        list_measure = ListMeasure(headers=["Name", "M1", "M2", "M3"],
                                   measures=[["Row1", None, 0, ""], ["Row2", None, None, ""]])
        self.assertEqual(list_measure.empty_columns(), {1, 3})
        list_measure.drop_empty_columns()
        self.assertEqual(list_measure.headers, ["Name", "M2"])
        self.assertEqual(list_measure.measures, [["Row1", 0], ["Row2", None]])

    def test_add_list_empty(self):
        pred = np.asarray([0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                           1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        target = np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1,
                             1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        measure = Measure(name="test")
        measure.add(accuracy)
        list_measure = ListMeasure()
        measure.update(pred, target)
        list_measure += measure.to_list()
        self.assertCountEqual(list_measure.headers, ['', 'accuracy'])
        self.assertEqual(len(list_measure.measures) == 1, True)
        self.assertEqual(list_measure.measures[0][0], "test")

    def test_add_list(self):
        pred = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        target = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        measure0 = Measure(name="test0")
        measure0.add(accuracy)
        measure1 = Measure(name="test1")
        measure1.add(accuracy)
        measure0.update(pred, target)
        measure1.update(pred, target)
        list_measure = measure0.to_list()
        list_measure += measure1.to_list()
        self.assertCountEqual(list_measure.headers, ['', 'accuracy'])
        self.assertEqual(len(list_measure.measures) == 2, True)
        self.assertEqual(list_measure.measures[0][0], "test0")


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.pred_l = np.asarray([0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0,
                                  0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.target = np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
                                  1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    def test_metrics(self):
        measure = Measure(name="test")
        measure.add(accuracy)
        measure.add(precision)
        measure.add(f1)
        measure.update(self.pred_l, self.target)
        metrics = [round(v, 2) for v in list(measure.scores())]
        self.assertEqual(metrics, [0.84, 0.83, 0.83])

    def test_metrics_batch(self):
        measure = Measure(name="test")
        measure.add(accuracy)
        measure.add(precision)
        measure.add(f1)
        measure.add(logloss)
        measure.update(self.pred_l, self.target)
        batch_size = 28
        measure_b = MeasureBatch(name="test", batch_size=batch_size)
        measure_b.add(accuracy)
        measure_b.add(precision)
        measure_b.add(f1)
        measure_b.add(logloss)
        it_p = Iterator(self.pred_l).batchs(chunks=(batch_size,))
        it_t = Iterator(self.target).batchs(chunks=(batch_size,))
        for pred, target in zip(it_p, it_t):
            measure_b.update(pred, target)
        m0 = [round(score, 2) for score in measure.scores()]
        mb = [round(score, 2) for score in measure_b.scores()]
        self.assertEqual(m0, mb)

    def test_metrics_fn(self):
        batch_size = 10
        measure = MeasureBatch(name="test", batch_size=batch_size)
        measure.add(accuracy, output='discrete')
        measure.add(precision, output='discrete')
        it_p = Iterator(self.pred_l).batchs(chunks=(batch_size, ))
        it_t = Iterator(self.target).batchs(chunks=(batch_size, ))
        for pred, target in zip(it_p, it_t):
            measure.update(pred, target)
        self.assertEqual(list(measure.scores()), [0.8387096774193549, 0.8306451612903226])

    def test_tolist(self):
        measure0 = Measure(name="test0")
        measure0.add(accuracy)
        measure0.add(precision)
        measure0.add(f1)
        measure0.update(self.pred_l, self.target)
        measure1 = Measure(name="test1")
        measure1.add(accuracy)
        measure1.add(precision)
        measure1.add(f1)
        measure1.update(self.pred_l, self.target)
        measure2 = Measure(name="test2")
        measure2.add(accuracy)
        measure2.add(precision)
        measure2.add(f1)
        measure2.update(self.pred_l, self.target)
        list_measure = measure0.to_list() + measure1.to_list() + measure2.to_list()
        self.assertEqual(list_measure.measures[0],
                         ['test0', 0.83870967741935487, 0.829059829059829, 0.83243243243243248])
        self.assertEqual(list_measure.measures[1],
                         ['test1', 0.83870967741935487, 0.829059829059829, 0.83243243243243248])
        self.assertEqual(list_measure.measures[2],
                         ['test2', 0.83870967741935487, 0.829059829059829, 0.83243243243243248])

    def test_gini(self):
        measure = Measure(name="test")
        measure.add(gini_normalized)
        measure.update(self.pred_l, self.target)
        metrics = [round(v, 2) for v in list(measure.scores())]
        self.assertEqual(metrics, [0.59])

    def test_output(self):
        measure = Measure(name="test0")
        measure.add(accuracy, output="discrete")
        measure.add(precision, output="discrete")
        measure.add(gini_normalized, output="uncertain")
        measure.add(f1, output=None)
        measure.update(self.pred_l, self.target)
        self.assertCountEqual(measure.outputs(), ['discrete', None, 'uncertain'])
        self.assertEqual(list(measure.scores()),
                         [0.83870967741935487, 0.829059829059829, 0.5877192982456142, 0.83243243243243248])


if __name__ == '__main__':
    unittest.main()
