
Welcome to python-ml's documentation!
=====================================

A framework for machine learning pipeline.

Highlights:
 * Build datasets with train, test, valid data and transformations applied.
 * Build datasets with metadata for reproducible experiments.
 * Easy way to add distincts machine learning algorithms from Keras, scikit-learn, etc.
 * Models with scores and predictors.
 * Convert csv files to datasets.
 * Uses transformations for manipulate data (images).

A simple pipeline example. A first step is build a dataset, and then pass it to a
classification model for training, once the training was finished you can predict some data.
Train data is always saved, therefore, steps 1, 2 are no needed for future predictions.


.. code-block:: python

    import ml
    import numpy as np
    from ml.clf.generic import Measure

    DIM = 21
    SIZE = 100000
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 
        for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    dataset_name = "test_dataset"
    dataset = ml.ds.DataSetBuilder(
        dataset_name, 
        dataset_path="/home/ds/datasets/", 
        transforms_global=[(ml.processing.FiTScaler.module_cls_name(), None)],
        validator="cross")
    dataset.build_dataset(X, Y)

    classif = ml.clf.extended.SVGPC(
        model_name="my_test_model",
        dataset=dataset,
        model_version="1",
        check_point_path="/home/ds/checkpoints/",
        group_name="basic")
    classif.train(batch_size=128, num_steps=10)
    classif.scores().print_scores(order_column="f1")

    SIZE_T = 10000
    X = np.random.rand(SIZE_T, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 
        for row in np.sin(6*X) + 0.1*np.random.randn(SIZE_T, 1)])
    classif = ml.clf.extended.SVGPC(
        model_name="my_test_model",
        model_version="1",
        check_point_path="/home/ds/checkpoints/")
    predictions = np.asarray(list(classif.predict(X, chunk_size=1)))
    print("{} elems SCORE".format(SIZE_T), 
        Measure(predictions, Y, classif.numerical_labels2classes).accuracy()


The classification model used in this example, was a Gaussian process with stochastic variational inference. You can extend the base model and create you own predictor. For more information about this, see the section :doc:`models`. 

More information
================

.. toctree::
   :maxdepth: 2
   :name: mastertoc

   datasets
   transforms
   models
   reporting_bugs
   changes


Support
=======
If you encounter bugs then `let me know`_ . Please see :doc:`reporting_bugs`
for information how best report them.

.. _let me know: https://github.com/elaeon/ML/issues

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

