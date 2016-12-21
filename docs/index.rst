
Welcome to python-ml's documentation!
=====================================

A framework for machine learning's pipelines.

Highlights:
 * Build datasets with train, test, valid data and transformations applied.
 * Build datasets with metadata for reproducible experiments.
 * Easy way to add distincts machine learning algorithms from Keras, scikit-learn, etc.
 * Models with scores and predictors.
 * Convert csv files to datasets.
 * Uses transformations for manipulate data (images).

Instalation
=====================

.. code-block:: bash

    git clone https://github.com/elaeon/ML.git


You can install the python dependences with pip, but we strongly
recommend install the dependences with conda and conda forge.

.. code-block:: bash

    conda config --add channels conda-forge
    conda create -n new_environment --file requirements.txt
    pip install ML/setup.py

Quick start
==================

First, build a dataset

.. code-block:: python

    from ml.ds import DataSetBuilder
    import numpy as np

    DIM = 21
    SIZE = 100000
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 
        for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    dataset_name = "test_dataset"
    dataset = DataSetBuilder(
        dataset_name,
        validator="cross")
    dataset.build_dataset(X, Y)
    

Then, pass it to a classification model for training, in this case we used SVGC (was a Gaussian process with stochastic variational inference), once the training was finished you can predict some data.

.. code-block:: python

    from ml.clf.extended.w_gpy import SVGPC

    classif = SVGPC(
        dataset=dataset,
        model_name="my_test_model",
        model_version="1")
    classif.train(batch_size=128, num_steps=10)
    classif.scores().print_scores(order_column="f1")

Using SVGPC for make predictions is like this:

.. code-block:: python

    classif = ml.clf.extended.SVGPC(
        model_name="my_test_model",
        model_version="1")
    predictions = np.asarray(list(classif.predict(X, chunk_size=258)))


You can use more predifined models and extend the base model and make you own predictor. For more information about this, see the section :doc:`models`. 

More information
================

.. toctree::
   :maxdepth: 2
   :name: mastertoc

   datasets
   preprocessing
   models
   reporting_bugs
   changes


Support
=======
If you encounter bugs then `let me know`_ . Please see :doc:`reporting_bugs`
for information how best report them.

.. _let me know: https://github.com/elaeon/ML/issues


