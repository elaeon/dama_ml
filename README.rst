.. image:: https://travis-ci.org/elaeon/ML.svg?branch=master
    :target: https://travis-ci.org/elaeon/ML

.. image:: https://api.codacy.com/project/badge/Grade/0ab998e72f4f4e31b3dc7b3c9921374a
    :target: https://www.codacy.com/app/elaeon/dama_ml?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=elaeon/dama_ml&amp;utm_campaign=Badge_Grade

Welcome to dama_ml documentation!
=====================================

A framework for machine learning's pipelines.

Highlights:
 * Build datasets with metadata for reproductible experiments.
 * Easy way to add distincts deep learning and machine learning frameworks (i.e Keras, scikit-learn, etc)
 * Manage yours models and scores.
 * Convert n type of files to datasets.

.. image:: docs/pipeline.png
    :align: center

For more detail read the docs_. 

.. _docs: https://elaeon.github.io/dama_ml/

Installation
=====================

.. code-block:: bash

    git clone https://github.com/elaeon/dama_ml.git


You can install the python dependences with pip, but we strongly
recommend install the dependences with conda and conda forge.

.. code-block:: bash

    conda config --add channels conda-forge
    conda create -n new_environment --file ML/requirements.txt
    source activate new_environment
    pip install ML/

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

    classif = SVGPC(
        model_name="my_test_model",
        model_version="1")
    predictions = np.asarray(list(classif.predict(X, chunk_size=258)))


CLI
==============
mlPyp has a CLI where you can admin your datasets and models.
For example

.. code-block:: bash

    ml datasets

Return a table of datasets previosly builded.

.. code-block:: python

    dataset    size       date
    ---------  ---------  --------------------------
    numbers    240.03 MB  2016-12-10 23:50:14.167061
    test2      16.79 MB   2016-12-17 23:28:46.739531

Or

.. code-block:: bash

    ml models

Returns

.. code-block:: python

    classif    model name      version  dataset    group
    ---------  ------------  ---------  ---------  -------
    Boosting   numerai               1  numerai
    SVGPC      test2                 1  test2      basic

You can use "--help" for view more options. 
