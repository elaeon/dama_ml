.. image:: https://travis-ci.org/elaeon/dama_ml.svg?branch=master
    :target: https://travis-ci.org/elaeon/dama_ml

.. image:: https://api.codacy.com/project/badge/Grade/0ab998e72f4f4e31b3dc7b3c9921374a
    :target: https://www.codacy.com/app/elaeon/dama_ml?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=elaeon/dama_ml&amp;utm_campaign=Badge_Grade


Warning
=============
    This work is in alpha steps and are methods that have limited functionality, altought, the API is stable.


Overview
=====================================

Dama ML is a framework for data management that is used to do data science and machine learning's pipelines, also dama-ml try to unify diverse data sources (csv, sql db, hdf5, zarr, etc) and machine learning frameworks (sklearn, Keras, LigthGBM, etc) with a simplify interface.

For more detail read the docs_. 

.. _docs: https://elaeon.github.io/dama_ml/


Installation
=====================

.. code-block:: bash

    git clone https://github.com/elaeon/dama_ml.git
    pip install dama_ml/
    or
    pip install dama_ml


You can install the python dependences with pip, but we strongly recommend install the dependences with conda and conda forge.

.. code-block:: bash

    conda config --add channels conda-forge
    conda create -n new_environment --file dama_ml/requirements.txt
    source activate new_environment
    pip install dama_ml/
   

Quick start
==================

First, configure the data paths where all data will be saved. This can be done with help of dama_ml cli tools.

.. code-block:: python

    $ dama-cli config --edit
  
This will display a nano editor where you can edit data_path, models_path, code_path, class_path, metadata_path.
data_path is where all datasets wiil be saved.
models_path is where all files from your models will be saved.
code_path is the repository os code.
metadata_path is where the metadata database will be saved.

* Build a dataset

.. code-block:: python

    from dama.data.ds import Data
    from dama.drivers.core import Zarr, HDF5
    import numpy as np
    
    array_0 = np.random.rand(100, 1)
    array_1 = np.random.rand(100,)
    array_2 = np.random.rand(100, 3)
    array_3 = np.random.rand(100, 6)
    array_4 = (np.random.rand(100)*100).astype(int)
    array_5 = np.random.rand(100).astype(str)
    with Data(name=name, driver=Zarr(mode="w")) as data:
        data.from_data({"x": array_0, "y": array_1, "z": array_2, "a": array_3, "b": array_4, "c": array_5})
    

Then pass it to a regression model, in this case we used RandomForestRegressor

.. code-block:: python

    from dama.reg.extended.w_sklearn import RandomForestRegressor
    from dama.utils.model_selection import CV

    data.driver.mode = "r"  # we changed mode "w" to "r" to not overwrite the data previously saved
    with data, Data(name="test_from_hash", driver=HDF5(mode="w")) as ds:
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)  # cross validation class
        stc = cv.apply(data)
        ds.from_data(stc, from_ds_hash=data.hash)
        reg = RandomForestRegressor()
        model_params = dict(n_estimators=25, min_samples_split=2)
        reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                  data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                  data_validation_group="validation_x", target_validation_group="validation_y")
        reg.save(name="test_model", model_version="1")

Using RandomForestRegressor to do predictions is like this:

.. code-block:: python

    with RandomForestRegressor.load(model_name="test_model", model_version="1") as reg:
        for pred in reg.predict(data):
            prediction = pred.batch.to_ndarray()


CLI
==============
dama-ml has a CLI where you can manage your datasets and models.
For example

.. code-block:: bash

    dama-cli datasets

Return a table of datasets previosly saved.

.. code-block:: python

    Using metadata /home/alejandro/softstream/metadata/metadata.sqlite3
    Total 2 / 2

    hash                    name            driver    group name    size       num groups  datetime UTC
    ---------------------  --------------  --------  ------------  --------  ------------  -------------------
    sha1.3124d5f16eb0e...  test_from_hash  HDF5      s/n           9.12 KB              6  2019-02-27 19:39:00
    sha1.e832f56e33491...  reg0            Zarr      s/n           23.68 KB             6  2019-02-27 19:39:00



You can use "--help" for view more options. 
