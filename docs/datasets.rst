Datasets
=====================================

The datasets are files in hdf5 format with data divided in train, test and validation. If this data was preprocessed, information about it is added.

For example.

.. code-block:: python

    DIM = 21
    SIZE = 100000
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 
        for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    dataset_name = "test"
    dataset = DataSetBuilder(
        dataset_name, 
        validator="cross")
    dataset.build_dataset(X, Y)
    dataset.info()

Results in:

.. code-block:: python

    DATASET NAME: test
    Author: Alejandro Martínez
    Transforms: {}
    MD5: daa92472f94c90b5bacb6ab172a73566
    Description: test dataset

    Dataset        Mean       Std  Shape        dType    Labels
    ---------  --------  --------  -----------  -------  --------
    train set  0.500195  0.288638  (70000, 21)  float64   70000
    test set   0.498693  0.288993  (20000, 21)  float64   20000
    valid set  0.499823  0.288698  (10000, 21)  float64   10000


There are two more DataSetBuilder classes, a dataset builder for images, and another for files in csv format.

.. code-block:: python

    transforms = [("poly_features", {"degree":2, "interaction_only":False)]
    dataset = DataSetBuilderFile(
        name="test_ds_file",
        train_folder_path="/home/ds/my_file.csv",
        transforms=transforms)
    dataset.build_dataset(label_column="target")

In the processing module are predefined functions for transforms, but, you can add your own functions. For more info about it, check :doc:`processing`.
