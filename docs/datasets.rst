Datasets
=====================================

The datasets are files in pickle format with data divided in train, test and validation. If this data was preprocessed, information about it is added.

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
    Transforms: [('global', []), ('row', [])]
    Preprocessing Class: None
    MD5: daa92472f94c90b5bacb6ab172a73566

    Dataset        Mean       Std  Shape        dType      Labels
    train set  0.500195  0.288638  (70000, 21)  float64     70000
    test set   0.498693  0.288993  (20000, 21)  float64     20000
    valid set  0.499823  0.288698  (10000, 21)  float64     10000


There are two more DataSetBuilder class, a dataset builder for images, and another for datasets in csv files. The next example is a DataSetBuilderFile

.. code-block:: python

    dataset = DataSetBuilderFile(
        dataset_name,
        train_folder_path="/home/ds/my_file.csv",
        transforms_global=transforms)
    dataset.build_dataset(label_column="target")

Preprocessing class is a way to add transformations to the data. There are predefined functions, 
but, you can add your own functions. For more info about it, check :doc:`preprocessing`.
