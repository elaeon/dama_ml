Datasets
=====================================

Data is a wrapper for the driver class to show the data saved
into his structure.

The datasets have groups of n-dimensions:

.. code-block:: python

    from dama.data.ds import Data
    from dama.drivers.core import Zarr
    import numpy as np

    array_0 = np.random.rand(100, 1)
    array_1 = np.random.rand(100, 2)
    array_2 = np.random.rand(100, 3)
    array_3 = np.random.rand(100, 6)
    array_4 = (np.random.rand(100)*100).astype(int)
    array_5 = np.random.rand(100).astype(str)
    with Data(name=name, driver=Zarr(mode="w")) as data:
        data.from_data({"x": array_0, "y": array_1, "z": array_2, "a": array_3, "b": array_4, "c": array_5})


In the above example the dataset have x, y, z, a, b and c groups,
each one with distinct shape, but with the same length.

.. code-block:: python


    with Data(name=name, driver=Zarr(mode="r"), auto_chunks=True) as data:
        print(data)
        print(data[["x", "y"]])
        print(data["x"] + data["y"])  # same as above
        data["x"] = data["x"].darray * 3
        print(data["x"].darray.dask)

.. code-block:: bash

    DaGroup OrderedDict([('a', (100, 6)), ('b', (100,)), ('c', (100,)), ('x', (100, 1)), ('y', (100,)), ('z', (100, 3))])
    DaGroup OrderedDict([('x', (100, 1)), ('y', (100,))])
    DaGroup OrderedDict([('x', (100, 1)), ('y', (100,))])
    <dask.highlevelgraph.HighLevelGraph object at 0x7f682a8e5b70>