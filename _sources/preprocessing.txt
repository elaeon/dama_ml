Preprocessing
=====================================

When you build a dataset you can add transformations to the data, i.e preprocessing it before training. Preprocessing is the parent class, so if you need you own transformation be applied, you can extend the Preprocessing class.

For example, we want two extra functions for preprocessing:

.. code-block:: python

    from ml.preprocessing import Preprocessing
    class MyPreprocessing(Preprocessing):

        def my_scale(self):
            self.data = np.log(1 + self.data)

        def plus_b(self, b=1):
            self.data = self.data + b


Now in the DataSetBuilder use this class:

.. code-block:: python

    dataset = DataSetBuilder(
        dataset_name, 
        preprocessing_class=MyPreprocessing
        transforms_row=[('my_scale', None), ('plus_b', {'b': 1})])
    dataset.build_dataset(X, Y)

transforms_row apply indepently transformations to each elem in the data, but if you want add a transformation
who interact (dependently) with each, you must use the FiT class.

Fit is similar to Preprocessing except, you need define a FiT for every function.

.. code-block:: python

    from ml.preprocessing import FiT
    class FiTScaler(FiT):
        # if the transformation not support high dimensions
        def dim_rule(self, data):
            if len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)
            return data

        #This is the only function you need to define
        def fit(self, data):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler(**self.params)
            scaler.fit(self.dim_rule(data))
            self.t = scaler

Now apply FiT to you dataset with:

.. code-block:: python

    dataset = DataSetBuilder(
        dataset_name,
        transforms_global=[(FiTScaler.module_cls_name(), None)])
    dataset.build_dataset(X, Y)


        
