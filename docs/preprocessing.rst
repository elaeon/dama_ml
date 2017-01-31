Processing
=====================================

When you build a dataset you can add transformations to the data, i.e preprocessing it before training. So if you need some transformations to be applied to the data, add the functions to the Transformations class.

For example, we want to add two extra functions for preprocessing:

.. code-block:: python
    
    from ml.processing import Transforms

    def my_scale(self):
        return np.log(1 + data)

    def plus_b(self, b=1):
        return data + b

    transforms = Transforms()
    transforms.add(my_scale) 
    transforms.add(plus_b, b=1)


Now in the DataSetBuilder use this class:

.. code-block:: python)

    dataset = DataSetBuilder(
        dataset_name,
        transforms=transforms)
    dataset.build_dataset(X, Y)



        
