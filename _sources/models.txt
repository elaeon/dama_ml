Models
=====================================

The models are classificators with operations like fit and predict. Only this two operations are
need for add models to python-ml.

By default the supported frameworks are:
    * tensorflow
    * keras
    * scikit-learn
    * tflearn

but you can define more, see: :doc:`wrappers`.

For example, we add a model called AdaBoost.

.. code-block:: python

    # SKLP is the wrapper for classificators with probabilistic 
    # predictions in scikit-learn
    from ml.clf.extended.w_sklearn import SKLP
    class AdaBoost(SKLP):
        def prepare_model(self):
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.calibration import CalibratedClassifierCV
            reg = CalibratedClassifierCV(
                AdaBoostClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
            reg.fit(self.dataset.train_data, self.dataset.train_labels)
            sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
            sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
            #sig_clf must have operations like fit and predict 
            #i.e sig_clf.fit(X, y), sig_clf.predict(X)
            #if not you can use MLModel as wrapper
            #from ml.models import MLModel
            #return MLModel(fit_fn=sig_clf.fit, 
            #                predictors=[sig_clf.predict],
            #                load_fn=self.load_fn, 
            #                save_fn=save) you would define save
            return sig_clf

Now we can use AdaBoost.

.. code-block:: python

    dataset = ml.dataset = ml.ds.DataSetBuilder.load_dataset(
        "cats_and_dogs_dataset")

    classif = AdaBoost(
        model_name="my_new_model",
        dataset=dataset,
        model_version="1",
        group_name="cats_and_dogs")
    classif.train(batch_size=128, num_steps=10)
    #Automaticly the train is saved and now can predict data.

Predict data is like
    
.. code-block:: python

    data = ...
    classif = AdaBoost(
        model_name="my_new_model",
        model_version="1")
    predictions = classif.predict(data)

If you want add a TensorFlow model i.e a multilayer perceptron

.. code-block:: python

    # TFL is the wrapper for classificators in TensorFlow
    from ml.clf.extended.w_tflearn import TFL
    class MLP(TFL):
        def __init__(self, *args, **kwargs):
            if "layers" in kwargs:
                self.layers = kwargs["layers"]
                del kwargs["layers"]
            else:
                self.layers = [128, 64]
            super(MLP, self).__init__(*args, **kwargs)

        def prepare_model(self):
            input_layer = tflearn.input_data(shape=[None, self.num_features])
            layer_ = input_layer
            for layer_size in self.layers:
                dense = tflearn.fully_connected(layer_, layer_size, activation='tanh',
                                                 regularizer='L2', weight_decay=0.001)
                layer_ = tflearn.dropout(dense, 0.5)

            softmax = tflearn.fully_connected(layer_, self.num_labels, activation='softmax')
            sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
            acc = tflearn.metrics.Accuracy()
            net = tflearn.regression(softmax, optimizer=sgd, metric=acc,
                             loss='categorical_crossentropy')
            return tflearn.DNN(net, tensorboard_verbose=3, max_checkpoints=10)

Prediction

.. code-block:: python
    
    data = ...
    classif = MLP(
        model_name="my_perceptron_model",
        model_version="1")
    predictions = classif.predict(data)

For more about it see :doc:`wrappers`.
