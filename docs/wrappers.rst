Wrappers
=====================================

The wrappers are classes where is posible define operations over distincts machine learning's frameworks.
mlPyp has wrappers for clasificators and auto encoders, manly for frameworks like tensorflow, keras, scikit-learn, but if you want to use another framework, the base class 'BaseAe' and 'BaseClassif' will help you to convert you model into mlPyp classes.

For example, if you are using the framework "X" for build a classification model, the BaseClassif is the parent class where you must define (aditionaly to prepare model) convert_label, train, reformat and load functions.

.. code-block:: python

   class Xclf(BaseClassif):
    # load_fn load a saved model
    def load_fn(self, path):
        from xclf.models import load_model
        model = load_model(path)
        self.model = MLModel(fit_fn=model.fit, 
                            predictors=[model.predict],
                            load_fn=self.load_fn,
                            save_fn=model.save)

    # you model maybe can use an array of data and labels in a certain format, here is where
    # you must transform it.
    def reformat(self, data, labels):
        data = self.transform_shape(data)
        labels = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return data, labels

    # about the labels, this is the inverse function of reformat.
    def convert_label(self, label, raw=False):
        if raw is True:
            return label
        else:
            return self.le.inverse_transform(np.argmax(label))

    def prepare_model(self):
        # define you model here

    # define the training here
    def train(self, batch_size=258, num_steps=50):
        model = self.prepare_model()
        if not isinstance(model, MLModel):
            self.model = MLModel(fit_fn=model.fit, 
                            predictors=[model.predict],
                            load_fn=self.load_fn,
                            save_fn=model.save)
        else:
            self.model = model
        self.model.fit(self.dataset.train_data, 
            self.dataset.train_labels,
            nb_epoch=num_steps,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(self.dataset.valid_data, self.dataset.valid_labels))
        self.save_model()
