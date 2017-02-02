Ensemble models
=====================================

The ensemble models have less overfit and lower error. There are 3 type of ensemble models
Boosting, Stacking and Bagging, each model have his pros an cons. You can experment and choose
whose of this is better for you job.


Boosting
--------------

.. code-block:: python

    from ml.clf import ensemble as clf_ensemble

    classif = clf_ensemble.Boosting({"0": [
                    w_sklearn.ExtraTrees,
                    w_tflearn.MLP,
                    w_sklearn.RandomForest]},
                    w_sklearn.SGDClassifier,
                    w_sklearn.SVC,
                    w_sklearn.LogisticRegression,
                    w_sklearn.AdaBoost,
                    w_sklearn.GradientBoost]},
                    dataset=dataset,
                    model_name="boosting",
                    model_version="1",
                    weights=[3, 1],
                    election='best-c',
                    num_max_clfs=5)
    classif.train(batch_size=128, num_steps=args.epoch)

.. code-block:: python

    classif = clf_ensemble.Boosting({},
                model_name="boosting",
                model_version="1")
    classif.predict(data, raw=True, chunk_size=258)


Stacking
--------------

.. code-block:: python

    classif = clf_ensemble.Stacking({"0": [
                w_sklearn.ExtraTrees,
                w_tflearn.MLP,
                w_sklearn.RandomForest,
                w_sklearn.SGDClassifier,
                w_sklearn.SVC,
                w_sklearn.LogisticRegression,
                w_sklearn.AdaBoost,
                w_sklearn.GradientBoost]},
                n_splits=3,
                dataset=dataset,
                model_name="stacking",
                model_version="1")

.. code-block:: python
    
    classif = clf_ensemble.Stacking({},
                    model_name="stacking",
                    model_version="1")
    classif.predict(data, raw=True, chunk_size=258)

        
Bagging
----------------

.. code-block:: python

    classif = clf_ensemble.Bagging(w_tflearn.MLP, {"0": [
                w_sklearn.ExtraTrees,
                w_tflearn.MLP,
                w_sklearn.RandomForest,
                w_sklearn.SGDClassifier,
                w_sklearn.SVC,
                w_sklearn.LogisticRegression,
                w_sklearn.AdaBoost,
                w_sklearn.GradientBoost]},
                dataset=dataset,
                model_name="bagging",
                model_version="1")

.. code-block:: python

    classif = clf_ensemble.Bagging({},
                    model_name="bagging",
                    model_version="1")
    classif.predict(data, raw=True, chunk_size=258)
