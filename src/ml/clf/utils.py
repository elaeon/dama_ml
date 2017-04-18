def proximity_label(label_ref, labels, dataset):
    from sklearn import svm
    dataset_ref, _ = dataset.only_labels([label_ref])
    clf = svm.OneClassSVM(nu=.2, kernel="rbf", gamma=0.5)
    clf.fit(dataset_ref.reshape(dataset_ref.shape[0], -1))
    for label in labels:
        dataset_other, _ = dataset.only_labels([label])
        y_pred_train = clf.predict(dataset_other.reshape(dataset_other.shape[0], -1))
        n_error_train = y_pred_train[y_pred_train == -1].size
        yield label, (1 - (n_error_train / float(y_pred_train.size)))


def proximity_dataset(label_ref, labels, dataset):
    from sklearn import svm
    dataset_ref, _ = dataset.only_labels([label_ref])
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(dataset_ref.reshape(dataset_ref.shape[0], -1))
    for label in labels:
        dataset_other_, _ = dataset.only_labels([label])
        y_pred_train = clf.predict(dataset_other_.reshape(dataset_other_.shape[0], -1))
        return filter(lambda x: x[1] == -1, zip(dataset_other_, y_pred_train))


def add_params_to_params(classifs_layer, others_models_args):
    if len(others_models_args) == 1:
        others_models_args_c = {m.cls_name(): [{}] for m, _ in classifs_layer["0"]}
    else:
        others_models_args_c = others_models_args[1].copy()

    for m, _ in classifs_layer["0"]:
        n_params = []
        for params in others_models_args_c.get(m.cls_name(), [{}]):
            if not "n_splits" in params:
                tmp_params = params.copy()
                tmp_params["n_splits"] = 5
                n_params.append(tmp_params)
            else:
                n_params.append(params)
        others_models_args_c[m.cls_name()] = n_params

    return others_models_args_c
