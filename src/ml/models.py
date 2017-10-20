from ml.utils.config import get_settings
import tensorflow as tf
import os

settings = get_settings("ml")

class MLModel:
    def __init__(self, fit_fn=None, predictors=None, load_fn=None, save_fn=None,
                transform_data=None):
        self.fit_fn = fit_fn
        self.predictors = predictors
        self.load_fn = load_fn
        self.save_fn = save_fn
        self.transform_data = transform_data

    def fit(self, *args, **kwargs):
        return self.fit_fn(*args, **kwargs)

    def predict(self, data):
        if self.transform_data is not None:
            prediction = self.predictors[0](self.transform_data(data))
        else:
            prediction = self.predictors[0](data)
        #for predictor in self.predictors[1:]:
        #    prediction = predictor(prediction)
        return prediction

    def load(self, path):
        return self.load_fn(path)

    def save(self, path):
        return self.save_fn(path)


class DataDrive(object):
    def __init__(self, check_point_path=None, model_version=None, model_name=None,
                group_name=None):
        if check_point_path is None:
            self.check_point_path = settings["checkpoints_path"]
        else:
            self.check_point_path = check_point_path
        self.model_version = model_version
        self.model_name = model_name
        self.group_name = group_name

    def _metadata(self):
        pass

    def save_meta(self):
        from ml.ds import save_metadata
        if self.check_point_path is not None:
            path = self.make_model_file()
            save_metadata(path+".xmeta", self._metadata())

    def edit_meta(self, key, value):
        from ml.ds import save_metadata
        meta = self.load_meta()
        meta[key] = value
        if self.check_point_path is not None:
            path = self.make_model_file()
            save_metadata(path+".xmeta", meta)

    def load_meta(self):
        from ml.ds import load_metadata
        if self.check_point_path is not None:
            path = self.make_model_file()
            return load_metadata(path+".xmeta")

    def get_model_path(self):
        model_name_v = self.get_model_name_v()
        path = os.path.join(self.check_point_path, self.__class__.__name__, model_name_v)
        return os.path.join(path, model_name_v)

    @classmethod
    def read_meta(self, data_name, path):        
        from ml.ds import load_metadata
        if data_name is not None:
            return load_metadata(path+".xmeta").get(data_name, None)
        return load_metadata(path+".xmeta")

    def get_model_name_v(self):
        if self.model_version is None:
            id_ = "0"
        else:
            id_ = self.model_version
        return "{}.{}".format(self.model_name, id_)

    def make_model_file(self):
        from ml.utils.files import check_or_create_path_dir
        model_name_v = self.get_model_name_v()
        check_point = check_or_create_path_dir(self.check_point_path, self.__class__.__name__)
        destination = check_or_create_path_dir(check_point, model_name_v)
        return os.path.join(check_point, model_name_v, model_name_v)

    def print_meta(self):
        print(self.load_meta())

    def destroy(self):
        """remove the dataset associated to the model and his checkpoints"""
        from ml.utils.files import rm
        self.dataset.destroy()
        if hasattr(self, 'dl') and self.dl is not None:
            self.dl.destroy()
        rm(self.get_model_path()+"."+self.ext)
        rm(self.get_model_path()+".xmeta")

    @classmethod
    def cls_name(cls):
        return cls.__name__

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)
