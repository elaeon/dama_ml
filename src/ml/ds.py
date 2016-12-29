from skimage import io

import os
import numpy as np
import pandas as pd
import cPickle as pickle
import random

from ml.processing import PreprocessingImage, Preprocessing, Transforms
from ml.utils.config import get_settings

settings = get_settings("ml")


def save_metadata(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_metadata(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def dtype_c(dtype):
    types = set(["float64", "float32", "int64", "int32"])
    if hasattr(np, dtype) and dtype in types:
        return getattr(np, dtype)


class DataSetBuilder(object):
    def __init__(self, name, 
                dataset_path=None, 
                test_folder_path=None, 
                train_folder_path=None,
                transforms_row=None,
                transforms_global=None,
                transforms_apply=True,
                processing_class=None,
                fits=None,
                train_size=.7,
                valid_size=.1,
                validator='cross',
                dtype='float64'):
        self.test_folder_path = test_folder_path
        self.train_folder_path = train_folder_path
        self.dtype = dtype

        if dataset_path is None:
            self.dataset_path = settings["dataset_path"]
        else:
            self.dataset_path = dataset_path
        self.name = name
        self.train_data = None
        self.train_labels = None
        self.valid_data = None
        self.valid_labels = None
        self.test_data = None
        self.test_labels = None
        self.processing_class = processing_class
        self.valid_size = valid_size
        self.train_size = train_size
        self.test_size = round(1 - (train_size + valid_size), 2)
        self.transforms_apply = transforms_apply
        self._cached_md5 = None
        self.validator = validator
        self.fit = None

        if transforms_row is None:
            transforms_row = ('row', [])
        else:
            transforms_row = ("row", transforms_row)

        if transforms_global is None:
            transforms_global = ("global", [])
        else:
            transforms_global = ("global", transforms_global)

        self.transforms = Transforms([transforms_global, transforms_row])

    def url(self):
        return os.path.join(self.dataset_path, self.name)

    def num_features(self):
        return self.train_data.shape[1]

    @property
    def shape(self):
        return self.train_data.shape

    def desfragment(self):
        data = np.concatenate((
            self.train_data, self.valid_data, self.test_data), axis=0)
        labels = np.concatenate((
            self.train_labels, self.valid_labels, self.test_labels), axis=0)
        return data, labels

    def labels_info(self):
        from collections import Counter
        _, labels = self.desfragment()
        return Counter(labels)

    def only_labels(self, base_labels):
        data, labels = self.desfragment()
        s_labels = set(base_labels)
        dataset, n_labels = zip(*filter(lambda x: x[1] in s_labels, zip(data, labels)))
        return np.asarray(dataset), np.asarray(n_labels)

    def info(self, classes=False):
        from ml.utils.order import order_table_print
        print('       ')
        print('DATASET NAME: {}'.format(self.name))
        print('Transforms: {}'.format(self.transforms.get_all_transforms()))
        print('Preprocessing Class: {}'.format(self.get_processing_class_name()))
        print('MD5: {}'.format(self._cached_md5))
        print('       ')
        if not isinstance(self.train_data.dtype, object):
            headers = ["Dataset", "Mean", "Std", "Shape", "dType", "Labels"]
            table = []
            table.append(["train set", self.train_data.mean(), self.train_data.std(), 
                self.train_data.shape, self.train_data.dtype, self.train_labels.size])

            if self.valid_data is not None:
                table.append(["valid set", self.valid_data.mean(), self.valid_data.std(), 
                self.valid_data.shape, self.valid_data.dtype, self.valid_labels.size])

            table.append(["test set", self.test_data.mean(), self.test_data.std(), 
                self.test_data.shape, self.test_data.dtype, self.test_labels.size])
            order_table_print(headers, table, "shape")
        else:
            headers = ["Dataset", "Shape", "dType", "Labels"]
            table = []
            table.append(["train set", self.train_data.shape, self.train_data.dtype, 
                self.train_labels.size])

            if self.valid_data is not None:
                table.append(["valid set", self.valid_data.shape, self.valid_data.dtype, 
                self.valid_labels.size])

            table.append(["test set", self.test_data.shape, self.test_data.dtype, 
                self.test_labels.size])
            order_table_print(headers, table, "shape")

        if classes is True:
            headers = ["class", "# items"]
            order_table_print(headers, self.labels_info().items(), "# items")

    def cross_validators(self, data, labels):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, train_size=round(self.train_size+self.valid_size, 2), random_state=0)

        valid_size_index = int(round(data.shape[0] * self.valid_size))
        X_validation = X_train[:valid_size_index]
        y_validation = y_train[:valid_size_index]
        X_train = X_train[valid_size_index:]
        y_train = y_train[valid_size_index:]
        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def get_processing_class_name(self):
        if self.processing_class is None:
            return None
        else:
            return self.processing_class.module_cls_name()

    def dtype_t(self, data):
        dtype = dtype_c(self.dtype)
        if data.dtype is not dtype and not isinstance(data.dtype, object):
            return data.astype(dtype_c(self.dtype))
        else:
            return data

    def to_raw(self):
        return {
            'train_dataset': self.dtype_t(self.train_data),
            'train_labels': self.train_labels,
            'valid_dataset': self.dtype_t(self.valid_data),
            'valid_labels': self.valid_labels,
            'test_dataset': self.dtype_t(self.test_data),
            'test_labels': self.test_labels,
            'transforms': self.transforms.get_all_transforms(),
            'preprocessing_class': self.get_processing_class_name(),
            'md5': self.md5()}

    @classmethod
    def from_raw_to_ds(self, name, dataset_path, data, save=True):
        ds = DataSetBuilder(name, 
                dataset_path=dataset_path)
        ds.from_raw(data)
        if save is True:
            ds.save()
        return ds

    def from_raw(self, raw_data):
        from pydoc import locate
        #if self.processing_class is None:
        if raw_data["preprocessing_class"] is not None:
            self.processing_class = locate(raw_data["preprocessing_class"])

        self.transforms = Transforms(raw_data["transforms"])
        self.train_data = self.dtype_t(raw_data['train_dataset'])
        self.train_labels = raw_data['train_labels']
        self.valid_data = self.dtype_t(raw_data['valid_dataset'])
        self.valid_labels = raw_data['valid_labels']
        self.test_data = self.dtype_t(raw_data['test_dataset'])
        self.test_labels = raw_data['test_labels']        
        self._cached_md5 = raw_data["md5"]

    def shuffle_and_save(self, data, labels):
        self.train_data, self.valid_data, self.test_data, self.train_labels, self.valid_labels, self.test_labels = self.cross_validators(data, labels)
        self.save()

    def adversarial_validator_and_save(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.valid_data = train_data[0:1]
        self.valid_labels = train_labels[0:1]

        # train class is labeled as 1 
        train_test_data_clf = self._clf()
        train_test_data_clf.train()
        class_train = np.argmax(train_test_data_clf.model.classes_)
        predictions = [p[class_train] 
            for p in train_test_data_clf.predict(self.train_data, raw=True, transform=False)]
        predictions = sorted(enumerate(predictions), key=lambda x: x[1], reverse=False)
        #print([(pred, index) for index, pred in predictions if pred < .5])
        #because all targets are ones (train data) is not necessary compare it
        false_test = [index for index, pred in predictions if pred < .5] # is a false test 
        ok_train = [index for index, pred in predictions if pred >= .5]
 
        valid_data = self.train_data[false_test]
        valid_labels = self.train_labels[false_test]
        valid_size = int(round(self.train_data.shape[0] * self.valid_size))
        self.valid_data = valid_data[:int(round(valid_size))]
        self.valid_labels = valid_labels[:int(round(valid_size))]
        self.train_data = np.concatenate((
            valid_data[int(round(valid_size)):], 
            self.train_data[ok_train]),
            axis=0)
        self.train_labels = np.concatenate((
            valid_labels[int(round(valid_size)):],
            self.train_labels[ok_train]),
            axis=0)
        self.save()

    def none_validator(self, train_data, train_labels, test_data, test_labels, 
                        valid_data, valid_labels):
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.valid_data = valid_data
        self.valid_labels = valid_labels

    def save(self):
        if self.dataset_path is not None:
            if not os.path.exists(self.dataset_path):
                    os.makedirs(self.dataset_path)
            destination = os.path.join(self.dataset_path, self.name)
            try:
                with open(destination, 'wb') as f:
                    pickle.dump(self.to_raw(), f, pickle.HIGHEST_PROTOCOL)
            except IOError as e:
                print('Unable to save data to: ', destination, e)

    @classmethod
    def load_dataset_raw(self, name, dataset_path=None):
        with open(dataset_path+name, 'rb') as f:
            save = pickle.load(f)
            return save

    @classmethod
    def load_dataset(self, name, dataset_path=None, info=True, 
            processing_class=None, dtype='float64'):
        if dataset_path is None:
             dataset_path = settings["dataset_path"]
        data = self.load_dataset_raw(name, dataset_path=dataset_path)
        dataset = DataSetBuilder(name, dataset_path=dataset_path, 
            processing_class=processing_class, dtype=dtype)
        dataset.from_raw(data)
        if info:
            dataset.info()
        return dataset        

    def is_binary(self):
        return len(self.labels_info()) == 2

    @classmethod
    def to_DF(self, dataset, labels):
        if len(dataset.shape) > 2:
            dataset = dataset.reshape(dataset.shape[0], -1)
        columns_name = map(lambda x: "c"+str(x), range(dataset.shape[-1])) + ["target"]
        return pd.DataFrame(data=np.column_stack((dataset, labels)), columns=columns_name)

    def to_df(self):
        data, labels = self.desfragment()
        return self.to_DF(data, labels)

    def processing_rows(self, data):
        if not self.transforms.empty('row') and self.transforms_apply and data is not None:
            pdata = []
            for row in data:
                preprocessing = self.processing_class(row, self.transforms.get_transforms('row'))
                pdata.append(preprocessing.pipeline())
            return np.asarray(pdata)
        else:
            return data if isinstance(data, np.ndarray) else np.asarray(data)

    def processing_global(self, data, base_data=None):
        if not self.transforms.empty('global') and self.transforms_apply and data is not None:
            if self.fit is None:
                #if base_data is None:
                #    base_data = data
                from pydoc import locate
                fiter, params = self.transforms.get_transforms('global')[0]
                fiter = locate(fiter)
                if isinstance(params, dict):
                    self.fit = fiter(**params)
                else:
                    self.fit = fiter()
                self.fit.fit(base_data)
                return self.fit.transform(data)
            else:
                return self.fit.transform(data)
        else:
            return data

    def build_dataset(self, data, labels, test_data=None, test_labels=None, 
                        valid_data=None, valid_labels=None):
        data = self.processing(data)
        test_data = self.processing_rows(test_data)
        valid_data = self.processing_rows(valid_data)

        if self.validator == 'cross':
            if test_data is not None and test_labels is not None:
                data = np.concatenate((data, test_data), axis=0)
                labels = np.concatenate((labels, test_labels), axis=0)
            self.shuffle_and_save(data, labels)
        elif self.validator == 'adversarial':
            self.adversarial_validator_and_save(data, labels, test_data, test_labels)
        elif self.validator is None:
            self.none_validator(data, labels, test_data, test_labels, valid_data, valid_labels)

    def copy(self, limit=None):
        dataset = DataSetBuilder(self.name)
        def calc_nshape(data, value):
            if value is None or not (0 < value <= 1) or data is None:
                value = 1

            limit = int(round(data.shape[0] * value, 0))
            return data[:limit]

        dataset.test_folder_path = self.test_folder_path
        dataset.train_folder_path = self.train_folder_path
        dataset.dataset_path = self.dataset_path
        dataset.train_data = calc_nshape(self.train_data, limit)
        dataset.train_labels = calc_nshape(self.train_labels, limit)
        dataset.valid_data = calc_nshape(self.valid_data, limit)
        dataset.valid_labels = calc_nshape(self.valid_labels, limit)
        dataset.test_data = calc_nshape(self.test_data, limit)
        dataset.test_labels = calc_nshape(self.test_labels, limit)
        dataset.transforms = self.transforms
        dataset.processing_class = self.processing_class
        dataset.md5()
        return dataset

    def processing(self, data, init=True):
        data = self.processing_rows(data)
        if init is True:
            return self.processing_global(data, base_data=data)
        elif init is False and not self.transforms.empty('global'):
            base_data, _ = self.desfragment()
            return self.processing_global(data, base_data=base_data)
        else:
            return data

    def subset(self, percentaje):
        return self.copy(percentaje)

    def md5(self):
        import hashlib
        data, labels = self.desfragment()
        h = hashlib.md5(data)
        self._cached_md5 = h.hexdigest()
        return self._cached_md5

    def _clf(self):
        from ml.clf.extended import RandomForest
        train_labels = np.ones(self.train_data.shape[0])
        test_labels = np.zeros(self.test_data.shape[0])
        valid_labels = np.ones(self.valid_data.shape[0])
        data = np.concatenate((self.train_data, self.valid_data, self.test_data), axis=0)
        labels = np.concatenate((train_labels, valid_labels, test_labels), axis=0)
        dataset = DataSetBuilder(None, transforms_apply=False)
        dataset.build_dataset(data, labels)
        return RandomForest(dataset=dataset)

    def score_train_test(self):
        classif = self._clf()
        classif.train()
        return classif.calc_scores(measures="auc").measures

    def plot(self):
        import matplotlib.pyplot as plt
        last_transform = self.transforms.get_transforms("row")[-1]
        data, labels = self.desfragment()
        if last_transform[0] == "tsne":
            if last_transform[1]["action"] == "concatenate":                
                dim = 2
                features_tsne = data[:,-dim:]
            else:
                features_tsne = data
        else:
            features_tsne = ml.processing.Preprocessing(data, [("tsne", 
                {"perplexity": 50, "action": "replace"})])

        classes = self.labels_info().keys()
        colors = ['b', 'r', 'y', 'm', 'c']
        classes_colors = dict(zip(classes, colors))
        fig, ax = plt.subplots(1, 1, figsize=(17.5, 17.5))

        r_indexes = {}        
        for index, target in enumerate(labels):
            r_indexes.setdefault(target, [])
            r_indexes[target].append(index)

        for target, indexes in r_indexes.items():
            features_index = features_tsne[indexes]
            ax.scatter(
                features_index[:,0], 
                features_index[:,1], 
                color=classes_colors[target], 
                marker='o',
                alpha=.4,
                label=target)
         
        ax.set(xlabel='X',
               ylabel='Y',
               title=self.name)
        ax.legend(loc=2)
        plt.show()

    def density(self):
        if not isinstance(self.train_data.dtype, object):
            data = self.train_data.reshape(self.train_data.shape[0], -1)
        else:
            data = np.asarray([row.reshape(1, -1)[0] for row in self.train_data])
        y = set((elem for row in data for elem in row))
        return float(len(y)) / data.size


class DataSetBuilderImage(DataSetBuilder):
    def __init__(self, name, image_size=None, **kwargs):
        super(DataSetBuilderImage, self).__init__(name, **kwargs)
        self.image_size = image_size
        self.images = []

    def add_img(self, img):
        self.images.append(img)

    def images_from_directories(self, directories):
        if isinstance(directories, str):
            directories = [directories]
        elif isinstance(directories, list):
            pass
        else:
            raise Exception

        images = []
        for root_directory in directories:
            for directory in os.listdir(root_directory):
                files = os.path.join(root_directory, directory)
                if os.path.isdir(files):
                    number_id = directory
                    for image_file in os.listdir(files):
                        images.append((number_id, os.path.join(files, image_file)))
        return images

    def images_to_dataset(self, folder_base):
        images = self.images_from_directories(folder_base)
        labels = np.ndarray(shape=(len(images),), dtype='|S1')
        data = []
        for image_index, (number_id, image_file) in enumerate(images):
            img = io.imread(image_file)
            data.append(img) 
            labels[image_index] = number_id
        data = self.processing(data)
        return data, labels

    @classmethod
    def save_images(self, url, number_id, images, rewrite=False):
        if not os.path.exists(url):
            os.makedirs(url)
        n_url = os.path.join(url, number_id)
        if not os.path.exists(n_url):
             os.makedirs(n_url)

        initial = 0 if rewrite else len(os.listdir(n_url)) 
        for i, image in enumerate(images, initial):
            try:
                image_path = "img-{}-{}.png".format(number_id, i)
                io.imsave(os.path.join(n_url, image_path), image)
            except IndexError:
                print("Index error", n_url, number_id)

    def clean_directory(self, path):
        import shutil
        shutil.rmtree(path)

    def original_to_images_set(self, url, test_folder_data=False):
        images_data, labels = self.labels_images(url)
        images = (self.processing(img, 'image') for img in images_data)
        image_train, image_test = self.build_train_test(zip(labels, images), sample=test_folder_data)
        for number_id, images in image_train.items():
            self.save_images(self.train_folder_path, number_id, images)

        for number_id, images in image_test.items():
            self.save_images(self.test_folder_path, number_id, images)

    def build_dataset(self):
        data, labels = self.images_to_dataset(self.train_folder_path)
        self.shuffle_and_save(data, labels)

    def build_train_test(self, process_images, sample=True):
        images = {}
        images_index = {}
        
        try:
            for number_id, image_array in process_images:
                images.setdefault(number_id, [])
                images[number_id].append(image_array)
        except TypeError:
            #if no faces are detected
            return {}, {}

        if sample is True:
            sample_data = {}
            images_good = {}
            for number_id in images:
                base_indexes = set(range(len(images[number_id])))
                if len(base_indexes) > 3:
                    sample_indexes = set(random.sample(base_indexes, 3))
                else:
                    sample_indexes = set([])
                sample_data[number_id] = [images[number_id][index] for index in sample_indexes]
                images_index[number_id] = base_indexes.difference(sample_indexes)

            for number_id, indexes in images_index.items():
                images_good.setdefault(number_id, [])
                for index in indexes:
                    images_good[number_id].append(images[number_id][index])
            
            return images_good, sample_data
        else:
            return images, {}

    def labels_images(self, urls):
        images_data = []
        labels = []
        if not isinstance(urls, list):
            urls = [urls]

        for url in urls:
            for number_id, path in self.images_from_directories(url):
                images_data.append(io.imread(path))
                labels.append(number_id)
        return images_data, labels

    def copy(self):
        dataset = super(DataSetBuilderImage, self).copy()
        dataset.image_size = self.image_size
        dataset.images = []
        return dataset

    def to_raw(self):
        raw = super(DataSetBuilderImage, self).to_raw()
        new = {'array_length': self.image_size}
        raw.update(new)
        return raw

    def from_raw(self, raw_data):
        super(DataSetBuilderImage, self).from_raw(raw_data)
        #self.transforms.add_transforms("local", raw_data["local_filters"])
        self.image_size = raw_data["array_length"]
        self.desfragment()

    def info(self):
        super(DataSetBuilderImage, self).info()
        print('Image Size {}x{}'.format(self.image_size, self.image_size))


class DataSetBuilderFile(DataSetBuilder):
    def from_csv(self, folder_path, label_column):
        data, labels = self.csv2dataset(folder_path, label_column)
        data = self.processing(data)
        return data, labels

    @classmethod
    def csv2dataset(self, path, label_column):
        df = pd.read_csv(path)
        dataset = df.drop([label_column], axis=1).as_matrix()
        labels = df[label_column].as_matrix()
        return dataset, labels

    def build_dataset(self, label_column=None):
        data, labels = self.from_csv(self.train_folder_path, label_column)
        if self.test_folder_path is not None:
            raise #NotImplemented
            #test_data, test_labels = self.from_csv(self.test_folder_path, label_column)
            #if self.validator == 'cross':
            #    data = np.concatenate((data, test_data), axis=0)
            #    labels = np.concatenate((labels, test_labels), axis=0)

        if self.validator == 'cross':
            self.shuffle_and_save(data, labels)
        else:
            self.adversarial_validator_and_save(data, labels, test_data, test_labels)

    @classmethod
    def merge_data_labels(self, data_path, labels_path, column_id):
        import pandas as pd
        data_df = pd.read_csv(data_path)
        labels_df = pd.read_csv(labels_path)
        return pd.merge(data_df, labels_df, on=column_id)


