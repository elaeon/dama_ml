from skimage import color
from skimage import filters
from skimage import transform
from sklearn import preprocessing
from skimage import img_as_ubyte
from skimage import exposure
from collections import OrderedDict
from pydoc import locate

import logging
import numpy as np
import pandas as pd
import json
import uuid
import sys
import inspect

from ml.utils.config import get_settings
from ml.layers import IterLayer

settings = get_settings("ml")

log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))

if not settings["class_path"] in sys.path:
    sys.path.insert(0, settings["class_path"])


def pixelate_mode(mode):
    if mode == 'mean':
        return mean
    elif mode == 'min':
        return min
    elif mode == 'max':
        return max


class TransformsFn(object):
    """
    In this class are deposit the functions for apply to the data.
    
    transforms = Transforms()

    transforms.add(function1, {'a': 1, 'b': 0}) -> function1(a=1, b=0)

    transforms.add(function2, {'x': 10}) -> function2(x=10)
    """
    def __init__(self, input_dtype='float'):
        self.transforms = []

    def __add__(self, o):
        all_transforms = TransformsFn.from_json(self.to_json())
        for fn, params in o.transforms:
            all_transforms.add(locate(fn), **params)
        return all_transforms

    @classmethod
    def cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    @classmethod
    def type(self):
        return "fn"

    def add(self, fn, **params):
        """
        :type fn: function
        :param fn: function to add

        :type params: dict
        :param params: the parameters of the function fn

        This function add to the class the functions to use with the data.
        """
        if hasattr(fn, '__module__'):
            if fn.__module__ == "__main__":
                from ml.utils.files import path2module
                from ml.utils.config import get_settings
                settings = get_settings("ml")
                fn_module = path2module(settings["class_path"])
            else:
                fn_module = fn.__module__
            fn_name = "{}.{}".format(fn_module, fn.__name__)
            self.transforms.append((fn_name, params))
            
    def is_empty(self):
        """
        return True if not transforms was added.
        """
        return len(self.transforms) == 0

    def to_json(self):
        """
        convert this class to json format
        """
        return json.dumps(self.info())

    def info(self):
        return {"transforms": self.transforms}

    @classmethod
    def from_json(self, json_transforms):
        """
        from json format to Transform class.
        """
        return self._from_json(json_transforms, TransformsFn)
        
    @classmethod
    def _from_json(self, json_transforms, transform_base_class):
        transforms_loaded = json.loads(json_transforms)#, object_pairs_hook=OrderedDict)
        transforms = transform_base_class()
        for fn, params in transforms_loaded["transforms"]:
            transforms.add(locate(fn), **params)
        return transforms

    def apply(self, data, chunks_size=258):
        """
        :type data: array
        :param data: apply the transforms to the data
        """
        if isinstance(data, np.ndarray):
            data = IterLayer(data, shape=data.shape, dtype=data.dtype).to_chunks(chunks_size)
        elif isinstance(data, pd.DataFrame):
            data = IterLayer(data, shape=data.shape).to_chunks(chunks_size)

        def iter_():
            locate_fn = {}
            for smx in data:
                for fn_, params in self.transforms:
                    fn = locate_fn.setdefault(fn_, locate(fn_))
                    smx = fn(smx, **params)
                yield smx
        
        return IterLayer(iter_(), shape=data.shape, chunks_size=chunks_size, 
                    has_chunks=True)


class TransformsClass(TransformsFn):
    
    @classmethod
    def type(self):
        return "class"

    @classmethod
    def from_json(self, json_transforms):
        """
        from json format to Transform class.
        """
        return self._from_json(json_transforms, TransformsClass)

    def initial_fn(self, data):
        for fn, params in self.transforms:
            fn = locate(fn)
            try:
                name = params["name_00_ml"]
            except KeyError:
                name = None
                n_params = params
            else:
                n_params = params.copy()
                del n_params["name_00_ml"]

            yield fn(data, name=name, **n_params)

    def apply(self, data, chunks_size=None):
        """
        :type data: array
        :param data: apply the transforms added to the data
        """
        if isinstance(data, IterLayer):
            data = data.to_memory()
        for fn_fit in self.initial_fn(data):
            data = fn_fit.transform(data).to_memory()

        if not isinstance(data, np.ndarray):
            dtype = [(name, data.columns.dtype.str) for name in data.columns]
        else:
            dtype = data.dtype
        return IterLayer(data, shape=data.shape, dtype=dtype).to_chunks(chunks_size=chunks_size)

    def destroy(self):
        for transform in self.initial_fn(None):
            if hasattr(transform, 'destroy'):
                transform.destroy()


class Transforms(object):
    """
    In this class are deposit the functions for apply to the data.
    
    transforms = Transforms()

    transforms.add(function1, {'a': 1, 'b': 0}) -> function1(a=1, b=0)

    transforms.add(function2, {'x': 10}) -> function2(x=10)
    """
    def __init__(self):
        self.transforms = []

    @classmethod
    def cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    def add(self, fn, name=None, **params):
        """
        :type fn: function
        :param fn: function to add

        :type params: dict
        :param params: the parameters of the function fn

        This function add to the class the functions to use with the data.
        """
        t_class = TransformsClass if inspect.isclass(fn) else TransformsFn
        if name is not None and t_class.type() == TransformsClass.type():
            params["name_00_ml"] = name

        if not self.is_empty():
            last_t_obj = self.transforms[-1]
            if t_class.type() == last_t_obj.type():
                last_t_obj.add(fn, **params)
            else:
                t_obj = t_class()
                t_obj.add(fn, **params)
                self.transforms.append(t_obj)
        else:
            t_obj = t_class()
            t_obj.add(fn, **params)
            self.transforms.append(t_obj)

    def is_empty(self):
        """
        return True if not transforms was added.
        """
        return len(self.transforms) == 0

    def clean(self):
        """
        clean the transformations in the object
        """
        self.transforms  = []

    def __add__(self, o):
        all_transforms = Transforms.from_json(self.to_json())
        if o is None:
            return all_transforms
        for transform in o.transforms:
            for fn, params in transform.transforms:
                all_transforms.add(locate(fn), **params)
        return all_transforms

    def to_json(self):
        """
        convert this class to json format
        """
        import json
        return json.dumps([{t.type(): t.info()} for t in self.transforms])

    @classmethod
    def from_json(self, json_transforms):
        """
        from json format to Transform class.
        """
        transforms_list = json.loads(json_transforms, object_pairs_hook=OrderedDict)
        transforms = Transforms()
        for transforms_type in transforms_list:
            for type_, transforms_dict in transforms_type.items():
                for fn_str, params in transforms_dict["transforms"]:
                    fn = locate(fn_str)
                    if fn is not None:
                        transforms.add(fn, **params)
                    else:
                        log.debug("Function {} not found in class path".format(fn_str))
        return transforms

    def apply(self, data, chunks_size=258):
        """
        :type data: array
        :param data: apply the transforms added to the data
        """
        if not self.is_empty():
            for t_obj in self.transforms:
                log.debug("APPLY TRANSFORMS:" + str(t_obj.transforms))
                log.debug("Transform type:" + t_obj.type())
                data = t_obj.apply(data, chunks_size=chunks_size)
        return data

    def destroy(self):
        for transform in self.transforms:
            if hasattr(transform, 'destroy'):
                transform.destroy()


class Fit(object):
    def __init__(self, data, name=None, path="", **kwargs):
        self.name = name if name is not None else uuid.uuid4().hex        
        self.meta_path = path + self.module_cls_name() + "_" + self.name
        self.t = self.fit(data, **kwargs)

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    def fit(self, data, **params):
        pass

    def transform(self, data):
        ndata = self.t(data)
        return IterLayer(ndata, shape=ndata.shape, dtype=ndata.dtype)

    def read_meta(self):
        from ml.ds import load_metadata
        return load_metadata(self.meta_path)

    def write_meta(self, data):
        from ml.ds import save_metadata
        save_metadata(self.meta_path, data)


class FitStandardScaler(Fit):

    def fit(self, data, **params):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(**params)
        scaler.fit(data)
        return scaler.transform


class FitRobustScaler(Fit):
    def fit(self, data, **params):
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler(**params)
        scaler.fit(data)
        return scaler.transform


class FitTruncatedSVD(Fit):
    def fit(self, data, **params):
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(**params)
        svd.fit(data)
        return svd.transform


class FitTsne(Fit):
    def dim_rule(self, data):
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        elif len(data.shape) == 1:
            data = data.reshape(1, data.shape)
        return data

    def fit(self, data, **params):
        from ml.ae.extended.w_keras import PTsne
        from ml.ds import Data

        tsne = PTsne(model_name=self.name)
        if not tsne.exist():
            dataset = Data(dataset_path="/tmp", rewrite=True)
            with dataset:
                dataset.build_dataset(data)
            tsne = PTsne(model_name=self.name, latent_dim=2)
            tsne.set_dataset(dataset)
            tsne.train(batch_size=50, num_steps=4)
            tsne.save(model_version="1")
        else:
            tsne.load(model_version="1")
        self.model = tsne
        return tsne.predict

    def transform(self, data):
        from itertools import izip

        def iter_():
            for row, predict in izip(data, self.t(self.dim_rule(data), chunks_size=5000)):
                yield np.append(row, list(predict), axis=0)

        return IterLayer(iter_(), shape=(data.shape[0], data.shape[1]+2), dtype=data.dtype)

    def destroy(self):
        if hasattr(self, 'model'):
            self.model.destroy()


class FitReplaceNan(Fit):
    def fit(self, data, **params):
        from ml.utils.numeric_functions import is_binary, is_integer
        if len(self.read_meta()) == 0:
            columns = {}
            for i, column in enumerate(data.T):
                if any(np.isnan(column)):
                    if is_binary(column):
                        replace_value = -1
                    elif is_integer(column):
                        replace_value = -1
                    else:
                        replace_value = np.nanpercentile(column, [50])[0]
                    columns[i] = replace_value
            self.write_meta(columns)

        def transform(n_data):
            columns = self.read_meta()
            def iter_():
                for row in n_data:
                    indx = np.where(np.isnan(row))
                    for i in indx[0]:
                        row[i] = columns[i]
                    yield row
            return IterLayer(iter_(), shape=n_data.shape)
        
        return transform

    def destroy(self):
        from ml.utils.files import rm
        rm(self.meta_path)
        


def resize(data, image_size_h=90, image_size_w=90):
    """
    :type data: array
    :param data: data to be resized

    :type image_size_h: int
    :param image_size_h: reduce the image height to this size.

    :type image_size_w: int
    param image_size_w: reduce the image weight to this size.
    """
    dim = (image_size_h, image_size_w)
    if dim < data.shape or data.shape <= dim:
        try:
            data = transform.resize(data, dim)
        except ZeroDivisionError:
            pass
    return data


def contrast(data):
    """
    :type data: array
    :param data: data to transform

    add contrast stretching to the data.
    """
    #contrast stretching
    p2, p98 = np.percentile(data, (2, 98))
    return exposure.rescale_intensity(data, in_range=(p2, p98))


def upsample(data):
    """
    :type data: array
    :param data: data to transform

    apply pyramid expand with params upscale 2 and order 1, mode reflect.
    """
    return transform.pyramid_expand(data, upscale=2, sigma=None, order=1, 
                                    mode='reflect', cval=0)

def rgb2gray(data):
    """
    :type data: array
    :param data: data to transform

    convert an image to gray scale
    """
    return color.rgb2gray(data)


def blur(data, level=.2):
    """
    :type data: array
    :param data: data to transform

    apply gaussian blur to the data.
    """
    return filters.gaussian(data, level)


#def align_face(data):
#    from ml.face_detector import FaceAlign
#    dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
#    align = FaceAlign(dlibFacePredictor)
#    return align.process_img(data)


#def detector(data):
#    from ml.face_detector import DetectorDlib
#    dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
#    align = DetectorDlib(dlibFacePredictor)
#    return align.process_img(data)


def cut(data, rectangle=None):
    """
    :type data: array
    :param data: data to cut    

    :type rectangle: tuple
    :param rectangle: (top, bottom, left, right)

    return the data restricted inside the rectangle.
    """
    top, bottom, left, right = rectangle
    return data[top:bottom, left:right]


def as_ubyte(data):
    return img_as_ubyte(data)


def merge_offset(data, image_size=90, bg_color=1):
    """
    :type data: array
    :param data: data transform

    :type image_size: int
    :param image_size: the new image size

    :type bg_color: float
    :param bg_color: value in [0, 1] for backgroung color

    transform a rectangular image of (with, height) or (widh, height, channel) to 
    a squared image of size (image_size, image_size) 
    """
    if len(data.shape) == 3:
        return merge_offset2(data, image_size=image_size, bg_color=bg_color)
    elif len(data.shape) == 4:
        return merge_offset3(data, image_size=image_size, bg_color=bg_color)


def merge_offset2(data, image_size=90, bg_color=1):
    bg = np.ones((image_size, image_size))
    ndata = np.empty((data.shape[0], image_size, image_size), dtype=data.dtype)
    for i, image in enumerate(data):
        offset = (int(round(abs(bg.shape[0] - image.shape[0]) / 2)), 
                int(round(abs(bg.shape[1] - image.shape[1]) / 2)))
        pos_v, pos_h = offset
        v_range1 = slice(max(0, pos_v), max(min(pos_v + image.shape[0], bg.shape[0]), 0))
        h_range1 = slice(max(0, pos_h), max(min(pos_h + image.shape[1], bg.shape[1]), 0))
        v_range2 = slice(max(0, -pos_v), min(-pos_v + bg.shape[0], image.shape[0]))
        h_range2 = slice(max(0, -pos_h), min(-pos_h + bg.shape[1], image.shape[1]))
        if bg_color is None:
            bg2 = bg - 1 + np.average(image) + random.uniform(-np.var(image), np.var(image))
        elif bg_color == 1:
            bg2 = bg
        else:
            bg2 = bg - 1
        
        bg2[v_range1, h_range1] = bg[v_range1, h_range1] - 1
        bg2[v_range1, h_range1] = bg2[v_range1, h_range1] + image[v_range2, h_range2]
    return ndata


def merge_offset3(data, image_size=90, bg_color=1):
    ndata = np.empty((data.shape[0], image_size, image_size, 3), dtype=data.dtype)
    for i, image in enumerate(data):
        bg = np.ones((image_size, image_size, 3))
        offset = (int(round(abs(bg.shape[0] - image.shape[0]) / 2)), 
                int(round(abs(bg.shape[1] - image.shape[1]) / 2)),
                 int(round(abs(bg.shape[2] - image.shape[2]) / 2)))
        pos_v, pos_h, pos_w = offset
        v_range1 = slice(max(0, pos_v), max(min(pos_v + image.shape[0], bg.shape[0]), 0))
        h_range1 = slice(max(0, pos_h), max(min(pos_h + image.shape[1], bg.shape[1]), 0))
        w_range1 = slice(max(0, pos_w), max(min(pos_w + image.shape[2], bg.shape[2]), 0))
        v_range2 = slice(max(0, -pos_v), min(-pos_v + bg.shape[0], image.shape[0]))
        h_range2 = slice(max(0, -pos_h), min(-pos_h + bg.shape[1], image.shape[1]))
        w_range2 = slice(max(0, -pos_w), min(-pos_w + bg.shape[2], image.shape[2]))
        if bg_color is None:
            bg2 = bg - 1 + np.average(image) + random.uniform(-np.var(image), np.var(image))
        elif bg_color == 1:
            bg2 = bg
        else:
            bg2 = bg - 1
        
        bg2[v_range1, h_range1, w_range1] = bg[v_range1, h_range1, w_range1] - 1
        bg2[v_range1, h_range1, w_range1] = bg2[v_range1, h_range1, w_range1] + image[v_range2, h_range2, w_range2]
        ndata[i] = bg2
    return ndata


def threshold(data, block_size=41):
    return filters.threshold_adaptive(data, block_size, offset=0)


def pixelate(data, pixel_width=None, pixel_height=None, mode='mean'):
    """
    :type data: array
    :param data: data to pixelate

    :type pixel_width: int
    :param pixel_width: pixel with in the image

    :type pixel_height: int
    :param pixel_height: pixel height in the image

    :type mode: string
    :param mode: mean, min or max

    add pixelation to the image in de data.
    """
    #import time
    #start_time = time.time()
    if len(data.shape) > 2:
        width, height, channels = data.shape
    else:
        width, height = data.shape

    data = np.ndarray(shape=data.shape, dtype=np.uint8)
    modefunc = pixelate_mode(mode)
    for w in range(0, width, pixel_width):
        minx, maxx = w, min(width, w + pixel_width)
        for h in range(0, height, pixel_height):
            miny, maxy = h, min(height, h + pixel_height)
            color = tuple(modefunc([data[x][y]
                    for x in xrange(minx, maxx)
                    for y in xrange(miny, maxy)]))
            for x in xrange(minx, maxx):
                for y in xrange(miny, maxy):
                    data[x][y] = color
    #print("--- %s seconds ---" % (time.time() - start_time))
    return data


def drop_columns(row, exclude_cols=None, include_cols=None):
    if include_cols is not None:
        return row[:, include_cols]
    elif exclude_cols is not None:
        features = set(x for x in xrange(row.shape[1]))
        to_keep = list(features.difference(set(exclude_cols)))
        return row[:, to_keep]
