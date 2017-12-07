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
import json
import uuid
import sys

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


class TransformsRow(object):
    """
    In this class are deposit the functions for apply to the data.
    
    transforms = Transforms()

    transforms.add(function1, {'a': 1, 'b': 0}) -> function1(a=1, b=0)

    transforms.add(function2, {'x': 10}) -> function2(x=10)
    """
    def __init__(self):
        self.transforms = []

    def __add__(self, o):
        all_transforms = TransformsRow.from_json(self.to_json())
        for fn, params in o.transforms:
            all_transforms.add(locate(fn), **params)
        return all_transforms

    @classmethod
    def cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    def type(self):
        return "row"

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
        import json
        return json.dumps(self.transforms)

    @classmethod
    def from_json(self, json_transforms):
        """
        from json format to Transform class.
        """
        transforms_loaded = json.loads(json_transforms)#, object_pairs_hook=OrderedDict)
        transforms = TransformsRow()
        for fn, params in transforms_loaded:
            transforms.add(locate(fn), **params)
        return transforms

    def apply(self, data, fmtypes=None):
        """
        :type data: array
        :param data: apply the transforms added to the data
        """
        if not isinstance(data, list) and len(data.shape) == 1:
            for fn, params in self.transforms:
                fn = locate(fn)
                data = fn(data, fmtypes=fmtypes, **params)
        else:
            data_n = []
            for row in data:
                for fn, params in self.transforms:
                    fn = locate(fn)
                    row = fn(row, fmtypes=fmtypes, **params)
                data_n.append(row)
            data = np.asarray(data_n) #fixme: add dtype
        if data is None:
            raise Exception
        else:
            return data, fmtypes


class TransformsCol(TransformsRow):
    
    def __add__(self, o):
        all_transforms = TransformsCol.from_json(self.to_json())
        for fn, params in o.transforms:
            all_transforms.add(locate(fn), **params)
        return all_transforms

    def type(self):
        return "column"

    @classmethod
    def from_json(self, json_transforms):
        """
        from json format to Transform class.
        """
        transforms_loaded = json.loads(json_transforms)#, object_pairs_hook=OrderedDict)
        transforms = TransformsCol()
        for fn, params in transforms_loaded:
            transforms.add(locate(fn), **params)
        return transforms

    def initial_fn(self, data, fmtypes=None):
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

            yield fn(data, name=name, fmtypes=fmtypes, **n_params)

    def apply(self, data, fmtypes=None):
        """
        :type data: array
        :param data: apply the transforms added to the data
        """
        for fn_fit in self.initial_fn(data, fmtypes=fmtypes):
            data = np.asarray(list(fn_fit.transform(data)))

        if data is None:
            raise Exception
        else:
            return data, fn_fit.fmtypes

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
        self.types = {"row": TransformsRow, "column": TransformsCol}

    @classmethod
    def cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    def add(self, fn, name=None, type="row", **params):
        """
        :type fn: function
        :param fn: function to add

        :type params: dict
        :param params: the parameters of the function fn

        This function add to the class the functions to use with the data.
        """
        t_class = self.types[type]
        t_obj = t_class()
        if name is not None and type == "column":
            params["name_00_ml"] = name
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
        for transform in o.transforms:
            for fn, params in transform.transforms:
                all_transforms.add(locate(fn), type=transform.type(), **params)
        return all_transforms

    def compact(self):
        """
        transforms a list of transformations to a more compact format

        original = [row, row, row, col, col, row]
        compact = [row, col, row]
        """
        if len(self.transforms) == 1:
            compact_list = [self.transforms[0]]
        elif len(self.transforms) == 0:
            compact_list = []
        else:
            types = {}
            compact_list = []
            for t0, t1 in zip(self.transforms, self.transforms[1:]):
                if t0.type() == t1.type():
                    types[t0.type()] = types.get(t0.type(), t0) + t1
                else:
                    compact_list.append(types.get(t0.type(), t0))
            compact_list.append(types.get(t1.type(), t1))
        return compact_list

    def to_json(self):
        """
        convert this class to json format
        """
        import json
        return json.dumps([{t.type(): t.transforms} for t in self.compact()])

    @classmethod
    def list2transforms(self, transforms_list, transforms):
        for transforms_type in transforms_list:
            for type, transforms_dict in transforms_type.items():
                for fn, params in transforms_dict.items():
                    transforms.add(locate(fn), type=type, **params)

    @classmethod
    def from_json(self, json_transforms):
        """
        from json format to Transform class.
        """
        transforms_list = json.loads(json_transforms, object_pairs_hook=OrderedDict)
        transforms = Transforms()
        for transforms_type in transforms_list:
            for type_, transforms_dict in transforms_type.items():
                for fn, params in transforms_dict:
                    try:
                        transforms.add(locate(fn), type=type_, **params)
                    except Exception, e:
                        print(e.message)
        return transforms

    def apply(self, data, fmtypes=None):
        """
        :type data: array
        :param data: apply the transforms added to the data
        """
        if self.is_empty():
            return data
        else:
            transforms = self.compact()
            for t_obj in transforms:
                log.debug("APPLY TRANSFORMS:" + str(t_obj.transforms))
                log.debug("Transform type:" + t_obj.type())
                data, fmtypes = t_obj.apply(data, fmtypes=fmtypes)

            if data is None:
                raise Exception
            else:
                return data

    def destroy(self):
        for transform in self.compact():
            if hasattr(transform, 'destroy'):
                transform.destroy()
            

class Fit(object):
    def __init__(self, data, name=None, fmtypes=None, path="", **kwargs):
        self.name = name if name is not None else uuid.uuid4().hex        
        self.meta_path = path + self.module_cls_name() + "_" + self.name
        self.fmtypes = fmtypes
        self.t = self.fit(data, **kwargs)

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    def fit(self, data, **params):
        pass

    def dim_rule(self, data):
        return data

    def transform(self, data):
        from ml.layers import IterLayer
        return IterLayer(self.t(self.dim_rule(data)))

    def read_meta(self):
        from ml.ds import load_metadata
        return load_metadata(self.meta_path)

    def write_meta(self, data):
        from ml.ds import save_metadata
        save_metadata(self.meta_path, data)


class FitStandardScaler(Fit):
    def dim_rule(self, data):
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        return data

    def fit(self, data, **params):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(**params)
        scaler.fit(self.dim_rule(data))
        return scaler.transform


class FitRobustScaler(Fit):
    def dim_rule(self, data):
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        return data

    def fit(self, data, **params):
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler(**params)
        scaler.fit(self.dim_rule(data))
        return scaler.transform


class FitTruncatedSVD(Fit):
    def dim_rule(self, data):
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        return data

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

        tsne = PTsne(model_name=self.name, model_version="1", autoload=False)
        if not tsne.exist():
            dataset = Data(dataset_path="/tmp", dtype="float64")
            dataset.build_dataset(data)
            tsne = PTsne(model_name=self.name, model_version="1", 
                        dataset=dataset, latent_dim=2, dtype='float64')
            tsne.train(batch_size=50, num_steps=4)
        else:
            tsne.load_dataset(None)
        self.model = tsne
        return tsne.predict

    def transform(self, data):
        from itertools import izip
        from ml.layers import IterLayer

        def iter_():
            for row, predict in izip(data, self.t(self.dim_rule(data), chunk_size=5000)):
                yield np.append(row, list(predict), axis=0)

        return IterLayer(iter_())

    def destroy(self):
        if hasattr(self, 'model'):
            self.model.destroy()


class FitReplaceNan(Fit):
    def dim_rule(self, data):
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        return data
    
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
            for row in n_data:
                indx = np.where(np.isnan(row))
                for i in indx[0]:
                    row[i] = columns[i]
                yield row
        
        return transform

    def destroy(self):
        from ml.utils.files import rm
        rm(self.meta_path)
        


def resize(data, fmtypes=None, image_size_h=90, image_size_w=90):
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


def contrast(data, fmtypes=None):
    """
    :type data: array
    :param data: data to transform

    add contrast stretching to the data.
    """
    #contrast stretching
    p2, p98 = np.percentile(data, (2, 98))
    return exposure.rescale_intensity(data, in_range=(p2, p98))


def upsample(data, fmtypes=None):
    """
    :type data: array
    :param data: data to transform

    apply pyramid expand with params upscale 2 and order 1, mode reflect.
    """
    return transform.pyramid_expand(data, upscale=2, sigma=None, order=1, 
                                    mode='reflect', cval=0)

def rgb2gray(data, fmtypes=None):
    """
    :type data: array
    :param data: data to transform

    convert an image to gray scale
    """
    return color.rgb2gray(data)


def blur(data, fmtypes=None, level=.2):
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


def cut(data, fmtypes=None, rectangle=None):
    """
    :type data: array
    :param data: data to cut    

    :type rectangle: tuple
    :param rectangle: (top, bottom, left, right)

    return the data restricted inside the rectangle.
    """
    top, bottom, left, right = rectangle
    return data[top:bottom, left:right]


def as_ubyte(data, fmtypes=None):
    return img_as_ubyte(data)


def merge_offset(data, fmtypes=None, image_size=90, bg_color=1):
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
    if len(data.shape) == 2:
        return merge_offset2(data, image_size=image_size, bg_color=bg_color)
    elif len(data.shape) == 3:
        return merge_offset3(data, image_size=image_size, bg_color=bg_color)


def merge_offset2(data, fmtypes=None, image_size=90, bg_color=1):
    bg = np.ones((image_size, image_size))
    offset = (int(round(abs(bg.shape[0] - data.shape[0]) / 2)), 
            int(round(abs(bg.shape[1] - data.shape[1]) / 2)))
    pos_v, pos_h = offset
    v_range1 = slice(max(0, pos_v), max(min(pos_v + data.shape[0], bg.shape[0]), 0))
    h_range1 = slice(max(0, pos_h), max(min(pos_h + data.shape[1], bg.shape[1]), 0))
    v_range2 = slice(max(0, -pos_v), min(-pos_v + bg.shape[0], data.shape[0]))
    h_range2 = slice(max(0, -pos_h), min(-pos_h + bg.shape[1], data.shape[1]))
    if bg_color is None:
        bg2 = bg - 1 + np.average(data) + random.uniform(-np.var(data), np.var(data))
    elif bg_color == 1:
        bg2 = bg
    else:
        bg2 = bg - 1
    
    bg2[v_range1, h_range1] = bg[v_range1, h_range1] - 1
    bg2[v_range1, h_range1] = bg2[v_range1, h_range1] + data[v_range2, h_range2]
    return bg2

def merge_offset3(data, fmtypes=None, image_size=90, bg_color=1):
    bg = np.ones((image_size, image_size, 3))
    offset = (int(round(abs(bg.shape[0] - data.shape[0]) / 2)), 
            int(round(abs(bg.shape[1] - data.shape[1]) / 2)),
             int(round(abs(bg.shape[2] - data.shape[2]) / 2)))
    pos_v, pos_h, pos_w = offset
    v_range1 = slice(max(0, pos_v), max(min(pos_v + data.shape[0], bg.shape[0]), 0))
    h_range1 = slice(max(0, pos_h), max(min(pos_h + data.shape[1], bg.shape[1]), 0))
    w_range1 = slice(max(0, pos_w), max(min(pos_w + data.shape[2], bg.shape[2]), 0))
    v_range2 = slice(max(0, -pos_v), min(-pos_v + bg.shape[0], data.shape[0]))
    h_range2 = slice(max(0, -pos_h), min(-pos_h + bg.shape[1], data.shape[1]))
    w_range2 = slice(max(0, -pos_w), min(-pos_w + bg.shape[2], data.shape[2]))
    if bg_color is None:
        bg2 = bg - 1 + np.average(data) + random.uniform(-np.var(data), np.var(data))
    elif bg_color == 1:
        bg2 = bg
    else:
        bg2 = bg - 1
    
    bg2[v_range1, h_range1, w_range1] = bg[v_range1, h_range1, w_range1] - 1
    bg2[v_range1, h_range1, w_range1] = bg2[v_range1, h_range1, w_range1] + data[v_range2, h_range2, w_range2]
    return bg2


def threshold(data, fmtypes=None, block_size=41):
    return filters.threshold_adaptive(data, block_size, offset=0)


def pixelate(data, fmtypes=None, pixel_width=None, pixel_height=None, mode='mean'):
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


def drop_columns(row, fmtypes=None, exclude_cols=None, include_cols=None):
    if include_cols is not None:
        row = row[include_cols]
    elif exclude_cols is not None:
        features = set(x for x in xrange(len(row)))
        to_keep = list(features.difference(set(exclude_cols)))
        row = row[to_keep]
    return row
