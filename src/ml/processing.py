from skimage import color
from skimage import filters
from skimage import transform
from sklearn import preprocessing
from skimage import img_as_ubyte
from skimage import exposure
from collections import OrderedDict
from pydoc import locate

import numpy as np
import json


def pixelate_mode(mode):
    if mode == 'mean':
        return mean
    elif mode == 'min':
        return min
    elif mode == 'max':
        return max


class Transforms(object):
    def __init__(self):
        self.transforms = OrderedDict({})

    def add(self, fn, **params):
        fn_name = "{}.{}".format(fn.__module__, fn.__name__)
        self.transforms[fn_name] = params

    def empty(self):
        return len(self.transforms) == 0

    def __add__(self, o):
        all_transforms = Transforms.from_json(self.to_json())
        for fn, params in o.transforms.items():
            all_transforms.add(locate(fn), **params)
        return all_transforms

    def to_json(self):
        import json
        return json.dumps(self.transforms)

    @classmethod
    def from_json(self, json_transforms):
        transforms_dict = json.loads(json_transforms, object_pairs_hook=OrderedDict)
        transforms = Transforms()
        for fn, params in transforms_dict.items():
            transforms.add(locate(fn), **params)
        return transforms

    def apply(self, data):
        if self.empty() is False:
            for fn, params in self.transforms.items():
                fn = locate(fn)
                data = fn(data, **params)
        if data is None:
            raise Exception
        else:
            return data


class FiT(object):
    def __init__(self, **kwargs):
        self.t = None
        self.params = kwargs

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    def fit(self, data):
        pass

    def dim_rule(self, data):
        return data

    def transform(self, data):
        return self.t(self.dim_rule(data))


class FiTScaler(FiT):
    def dim_rule(self, data):
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        return data

    def fit(self, data):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(**self.params)
        scaler.fit(self.dim_rule(data))
        self.t = scaler.transform


def poly_features(data, degree=2, interaction_only=False, include_bias=True):
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    selector = preprocessing.PolynomialFeatures(
        degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    return selector.fit_transform(data)


def resize(data, image_size=90, type_r="asym"):
    if type_r == "asym":
        dim = []
        for v in data.shape:
            if v > image_size:
                dim.append(image_size)
            else:
                dim.append(v)
    else:
        dim = (image_size, image_size)

    if dim < data.shape or data.shape <= dim:
        try:
            data = transform.resize(data, dim)
        except ZeroDivisionError:
            pass
    return data


def contrast(data):
    #contrast stretching
    p2, p98 = np.percentile(data, (2, 98))
    return exposure.rescale_intensity(data, in_range=(p2, p98))


def upsample(data):
    return transform.pyramid_expand(data, upscale=2, sigma=None, order=1, 
                                    mode='reflect', cval=0)

def rgb2gray(data):
    return color.rgb2gray(data)


def blur(data, level=.2):
    return filters.gaussian(data, level)


def align_face(data):
    from ml.face_detector import FaceAlign
    dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
    align = FaceAlign(dlibFacePredictor)
    return align.process_img(data)


def detector(data):
    from ml.face_detector import DetectorDlib
    dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
    align = DetectorDlib(dlibFacePredictor)
    return align.process_img(data)


def cut(data, rectangle=None):
    top, bottom, left, right = rectangle
    return data[top:bottom, left:right]


def as_ubyte(data):
    return img_as_ubyte(data)


def merge_offset(data, image_size=90, bg_color=1):
    if len(data.shape) == 2:
        return merge_offset2(data, image_size=image_size, bg_color=bg_color)
    elif len(data.shape) == 3:
        return merge_offset3(data, image_size=image_size, bg_color=bg_color)


def merge_offset2(data, image_size=90, bg_color=1):
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

def merge_offset3(data, image_size=90, bg_color=1):
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


def threshold(data, block_size=41):
    return filters.threshold_adaptive(data, block_size, offset=0)


def pixelate(data, pixel_width=None, pixel_height=None, mode='mean'):
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
