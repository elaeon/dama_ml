from skimage import color
from skimage import filters
from skimage import transform
from sklearn import preprocessing
from skimage import img_as_ubyte
from skimage import exposure
from collections import OrderedDict
import numpy as np


def mean(array):
    return np.mean(array, axis=0, dtype=np.int32)

def pixelate_mode(mode):
    if mode == 'mean':
        return mean
    elif mode == 'min':
        return min
    elif mode == 'max':
        return max


class Transforms(object):
    def __init__(self, transforms):        
        self.transforms = {}
        for group, transform in transforms:
            self.add_group_transforms(group, transform)

    def add_first_transform(self, group, name, value):
        tail = self.transforms[group]
        if name in tail:
            if name == tail.iterkeys().next():
                self.add_transform(group, name, value)
            else:
                del tail[name]
                self.transforms[group] = OrderedDict({name: value})                
                self.transforms[group].update(tail)
        else:
            self.transforms[group] = OrderedDict({name: value})
            self.transforms[group].update(tail)

    def add_transform(self, group, name, value):
        try:
            self.transforms[group][name] = value
        except KeyError:
            self.transforms[group] = OrderedDict({name: value})

    def add_transforms(self, group, transforms):
        if not group in self.transforms:
             self.add_group_transforms(group, transforms)
        else:
            for name, value in transforms.items():
                self.transforms[group][name] = value

    def get_transforms(self, group):
        return self.transforms[group].items()

    def get_all_transforms(self):
        return [(key, list(self.transforms[key].items())) 
            for key in self.transforms]

    def add_group_transforms(self, group, transforms):
        self.transforms[group] = OrderedDict(transforms)

    def empty(self):
        return len(self.transforms) == 0


class Preprocessing(object):
    
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms
    
    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)
    
    def scale(self):   
        self.data = preprocessing.scale(self.data)
        
    def poly_features(self, degree=2, interaction_only=False, include_bias=True):
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(1, -1)
        selector = preprocessing.PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        self.data = selector.fit_transform(self.data)

    #def tsne(self, perplexity=50, action='concatenate'):
    #    from bhtsne import tsne
    #    data_reduction = tsne(self.data, perplexity=perplexity)
    #    if action == 'concatenate':
    #        self.data = np.concatenate((self.data, data_reduction), axis=1)
    #    elif action == 'replace':
    #        self.data = data_reduction

    def pipeline(self):
        if self.transforms is not None:
            for filter_, value in self.transforms:
                if isinstance(value, dict):
                    getattr(self, filter_)(**value)
                else:
                    getattr(self, filter_)()
        return self.data


class FiT(object):
    def __init__(self, data):
        self.t = None

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    def fit(self, data):
        pass

    def transform(self, data):
        return self.t.transform(data)


class FiTScaler(FiT):
    def fit(self, data):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(data)
        self.t = scaler


class PreprocessingImage(Preprocessing):
    def resize(self, image_size=90, type_r="asym"):
        if type_r == "asym":
            dim = []
            for v in self.data.shape:
                if v > image_size:
                    dim.append(image_size)
                else:
                    dim.append(v)
        else:
            dim = (image_size, image_size)
        if dim < self.data.shape or self.data.shape <= dim:
            try:
                self.data = transform.resize(self.data, dim)
            except ZeroDivisionError:
                pass
        
    def contrast(self):
        #contrast stretching
        p2, p98 = np.percentile(self.data, (2, 98))
        self.data = exposure.rescale_intensity(self.data, in_range=(p2, p98))

    def upsample(self):
        self.data = transform.pyramid_expand(
            self.data, upscale=2, sigma=None, order=1, mode='reflect', cval=0)

    def rgb2gray(self):
        self.data = color.rgb2gray(self.data)

    def blur(self, level):
        self.data = filters.gaussian(self.data, level)

    def align_face(self):
        from ml.face_detector import FaceAlign
        dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
        align = FaceAlign(dlibFacePredictor)
        self.data = align.process_img(self.data)

    def detector(self):
        from ml.face_detector import DetectorDlib
        dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
        align = DetectorDlib(dlibFacePredictor)
        self.data = align.process_img(self.data)

    def cut(self, rectangle):
        top, bottom, left, right = rectangle
        self.data = self.data[top:bottom, left:right]

    def as_ubyte(self):
        self.data = img_as_ubyte(self.data)

    def merge_offset(self, image_size=90, bg_color=1):
        bg = np.ones((image_size, image_size))
        offset = (int(round(abs(bg.shape[0] - self.data.shape[0]) / 2)), 
                int(round(abs(bg.shape[1] - self.data.shape[1]) / 2)))
        pos_v, pos_h = offset
        v_range1 = slice(max(0, pos_v), max(min(pos_v + self.data.shape[0], bg.shape[0]), 0))
        h_range1 = slice(max(0, pos_h), max(min(pos_h + self.data.shape[1], bg.shape[1]), 0))
        v_range2 = slice(max(0, -pos_v), min(-pos_v + bg.shape[0], self.data.shape[0]))
        h_range2 = slice(max(0, -pos_h), min(-pos_h + bg.shape[1], self.data.shape[1]))
        if bg_color is None:
            #print(np.std(image))
            #print(np.var(image))
            bg2 = bg - 1 + np.average(self.data) + random.uniform(-np.var(self.data), np.var(self.data))
        elif bg_color == 1:
            bg2 = bg
        else:
            bg2 = bg - 1
        
        bg2[v_range1, h_range1] = bg[v_range1, h_range1] - 1
        bg2[v_range1, h_range1] = bg2[v_range1, h_range1] + self.data[v_range2, h_range2]
        self.data = bg2

    def threshold(self, block_size=41):
        self.data = filters.threshold_adaptive(self.data, block_size, offset=0)

    def pixelate(self, pixel_width=None, pixel_height=None, mode='mean'):
        #import time
        #start_time = time.time()
        if len(self.data.shape) > 2:
            width, height, channels = self.data.shape
        else:
            width, height = self.data.shape

        data = np.ndarray(shape=self.data.shape, dtype=np.uint8)
        modefunc = pixelate_mode(mode)
        for w in range(0, width, pixel_width):
            minx, maxx = w, min(width, w + pixel_width)
            for h in range(0, height, pixel_height):
                miny, maxy = h, min(height, h + pixel_height)
                color = tuple(modefunc([self.data[x][y]
                        for x in xrange(minx, maxx)
                        for y in xrange(miny, maxy)]))
                for x in xrange(minx, maxx):
                    for y in xrange(miny, maxy):
                        data[x][y] = color
        #print("--- %s seconds ---" % (time.time() - start_time))
        self.data = data
