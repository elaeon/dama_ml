from skimage import color
from skimage import filters
from skimage import transform
from sklearn import preprocessing
from skimage import img_as_ubyte
from skimage import exposure
from collections import OrderedDict


class Transforms(object):
    def __init__(self, transforms):        
        self.transforms = {}
        for group, transform in transforms:
            self.add_group_transforms(group, transform)
        #self.placeholders = [k for k, v in self.transforms.items() if v is None]

    #def get_placeholders(self):
    #    return self.placeholders

    def add_transform(self, group, name, value):
        self.transforms[group][name] = value

    def add_transforms(self, group, transforms):
        if not group in transforms:
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

    def scale(self):
        self.data = preprocessing.scale(self.data)

    def pipeline(self):
        if self.transforms is not None:
            for filter_, value in self.transforms:
                if value is not None:
                    getattr(self, filter_)(value)
                else:
                    getattr(self, filter_)()
        return self.data


class PreprocessingImage(Preprocessing):
    def resize(self, image_size):
        if isinstance(image_size, int):
            type_ = "sym"
        elif isinstance(image_size, tuple):
            image_size, type_ = image_size

        if type_ == "asym":
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
        self.data = img_as_ubyte(color.rgb2gray(self.data))

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

    def merge_offset(self, image_size):
        if isinstance(image_size, int):
            bg_color = 1
        elif isinstance(image_size, tuple):
            image_size, bg_color = image_size

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
