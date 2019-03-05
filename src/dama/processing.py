from skimage import color
from skimage import filters
from skimage import transform
from skimage import img_as_ubyte
from skimage import exposure
import numpy as np
import random

from dama.utils.logger import log_config

log = log_config(__name__)


def pixelate_mode(mode: str):
    if mode == 'mean':
        return np.mean
    elif mode == 'min':
        return min
    elif mode == 'max':
        return max


def resize(data: np.ndarray, image_size_h: int = 90, image_size_w: int = 90) -> np.ndarray:
    dim = (image_size_h, image_size_w)
    if dim < data.shape or data.shape <= dim:
        try:
            data = transform.resize(data, dim)
        except ZeroDivisionError:
            pass
    return data


def contrast(data: np.array) -> np.array:
    """
    add contrast stretching to the data.
    """
    p2, p98 = np.percentile(data, (2, 98))
    return exposure.rescale_intensity(data, in_range=(p2, p98))


def upsample(data: np.ndarray) -> np.array:
    """
    :type data: array
    :param data: data to transform

    apply pyramid expand with params upscale 2 and order 1, mode reflect.
    """
    return transform.pyramid_expand(data, upscale=2, sigma=None, order=1, 
                                    mode='reflect', cval=0)


def rgb2gray(data: np.ndarray) -> np.ndarray:
    """
    convert an image to gray scale
    """
    return color.rgb2gray(data)


def blur(data: np.ndarray, level: float = .2) -> np.ndarray:
    return filters.gaussian(data, level)


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


def merge_offset(data: np.ndarray, image_size: int = 90, bg_color: int = 1) -> np.ndarray:
    """
    transform a rectangular image of (with, height) or (widh, height, channel) to
    a squared image of size (image_size, image_size)
    """
    if len(data.shape) == 3:
        return merge_offset2d(data, image_size=image_size, bg_color=bg_color)
    elif len(data.shape) == 4:
        return merge_offset3d(data, image_size=image_size, bg_color=bg_color)


def merge_offset2d(data: np.ndarray, image_size: int = 90, bg_color: int = 1) -> np.ndarray:
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


def merge_offset3d(data: np.ndarray, image_size: int = 90, bg_color: int = 1) -> np.ndarray:
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


def threshold(data: np.ndarray, block_size: int = 41) -> np.array:
    return filters.threshold_adaptive(data, block_size, offset=0)


def pixelate(data: np.ndarray, pixel_width: int = None, pixel_height: int = None, mode: str = 'mean') -> np.ndarray:
    """
    add pixelation to the image.
    """
    # import time
    # start_time = time.time()
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
            color_p = tuple(modefunc([data[x][y]
                                      for x in range(minx, maxx) for y in range(miny, maxy)]))
            for x in range(minx, maxx):
                for y in range(miny, maxy):
                    data[x][y] = color_p
    # print("--- %s seconds ---" % (time.time() - start_time))
    return data


def drop_columns(row, exclude_cols=None, include_cols=None) -> np.ndarray:
    if include_cols is not None:
        return row[:, include_cols]
    elif exclude_cols is not None:
        features = set(x for x in range(row.shape[1]))
        to_keep = list(features.difference(set(exclude_cols)))
        return row[:, to_keep]
