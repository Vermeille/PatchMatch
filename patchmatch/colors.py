import numpy as np
from PIL import Image, ImageCms


def rgb_to_yuv(rgb):
    m = np.array([[0.29900, -0.16874, 0.50000], [0.58700, -0.33126, -0.41869],
                  [0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb, m)
    yuv[:, :, 1:] += 128.0
    return yuv.astype('float32')


def yuv_to_rgb(yuv):
    m = np.array(
        [[1.0, 1.0, 1.0],
         [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
         [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

    rgb = np.dot(yuv, m)
    rgb[:, :, 0] -= 179.45477266423404
    rgb[:, :, 1] += 135.45870971679688
    rgb[:, :, 2] -= 226.8183044444304
    return rgb.clip(0, 255).astype('float32')


# Convert to Lab colourspace
srgb_p = ImageCms.createProfile("sRGB")
lab_p = ImageCms.createProfile("LAB")

rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")


def rgb_to_lab(x):
    img = Image.fromarray(x.astype('uint8'), mode='RGB')
    return np.array(ImageCms.applyTransform(img, rgb2lab)).astype('float32')


def lab_to_rgb(x):
    img = Image.fromarray(x.astype('uint8'), mode='LAB')
    return np.array(ImageCms.applyTransform(img, lab2rgb)).astype('float32')


def rgb_to(x, space):
    if space == 'yuv':
        return rgb_to_yuv(x)
    if space == 'lab':
        return rgb_to_lab(x)
    if space == 'rgb':
        return x
    print('no such color space')
    assert False
