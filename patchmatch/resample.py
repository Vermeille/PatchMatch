import numpy as np
from PIL import Image


def resample_size_nearest(x, h, w):
    samp_y = np.round(np.linspace(0, x.shape[0]-1, h)).astype(int)
    samp_x = np.round(np.linspace(0, x.shape[1]-1, w)).astype(int)

    return x[samp_y[:, None], samp_x]

def resample_size_bilinear(x, h, w):
    dt = x.dtype
    x = x.astype(float)
    ys = np.linspace(0, x.shape[0]-1, h)[:, None]
    top = np.floor(ys).astype(int)
    bottom = np.ceil(ys).astype(int)

    xs = np.linspace(0, x.shape[1]-1, w)
    left = np.floor(xs).astype(int)
    right = np.ceil(xs).astype(int)

    tl = x[top, left]
    bl = x[bottom, left]
    tr = x[top, right]
    br = x[bottom, right]

    y_weight = 1 - (ys - top)
    x_weight = 1 - (xs - left)

    return (
            (y_weight * x_weight)[:, :, None] * tl +
            ((1 - y_weight) * x_weight)[:, :, None] * bl +
            (y_weight * (1 - x_weight))[:, :, None] * tr +
            ((1 - y_weight) * (1 - x_weight))[:, :, None] * br
            ).astype(dt)


def resample_size(x, h, w, mode='bilinear'):
    if mode == 'bilinear':
        return resample_size_bilinear(x, h, w)
    elif mode == 'nearest':
        return resample_size_nearest(x, h, w)
    else:
        assert False, 'mode must be bilinear or nearest'

def resample_factor(x, fh, fw, mode='bilinear'):
    return resample_size(x, int(x.shape[0] * fh), int(x.shape[1] * fw), mode)



if __name__ == '__main__':
    img = np.array(Image.open('/home/vermeille/dream_me.jpg'))
    #img = np.random.uniform(high=255, size=(32, 32, 3)).astype('uint8')
    resample_size_nearest(img, 1028, 1028)
    #Image.fromarray(img).show()
    img = resample_size_bilinear(img, 256, 256)
    Image.fromarray(img).show()

