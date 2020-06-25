import numpy as np
from PIL import Image
from patchmatch.resample import resample_factor, resample_size
from patchmatch.pm import PatchMatch
from cv2 import imshow, waitKey
import math
from patchmatch.simakov import SimakovOptim


def compute_n_downscale(src_sz, tgt_sz):
    R = 0
    while min(src_sz[0] // (2**R), src_sz[1] // (2**R)) > tgt_sz:
        R += 1
    return R, (src_sz[0] // (2**R), src_sz[1] // (2**R))


def show(x, tgt_sz, win='img'):
    #x = resample_size(x, *tgt_sz, 'nearest').astype('uint8')
    x = x.astype('uint8')
    imshow(win, x[:, :, ::-1])
    waitKey(10)


def retarget(opt, img, target_size, n_it, working_size=32):
    w, h = img.size
    target_size = target_size[1], target_size[0]
    n_reduc, working = compute_n_downscale((h, w), working_size)
    increment = (math.pow(working[0] / target_size[0], 1 / n_it),
                 math.pow(working[1] / target_size[1], 1 / n_it))

    reference = np.array(img).astype('float32')
    refine = resample_size(reference, working[0], working[1], 'bilinear')

    nn = None
    nn2 = None
    for e in range(1, n_it):
        aug = (increment[0]**e, increment[1]**e)
        red = (increment[0]**(n_it-e), increment[1]**(n_it-e))

        refine = resample_size(refine, int(target_size[0] * red[0]),
                               int(target_size[1] * red[1]), 'bilinear')

        ref = resample_size(reference, int(reference.shape[0] * red[0]),
                            int(reference.shape[1] * red[1]), 'bilinear')

        if nn2 is not None:
            nn2 = (nn2 * np.array(increment)).astype('int32')

        if nn is not None:
            nn = resample_size(nn * np.array(increment), *refine.shape[:2], 'nearest')

        refine, nn, nn2 = opt(refine, ref, nn_forward=nn, nn_backward=nn2)

    return refine


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('img')
    args.add_argument('tgt_height', type=float)
    args.add_argument('tgt_width', type=float)
    args.add_argument('--pm-thresh', type=float, default=0.99)
    args.add_argument('--em-thresh', type=float, default=0.99)
    args.add_argument('--n-it', default=10, type=int)
    args.add_argument('--em-it', default=10, type=int)
    args.add_argument('--pm-it', default=10, type=int)
    args.add_argument('--no-reuse-nn', action='store_false')
    args.add_argument('--no-bidir', action='store_false')
    args.add_argument('--working-size', default=64, type=int)
    args.add_argument('--metric', default='l1', choices=['l1', 'l2', 'cos'])
    args.add_argument('--space', default='yuv', choices=['rgb', 'yuv', 'lab'])
    args.add_argument('--padding', default='yuv', choices=['reflect', 'constant'])
    opts = args.parse_args()

    img = Image.open(opts.img).convert('RGB')

    patchmatch = PatchMatch(space=opts.space,
                            auto_thresh=opts.pm_thresh,
                            metric=opts.metric,
                            nb_max_it=opts.pm_it, pad=opts.padding)

    tgt = int(img.width * opts.tgt_width), int(img.height * opts.tgt_height)
    cb = lambda x: show(x, tgt[::-1])

    opt = SimakovOptim(patchmatch,
                       auto_thresh=opts.em_thresh,
                       nb_max_it=opts.em_it,
                       reuse_nn=opts.no_reuse_nn,
                       bidir=opts.no_bidir,
                       cb=cb)
    out = retarget(opt, img, tgt, opts.n_it, opts.working_size)
    Image.fromarray(out.astype('uint8')).save('out.png')
