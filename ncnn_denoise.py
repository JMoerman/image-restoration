#!/usr/bin/env python3

import sys
sys.path.append('./')
import os
import argparse
import cv2
from skimage import img_as_float32, img_as_ubyte
from skimage.io import imsave
from pathlib import Path
from multiprocessing import sharedctypes
import numpy as np
import time
import ncnn

def make_raw_array(nparray):
    nparray_c = np.ctypeslib.as_ctypes(nparray)
    return sharedctypes.RawArray (nparray_c._type_, nparray_c)

def reduce_noise(net, shape, tile_w, tile_h, offset, ry1, ry2, input_arr, output_arr, overlap = 12):
    im_gt = np.ctypeslib.as_array(input_arr)
    im_gt_out = np.ctypeslib.as_array(output_arr)

    C = shape[0]
    H = shape[1]
    W = shape[2]

    while H - offset < tile_h and tile_h > 128:
        tile_h = tile_h / 2
    y = min (H-tile_h,offset)
    x = 0
    first = True
    while True:
        buf = im_gt[:, y:y+tile_h, x:x+tile_w].copy ()
        mat_in = ncnn.Mat (buf)
        ex = net.create_extractor ()
        noise = ex.input ('modelInput', mat_in)
        tic = time.time()
        status, mat_out = ex.extract ('modelOutput')
        toc = time.time()
        if first:
            im_gt_out[:, y+ry1:y+tile_h-ry2, x:x+tile_w] = np.array(mat_out)[0:C, ry1:tile_h-ry2, 0:]
        else:
            im_gt_out[:, y+ry1:y+tile_h-ry2, x+overlap:x+tile_w] = np.array(mat_out)[0:C, ry1:tile_h-ry2, overlap:]
        print('{}x{} time={:.4f}'.format(x,y, toc-tic))

        if x == W - tile_w:
            break
        x = x+tile_w-2*overlap
        while W - x < tile_w and tile_w > 256:
            tile_w = tile_w // 2
        x = min (x,W-tile_w)
        first = False

def reduce_noise_full (net, img, shape, in_arr, output_arr, scale=1, tile_size=(512,512), overlap = 12):
    tic = time.time()
    y = 0
    needmore = True
    tile_h = tile_size[0]
    tile_w = tile_size[1]
    H = shape[1]
    W = shape[2]
    while W < tile_w:
        tile_w = tile_w // 2
    while tile_h > H:
        tile_h = tile_h // 2
    while needmore:
        if y == 0:
            ry1 = 0
            if H != tile_h:
                ry2 = overlap
            else:
                ry2 = 0
                needmore = False
        else:
            if y == H - tile_h:
                ry1 = overlap
                ry2 = 0
                needmore = False
            else:
                ry1 = overlap
                ry2 = overlap
        reduce_noise (net, shape, tile_w, tile_h, y, ry1, ry2, in_arr, output_arr)
        if not needmore:
            break
        y = y + tile_h - 2 * overlap
        while H - y < tile_h and tile_h > 128:
            tile_h = tile_h // 2
        y = min (y, H - tile_h)
    toc = time.time()
    print('time={:.4f}'.format(toc-tic))

    out_im = np.ctypeslib.as_array(output_arr)

    return (img - out_im)#.clip(0,1)

def read_image (path):
    m = cv2.imread(im_path, flags=(cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH))
    if m is None:
            print("cv2.imread %s failed\n" % (im_path))
            sys.exit(0)
    return img_as_float32(m[:, :, ::-1]).transpose (2,0,1).squeeze (), m.dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=Path,
                                                             help="Input file")
    parser.add_argument('outfile', type=Path,
                                                             help="Output file")
    parser.add_argument('--save_intermediates', type=Path, default=None,
                                                             help="Directory to store intermediate images")
    parser.add_argument('--extra_passes', default=[], type=float, nargs='+',
                                                             help="values to linearly scale the input with and denoise separately")
    parser.add_argument('--tile_size', default=(512,512), type=int, nargs=2,
                                                             help="Size of tiles in which the image is segemented and processed")
    parser.add_argument('--overlap', default=12, type=int, nargs=2,
                                                             help="Overlap between tiles")
    parser.add_argument('--model', default=Path('models/Denoise_VIRNet'), type=Path,
                                                             help="Model to use while denoising")
    parser.add_argument('--full_precision', action='store_true')

    args = parser.parse_args()

    im_path = str(args.infile)
    im_name = os.path.splitext(args.infile.parts[-1])[0]

    img_32, _ = read_image (im_path)
    shape = img_32.shape

    threshold = 0.8
    blur_size = 5
    blur_scale = 7

    img = img_32.copy ()
    in_arr = make_raw_array (img)
    output_arr = make_raw_array (np.full_like(img, 0))

    net = ncnn.Net()
    net.opt.use_vulkan_compute = True

    if args.full_precision:
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        net.opt.use_int8_storage = False
        net.opt.use_int8_arithmetic = False

    net.load_param("{}.param".format (str(args.model)))
    net.load_model("{}.bin".format (str(args.model)))

    final = reduce_noise_full (net, img, shape, in_arr, output_arr, 1, args.tile_size, args.overlap)

    extra_passes = args.extra_passes
    save_intermediates = args.save_intermediates

    if extra_passes and save_intermediates:
        imsave (str(save_intermediates / '{}_noise_1.tif'.format (im_name)), final)

    for s in extra_passes:
        img = np.ctypeslib.as_array(in_arr)
        img[:,] = (img_32 * s) #.clip(0,1)

        img = reduce_noise_full (net, img, shape, in_arr, output_arr, s, args.tile_size, args.overlap)
        if save_intermediates:
            imsave (str(save_intermediates / '{}_{}_noise.tif'.format (im_name,s)), img.clip (0,1))

        mask = (img[0] < threshold) & (img[1] < threshold) & (img[2] < threshold)
        mask = np.where (mask, np.zeros_like (img[0]), np.ones_like (img[0]))
        mask_blur = (cv2.GaussianBlur(mask, (blur_size, blur_size), 0) * blur_scale).clip (0,1)
        if save_intermediates:
            imsave (str(save_intermediates / '{}_{}_mask.tif'.format (im_name,s)), mask_blur)

        img = (img/s).clip (0,1)

        final[0] = (final[0] * mask_blur) + img[0] * (1 - mask_blur)
        final[1] = (final[1] * mask_blur) + img[1] * (1 - mask_blur)
        final[2] = (final[2] * mask_blur) + img[2] * (1 - mask_blur)
        if save_intermediates:
            imsave (str(save_intermediates / '{}_{}_noise_rescaled.tif'.format (im_name,s)), img)

    imsave (str(args.outfile), final)
