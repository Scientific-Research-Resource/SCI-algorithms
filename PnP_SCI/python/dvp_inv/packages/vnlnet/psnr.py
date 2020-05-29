"""
Copyright (C) 2018  Axel Davy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import os
import os.path
import fnmatch
import argparse
import imageio
import tifffile
from skimage.measure.simple_metrics import compare_psnr

def get_files_pattern(d, pattern):
    files = os.listdir(d)
    files = fnmatch.filter(files, pattern)
    return sorted(files)

def print_psnr(refdir, imgdir):
    reffiles = get_files_pattern(refdir, '*')
    imgfiles = get_files_pattern(imgdir, '*')
    psnr = 0.
    l = 0.
    acc = np.zeros([1], np.float64)
    acc2 = np.zeros([1], np.float64)
    for fref, fimg in zip(reffiles, imgfiles):
        ref = imageio.imread(refdir + '/' + fref)
        if os.path.exists(imgdir + '/' + fimg):
            img = imageio.imread(imgdir + '/' + fimg)
        elif os.path.exists(imgdir + '/' + fimg[:-3] + 'tiff'):
            img = tifffile.imread(imgdir + '/' + fimg[:-3] + 'tiff')
        else:
            continue
        psnr_img = compare_psnr(ref, img, data_range=255)
        # print('  {} PSNR {:2.2f} dB'.format(fref, psnr_img))
        psnr += psnr_img
        l = l + 1.

        acc += np.sum(np.square(ref - img))
        acc2 += np.sum((ref >= 1 )*1.)
    print('Average PSNR {:2.2f} dB.'.format(psnr/l))
    # The PSNR below is the correct one for video
    # print('PSNR on the video sequence: {:2.2f} dB.'.format(10 * np.log10((255. ** 2) / (acc[0]/acc2[0]))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imgdir', help='Directory of denoised images (png, tiff)')
    parser.add_argument('refdir', help='Directory of reference images (png, tiff), should only contain these images')
    args = parser.parse_args()
    print_psnr(args.refdir, args.imgdir)
