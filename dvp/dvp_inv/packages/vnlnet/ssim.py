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
from skimage.measure import compare_ssim


def get_files_pattern(d, pattern):
    files = os.listdir(d)
    files = fnmatch.filter(files, pattern)
    return sorted(files)

def print_ssim(refdir, imgdir):
    reffiles = get_files_pattern(refdir, '*')
    imgfiles = get_files_pattern(imgdir, '*')
    ssim = 0.
    c = 0.
    for fref, fimg in zip(reffiles, imgfiles):
        ref = imageio.imread(refdir + '/' + fref)
        if os.path.exists(imgdir + '/' + fimg):
            img = imageio.imread(imgdir + '/' + fimg)
        elif os.path.exists(imgdir + '/' + fimg[:-3] + 'tiff'):
            img = tifffile.imread(imgdir + '/' + fimg[:-3] + 'tiff')
        else:
            continue
        img = np.squeeze(img)
        ref = np.squeeze(ref)
        ssim_img = compare_ssim(ref, img, data_range=255, multichannel=(len(img.shape)==3))
        # print('   {} SSIM {:1.4f}'.format(fref, ssim_img))
        ssim += ssim_img
        c = c + 1.

    print('Average SSIM {:1.4f}.'.format(ssim/c))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imgdir', help='Directory of denoised images (png, tiff)')
    parser.add_argument('refdir', help='Directory of reference images (png), should only contain these images')
    args = parser.parse_args()
    print_ssim(args.refdir, args.imgdir)
