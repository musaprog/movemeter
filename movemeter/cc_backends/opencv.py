'''
Cross-correlation backend using OpenCV, a computer vision programming library.
'''

import numpy as np
import cv2
import time


def resize(image, factor):
    y, x = image.shape
    w = int(x * factor)
    h = int(y * factor)
    return cv2.resize(image, (w,h))


def _find_location(orig_im, ROI, orig_im_ref, max_movement=None, upscale=1):
    '''
    Returns the location of orig_im, cropped with crop (x,y,w,h), in the location
    of orig_im_ref coordinates.

    orig_im         Image
    ROI            Crop of the orig_im
    orig_im_ref     Reference image
    '''
    cx,cy,cw,ch = ROI

    if max_movement:

        # rx,ry,rw,rh ; Coordinates for original image cropping
        rx,ry,rw,rh = (cx-max_movement, cy-max_movement, cw+2*max_movement, ch+2*max_movement)
        
        # Make sure not cropping too much top or left
        if rx < 0:
            rx = 0
        if ry < 0:
            ry = 0

        # Make sure not cropping too much right or bottom
        ih, iw = orig_im_ref.shape
        if rx+rw>iw:
            rw -= rx+rw-iw
        if ry+rh>ih:
            rh -= rh+rh-ih

        # Crop
        im_ref = np.copy(orig_im_ref[ry:ry+rh, rx:rx+rw])

    else:
        # No max movement specified; Template
        # matching the whole reference image.
        im_ref = orig_im_ref

    # Select the ROI part of the template matched image
    im = np.copy(orig_im[cy:cy+ch, cx:cx+cw])
    
    im_ref /= (np.max(im_ref)/1000)
    im /= (np.max(im)/1000)
    
    if upscale != 1:
        im = resize(im, upscale)
        im_ref = resize(im_ref, upscale)
    
    res = cv2.matchTemplate(im, im_ref, cv2.TM_CCOEFF_NORMED)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    x, y = max_loc

    # back to non-upscaled coordinates
    x /= upscale
    y /= upscale

    if max_movement:
        # Correct the fact if used cropped reference image
        x += rx
        y += ry

    return x, y


