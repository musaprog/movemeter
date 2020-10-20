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


def _find_translation(orig_im, orig_im_ref, crop, max_movement=False, upscale=1):
    
    cx,cy,cw,ch = crop
    if max_movement != False:
        
        rx,ry,rw,rh = (cx-max_movement, cy-max_movement, cw+2*max_movement, ch+2*max_movement)
        if rx < 0:
            rx = 0
        if ry < 0:
            ry = 0
        ih, iw = orig_im_ref.shape
        if rx+rw>iw:
            rw -= rx+rw-iw
        if ry+rh>ih:
            rh -= rh+rh-ih
        im_ref = np.copy(orig_im_ref[ry:ry+rh, rx:rx+rw])
    else:
        im_ref = orig_im_ref
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
        x -= cx - rx #* upscale
        y -= cy - ry #* upscale
    else:
        x -= cx
        y -= cy
   
    return x, y


