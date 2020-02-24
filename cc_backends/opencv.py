'''
Cross-correlation backend using OpenCV, a computer vision programming library.
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

DEBUG = False

def resize(image, factor):
    y, x = image.shape
    w = int(x * factor)
    h = int(y * factor)
    return cv2.resize(image, (w,h))



def _find_translation(orig_im, orig_im_ref, crop, max_movement=False, upscale=1):
    
    # Take copies or otherwise originals are edited yielding from results
    # Too slow like this
    #im = np.copy(orig_im)
    #im_ref = np.copy(orig_im_ref)
    
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
    

    #res = cv2.matchTemplate(im[cy:cy+ch, cx:cx+cw], im_ref, cv2.TM_SQDIFF_NORMED)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #confidence = max_val
    
    x, y = max_loc

    # back to non-upscaled coordinates
    x /= upscale
    y /= upscale

    #x, y = min_loc 
    if max_movement:
        x -= cx - rx #* upscale
        y -= cy - ry #* upscale
    else:
        x -= cx
        y -= cy


    #print('{} {}'.format((cx,cy,cw,ch),(rx,ry,rw,rh)))
    if DEBUG and confidence>0.5:

        top_left = max_loc
        bottom_right = (top_left[0]+cw, top_left[1]+ch)
        
        
        #im_ref[max_loc[1]:max_loc[1]+ch, max_loc[0]:max_loc[0]+cw] += im[cy:cy+h, cx:cx+cw] 


        cv2.rectangle(im_ref, top_left, bottom_right, 255, 1)
        
        plt.imshow(im_ref,cmap = 'gray')
        plt.show()
    
    return x, y


