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


def rotate(image, degrees):
    '''Rotates an image along its center.

    Returned image has the same shape as the original,
    '''
    shape = image.shape
    matrix = cv2.getRotationMatrix2D(
            [(shape[1]-1)/2, (shape[0]-1)/2],
            degrees, 1)

    return cv2.warpAffine(image, matrix, (shape[1], shape[0]))


def find_location(orig_im, ROI, orig_im_ref, max_movement=None, upscale=1):
    '''Runs translational motion analysis two images.

    Returns the location of orig_im, cropped with crop (x,y,w,h),
    in the location of orig_im_ref coordinates.

    Arguments
    ---------
    orig_im : numpy array
        The N+1 image
    ROI : sequence of ints, sequence of floats
        Crop of the orig_im (x,y,w,h)
    orig_im_ref : numpy array
        Reference or the template N image
    max_movement : int
        Maximum allowed displacement for increased
        reliability and speed
    upscale : float
        Upscaling of images for better resolution

    Returns
    -------
    x, y : float
        Cross-correlated X and Y translations between
        the images.
    '''
    cx,cy,cw,ch = ROI
    cx = int(round(cx))
    cy = int(round(cy))
    cw = int(round(cw))
    ch = int(round(ch))

    if max_movement:
        max_movement = int(max_movement)

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



def _similarity(image1, image2, max_movement=None):
    '''Uses OpenCVs matchTemplate to give a score.
    
    image1, image2 : np.arrays
        The compared images.
    max_movement : int
        In pixels, what is the maximum distance to look for the match.
        If 0, only look to match image1 center.
    '''
    if max_movement:
        w1, h1 = image1.shape
        w2, h2 = image2.shape

        if w1 > w2 or h1 > w2:
            # smaller
            mx = int(max(0, w1-w2)/2)
            my = int(max(0, h1-h2)/2)


            image1 = np.copy(image1)[my:h1-my, mx:h1-mx]

    values = cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED)
    

    min_val, max_val, min_loc, max_lov = cv2.minMaxLoc(values)
    return np.max(values)



def find_rotation(orig_im, ROI, orig_im_ref, upscale=1,
                   max_movement=None, max_rotation=None,
                   round1_steps=36, extra_rounds=5, extra_round_steps=6):
    '''Rotates template along its center and checks the best rotation.
 
    The ROI is internally change to the largest *square* ROI that fits
    withing the given (rectangular) ROI.

    Arguments
    ---------
    orig_im : numpy array
        The N+1 image
    ROI : sequence of ints, sequence of floats
        Crop of the orig_im (x,y,w,h)
    orig_im_ref : numpy array
        Reference or the template N image
    upscale : float
        Upscaling of images for better resolution
    max_movement : int
        Maximum allowed displacement for increased
        reliability and speed
    max_rotation : float or int
        In degrees, how many degrees is the total range of rotation
        to be checked. Between 0 and 360 degrees.
    round1_steps : int
        How many steps to take on the round 1.
    extra_rounds : int
        How many extra rounds to take in the vicinity of the first
        round result.
    extra_round_steps : int
        How many rotations to test on the extra rounds.
    '''
    scores = []


    if max_rotation is None:
        a = -179.5
        b = 179.5
    else:
        a = -max_rotation / 2.
        b =  max_rotation / 2.
    
    cx,cy,cw,ch = ROI
    cx = int(round(cx))
    cy = int(round(cy))
    cw = int(round(cw))
    ch = int(round(ch))

    # Squarify to the largest square that fits in the ROI
    if cw != ch:
        if cw > ch:
            diff = cw-ch
            cw = ch
            cx += int(diff/2)
        else:
            diff = ch-cw
            ch = cw
            cy += int(diff/2)


    template = orig_im_ref[cy:cy+ch, cx:cx+cw]
    if upscale != 1:
        template = resize(np.copy(template), upscale)
        original = resize(np.copy(orig_im[cy:cy+ch, cx:cx+cw]), upscale)
    else:
        original = orig_im[cy:cy+ch, cx:cx+cw]

    # Round 1 - Do all rotations

    rotations = np.linspace(a, b, round1_steps)

    for i_round in range(1+extra_rounds):
        scores = []
        
        if i_round != 0:
            rotations = np.linspace(rotations[a], rotations[b], extra_round_steps+2)[1:-1]
        for rotation in rotations:

            shape = template.shape
            rotated_template = rotate(template, rotation)
            
            # Margins
            shouldbe = int(min(shape)/np.sqrt(2))
            my = int((shape[0]-shouldbe) / 2.)
            mx = int((shape[1]-shouldbe) / 2.)
            rotated_template = rotated_template[my:shape[0]-my, mx:shape[1]-mx]

            
            score = _similarity(original, rotated_template, max_movement)
            scores.append(score)
        
        i_best = np.argmax(scores)
        #print(f'Round {i_round+1}: {rotations[i_best]} is the best rotation bewteen {rotations[0]} and {rotations[-1]}.')
        a = i_best-1
        if a < 0:
            a = 0
        b = i_best+1
        if b > len(rotations) -1:
            b = len(rotations)-1
        if a == b:
            break
    
    return rotations[i_best]

