'''Stabilizing a video or a image stack using Movemeter.
'''

import argparse

import tifffile
import numpy as np

from .movemeter import Movemeter
from .stacks import stackread, stackwrite
from .cc_backends.opencv import resize, rotate


import matplotlib.pyplot as plt

def _create_movemeter(input_fn, roi, upscaling=None, max_movement=10):
    if upscaling is None:
        upscale = 1
    else:
        upscale = upscaling

    # A) Measure movement
    movemeter = Movemeter(upscale=upscale, template_method='mean', max_movement=max_movement)
    movemeter.set_data([[input_fn]], [[[0,0,0,0]]])

    width, height = movemeter.get_image_resolution(0)
    
    ms = 1/5   # margin scaler

    if not roi:
        margin_x = int(width * ms)
        margin_y = int(height * ms)
        roi = [margin_x, margin_y, width-2*margin_x, height-2*margin_y]
    
    movemeter.set_data([[input_fn]], [[roi]])
    
    return movemeter


def _get_2meters(input_fn, roi, upscaling, mediary_fn):
    '''Returns a measurer and reader movemeters.
    '''
    measurer = _create_movemeter(input_fn, roi, upscaling)
    if mediary_fn is None:
        reader = _create_movemeter(input_fn, roi, upscaling)
    else:
        reader = _create_movemeter(mediary_fn, roi, upscaling)
    return measurer, reader




def _stabilize_xy(movemeter, reader_movemeter, final_upscaling):
    '''Generator that yields XY-stabilized frames.
    '''
    if final_upscaling is None:
        final_upscaling = 1
    else:
        final_upscaling = int(final_upscaling)
 
    upscale = movemeter.upscale
    #width, height = movemeter.get_image_resolution(0)
    
    results = movemeter.measure_movement(0) 
    X, Y = results[0]
    
    # B) Write new, motion stabilized stack
    for i, image in enumerate(stackread(reader_movemeter.stacks[0][0])):
        y = - int(round(Y[i]*upscale))
        x = - int(round(X[i]*upscale))

        image = resize(image, upscale)

        width = image.shape[1]
        height = image.shape[0]
        for ax, dim in zip([y, x], [width, height]):
            if ax > 0:
                image = np.concatenate( (image[ax:], np.zeros((dim, ax)).T) , axis=0)
            elif ax < 0:
                image = np.concatenate( (np.zeros((dim, abs(ax))).T, image[:-abs(ax)]) , axis=0 )
            image = image.T
        
        image = resize(image, final_upscaling/upscale)
        yield image



def stabilize_xy(input_fn, roi, output_fn, mediary_fn=None, upscaling=None, final_upscaling=None):
    '''Writes a stabilized video file on disk.

    input_fn : string
        Input file name
    roi : tuple
        (x,y,w,h)
    output_fn : string
        Output file name
    '''
    measurer, reader = _get_2meters(input_fn, roi, upscaling, mediary_fn)#

    with stackwrite(output_fn) as writer:
        for image in _stabilize_xy(measurer, reader, final_upscaling):
            writer.write(image.astype(np.uint16))


def _stabilize_rot(movemeter, reader_movemeter, final_upscaling):
    '''Generator that yields XY-stabilized frames.
    '''
    if args.final_upscaling:
        args.final_upscaling = int(args.final_upscaling)
    else:
        args.final_upscaling = 1 
 
    upscale = movemeter.upscale
    rotations = movemeter.measure_rotation(0)[0]
    for i, image in enumerate(stackread(movemeter.stacks[0][0])):
        
        image = resize(image, upscale)

        image = rotate(image, -rotations[i])

        image = resize(image, final_upscaling/upscale)
        yield image


def stabilize_rot(input_fn, roi, output_fn, mediary_fn=None, upscaling=None, final_upscaling=None):
    '''Writers a stabilized video file on disk.
    '''
    measurer, reader = _get_2meters(input_fn, roi, upscaling, mediary_fn)
    
    with stackwrite(output_fn) as writer:
        for image in _stabilize_rot(measurer, reader, final_upscaling):
            writer.write(image.astype(np.uint16))


def stabilize(input_fn, roi, output_fn, mediary_fn=None, upscaling=None, final_upscaling=None,
              movemeter_settings=None):
    '''Runs stabilization on two passes: First on rotation, then on XY.

    Arguments
    ---------
    input_fn : string
    roi : tuple
    output_fn : string
    mediary_fn : string or tuple
    '''
    tmpname = 'movemeter_stabilize_tmp.tif'
    stabilize_rot(input_fn, roi, tmpname, mediary_fn, upscaling, final_upscaling)
    stabilize_xy(tmpname, roi, output_fn, mediary_fn, upscaling/final_upscaling, final_upscaling)
    

def main():
    parser = argparse.ArgumentParser(description='Stabilize a video')
    parser.add_argument('input', help='Input file for motion analysis')
    parser.add_argument('output', help='Output file')

    parser.add_argument('-m', '--mediary', help='For output, use this instead of the input file')

    parser.add_argument('-r', '--roi', help='Region of interest (x,y,w,h)')
    parser.add_argument('-u', '--upscaling', help='Upscale during template matching')
    parser.add_argument('-U', '--final-upscaling', help='Output upscaling during transformations')

    parser.add_argument('--max-movement', help='In original image pixels, maximum movement of the template image')
    parser.add_argument('--max-rotation', help='In original image pixels, maximum rotation of the template image')
    
    args = parser.parse_args()

   
    if args.upscaling:
        args.upscaling = int(args.upscaling)
    else:
        args.upscaling = 1
   
    movemeter_settings = {
            'max_movement': args.max_movement,
            'max_rotation': args.max_rotation,
            }
 
    stabilize(
            args.input,
            args.roi,
            args.output,
            args.mediary,
            args.upscaling,
            args.final_upscaling,
            movemeter_settings)



if __name__ == "__main__":
    main()
