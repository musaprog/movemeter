'''
Contains the Movemeter class for analysing translational movement from a time series
of images.

Under the hood, it uses OpenCV's template matching (normalized cross-correlation,
cv2.TM_CCOEFF_NORMED). Also other backends are supported (but currently, not implemeted).
'''

import os

import exifread
import numpy as np
import cv2


class Movemeter:
    '''
    Analyses translational movement from time series of images.

    Usage:
        1) set_data: Set images and ROIs (regions of interest)
        2) measure_movement: Returns the movement data

    ------------
    Attributers
    ------------
    self.cc_backend     Movement analysis backend.
    self.im_backend     Image loading backend
    self.upscale        Amount to upscale during movement analysing
    
    self.subtract_previous
    self.tracking_rois
    self.compare_to_first
    
    --------
    METHODS
    --------
    self.set_data
    self.measure_movement
    self.get_metadata
    self.get_image_resolution
 
    ----------------
    PRIVATE METHODS
    ----------------
    self._imread
    self._find_translation

   
    '''
    

    def __init__(self, cc_backend='OpenCV', imload_backend='OpenCV', upscale=1):
        '''
        Initialize the movemeter.
        The backends and the upscale factor can be specified.         

        INPUT ARGUMETNS         DESCRIPTION
        cc_backend              Backend to calculate the "cross-correlation".
                                    Currently only 'OpenCV' is supported
        im_backend              Backend to open the images.             
                                    Currently 'OpenCV' and 'tifffile' are supported.
                                    Can also be a callable, that takes in a filenam and
                                        returns a 2D numpy array.
        
        
        '''
        
        self.upscale = upscale

        self.cc_backend = cc_backend
        self.im_backend = imload_backend

        # IMAGE LOADING BACKEND
        if imload_backend == 'OpenCV':
            import cv2
            self.imload = lambda fn: cv2.imread(fn, -1)
        
        elif imload_backend == 'tifffile':
            import tifffile
            self.imload = lambda fn: tifffile.imread(fn)

        elif callable(imload_backend):
            self.imload = imload_backend 

        else:
            raise ValueError('Given backend {} is not "OpenCV" or "tifffile" or a callable'.format(print(imload_backend)))
        
        
        # CROSS CORRELATION BACKEND
        
        if cc_backend == 'OpenCV':
            from .cc_backends.opencv import _find_translation
            self._findTranslation = _find_translation
       
        # Movement measure settings
        self.subtract_previous = False
        self.tracking_rois = False
        self.compare_to_first = True



    @staticmethod
    def _find_translation(im, im_ref, crop):
        '''
        This method is overwritten by any cross-correlation backend that is loaded.
        '''
        raise NotImplementedError('_findTranslation (a method in Movemeter class) needs to be overridden by the selected cc_backend implementation.')
    


    def _imread(self, fn):
        '''
        Wrapper for self.imload (that depends on the image load backed).

        Verifies the dimensionality of the loaded data and normalizes 
        the image to float32 range.
        '''
        
        # If fn is an image already (np.array) just pass, otherwise, load
        if type(fn) == np.ndarray:
            pass
        else:
            image = self.imload(fn)
        
        # Grayscale by taking first channel if color image
        if len(image.shape) == 3:
            print("Color image ({}), grayscaling it by dropping dimensions.".format(image.shape))
            image = image[:,:,0]

       
        # Normalize values to interval 0...1000
        # FIXME Is the range 0...1000 optimal?
        image -= np.min(image)
        image = (image / np.max(image)) * 1000

        # FIXME Add option for image data type / not to enforce data type
        return image.astype(np.float32)
    


    def _measure_movement_optimized_manyrois(self, image_fns, ROIs, max_movement=False):
        '''
        Optimized version when there's many rois and subtract previous
        is True and compare_to_first is False.
        '''
        print('Running optimized version for xray data')

        results = []

        # Create a mask image that is subtracted from the images to enhance moving features
        mask_image = self._imread(image_fns[0]) 
        for fn in image_fns[1:]:
            mask_image = np.min([mask_image, self._imread(fn)], axis=0)
    

        previous_image = self._imread(image_fns[0]) - mask_image

        X = [[0] for roi in ROIs]
        Y = [[0] for roi in ROIs]
 
        for i, fn in enumerate(image_fns[1:]):

            print('Frame {}/{}'.format(i+1, len(image_fns)))
           
            image = self._imread(fn) - mask_image
            
            for i_roi, ROI in enumerate(ROIs):
                print('  _ROI: {}/{}'.format(i+1, len(ROIs)))
                
                x, y = self._findTranslation(image, previous_image, ROI,
                        max_movement=int(max_movement), upscale=self.upscale)
                    
                X[i_roi].append(x)
                Y[i_roi].append(y)

            previous_image = image
        
        for x,y in zip(X,Y):
        
            x = np.asarray(x)
            y = np.asarray(y)

            x = x-x[0]
            y = y-y[0]

            x = np.cumsum(x)
            y = np.cumsum(y)

            results.append([x.tolist(), y.tolist()])
    
        return results


    def _measure_movement(self, image_fns, ROIs, max_movement=False):
        '''
        Generic way to analyse movement using _findTranslation.
        
        Could be overridden by a cc_backend.
        '''
        
        results = []
     
        if self.subtract_previous:
            mask_image = self._imread(image_fns[0])
            
            for fn in image_fns[1:]:
                mask_image = np.min([mask_image, self._imread(fn)], axis=0)
        
        for i, ROI in enumerate(ROIs):
            print('  _measureMovement: {}/{}'.format(i+1, len(ROIs)))
            if self.compare_to_first:
                previous_image = self._imread(image_fns[0])

            X = [0]
            Y = [0]

            for i, fn in enumerate(image_fns[1:]):
                print('ROI IS {}'.format(ROI))
                print('Frame {}/{}'.format(i+1, len(image_fns)))
                
                if self.compare_to_first == False:
                    print('not comparing to first')
                    if self.subtract_previous:
                        previous_image = self._imread(image_fns[i]) -  mask_image
                    else:
                        previous_image = self._imread(image_fns[i])
                
                if self.subtract_previous:
                    print('subtracting previous')
                    image = self._imread(fn) - mask_image
                else:
                    image = self._imread(fn)
                
                x, y = self._findTranslation(image, previous_image, [int(c) for c in ROI],
                        max_movement=int(max_movement), upscale=self.upscale)
                    
                X.append(x)
                Y.append(y)

                if self.tracking_rois:
                    print('roi tracking')
                    #ROI = [ROI[0]+x, ROI[1]+y, ROI[2], ROI[3]]
                
            X = np.asarray(X)
            Y = np.asarray(Y)

            X = X-X[0]
            Y = Y-Y[0]

            if not self.compare_to_first:
                X = np.cumsum(X)
                Y = np.cumsum(Y)

            results.append([X.tolist(), Y.tolist()])
        
        return results



    def set_data(self, stacks, ROIs):
        '''
        Setting image filenames and recpective ROIs (regions of interest).

        INPUT ARGUMENTS
        stacks          List of filename lists: [ [stack1_im1, stack1_im2...],[stack2_im1, stack2_im2], ...]
        ROIs            [[ROI1_for_stack1, ROI2_for_stack1, ...], [ROI1_for_stack2, ...],...].
                        ROIs's length is 1 means same ROIs for all stacks

                        ROI format: (x, y, w, h)
        '''
        
        self.stacks = stacks
        # DETERMINE
        print('Determining stack/ROI relationships in movemeter')
        if len(ROIs) > 1:
            # Separate ROIs for each stack
            self.ROIs = ROIs
        
        if len(ROIs) == 1:
            # Same ROIs for all the stacks
            
            self.ROIs = [ROIs[0] for i in range(len(stacks))]
            
        elif len(ROIs) != len(stacks):
            raise ValueError("Movemeter.setData: stacks ({}) and ROIs ({}) has to have same length OR ROIs's length has to be 1".format(len(stacks), len(ROIs)))
        
        # ensure ROIs to ints
        self.ROIs = [[[int(x), int(y), int(w), int(h)] for x,y,w,h in ROI] for ROI in self.ROIs]



    def measure_movement(self, stack_i, max_movement=False):
        '''
        Analysing translational movement in stacks and ROIs (set with set_data method)
        
        INPUT ARGUMETS      DESCRIPTION
        stack_i             Analyse only stack with index stack_i
        max_movement        Speed up the computation by specifying the maximum translation
                                between subsequent frames, in pixels.

        Returns
            results_stack_i = [results_ROI1_for_stack_i, results_ROI2_for_stack_i, ...]
            where results_ROIj_for_stack_i = [movement_points_in_X, movement_points_in_Y]

            Ordering is quaranteed to be same as when setting data in Movemeter's setData
        '''
        
        if self.subtract_previous == True and self.compare_to_first == False:
            results = self._measure_movement_optimized_manyrois(self.stacks[stack_i], self.ROIs[stack_i], max_movement=max_movement)
        else:
            results = self._measure_movement(self.stacks[stack_i], self.ROIs[stack_i], max_movement=max_movement)
        
        return results



    def get_metadata(self, stack_i, image_i=0):
        '''
        Uses exifread to get the metadata for stack number stack_i.

        Returns a dictionary of exifread objects.
        '''

        with open(self.stacks[stack_i][image_i], 'rb') as fp:
            tags = exifread.process_file(fp)

        return tags
    

    def get_image_resolution(self, stack_i):
        '''
        Returns resolution of the images in stack_i.

        TODO: - Currently opens the first image to see the resolution (slow).
                Would be better to read from the metadata directly
        '''
        height, width = self._imread(self.stacks[stack_i][0]).shape
        return width, height

       


