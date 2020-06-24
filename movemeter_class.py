'''
Analyse translational movement from a time series of images.
Uses OpenCV's template matching (normalized cross-correlation, cv2.TM_CCOEFF_NORMED).


TODO
    Backend implementations

        IMPLEMENTATION          THOUGHTS                                    STATUS
        openCV                  Is good but a big dependency                Works
        openCL kernels          May be faster but not always supported      na        
        C                       Widely supported but has to be compiled     na
        python using numpy      Widely supported                            na
        pure python             Slow

    Automated batching to the uni cluster / other distributed computing platform?
        - time to transfer the data there may be limiting


'''

import os
import time
import multiprocessing

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
    self.cc_backend     Movement analysis backend
    self.im_backend     Image loading backend
    self.upscale        Amount to upscale during movement analysing

    '''
    
    # ---------------
    # PRIVATE METHODS

    def __init__(self, cc_backend='OpenCV', imload_backend='OpenCV', upscale=1,
            multiprocess=False, print_callback=print):
        '''
        Initializing the movemeter.
        
        The first time this can be slow if the libraries needed for the selected
        backends have not been imported before.

        INPUT ARGUMETNS         DESCRIPTION
        stacks                  List of stacks, where a stack is list of image filenames [fn1, fn2, ...]
        cc_backend              Either 'OpenCV',
        im_backend              Either 'OpenCV' or 'tifffile'. CAN also be a callable, that returns in similar format.
        multiprocess            If False, no multiprocessing. Otherwise interger to specify number of sub processes
        print_callback          If a custom print is desired instead python's print function
        '''
        
        self.upscale = upscale

        self.cc_backend = cc_backend
        self.im_backend = imload_backend

        self.multiprocess = multiprocess
        self.print_callback = print_callback

        self.stop = False

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
        This method is overwritten by any cross-correlation backend that is
        to be loaded.
        '''
        raise NotImplementedError('_findTranslation (a method in Movemeter class) needs to be overridden by the selected cc_backend implementation.')
    
   
    def _imread(self, fn):
        '''
        Abstract image read method, that loads the image using imload_backend's function
        and preprocesses the image using im_processing_func's callable.
        '''
        
        # If fn is an image already (np.array) just pass, otherwise, load
        if type(fn) == np.ndarray:
            pass
        else:
            image = self.imload(fn)
        
        # Grayscale by taking first channel if color image
        if len(image.shape) == 3:
            self.print_callback("Color image ({}), grayscaling it by dropping dimensions.".format(image.shape))
            image = image[:,:,0]

       
        # Normalize values to interval 0...1000
        image -= np.min(image)
        image = (image / np.max(image)) * 1000

        return image.astype(np.float32)
    

    def _measure_movement_optimized_xray_data(self, image_fns, ROIs,
            max_movement=False, results_list=None, worker_i=0, messages=[]):
        '''
        Optimized version when there's many rois and subtract previous is True and compare_to_first is False.
        '''

        results = []

        if worker_i == False:
            nexttime = time.time()

        # Create a mask image that is subtracted from the images to enhance moving features
        mask_image = self._imread(image_fns[0]) 
        for fn in image_fns[1:]:
            mask_image = np.min([mask_image, self._imread(fn)], axis=0)
    

        previous_image = self._imread(image_fns[0]) - mask_image

        X = [[0] for roi in ROIs]
        Y = [[0] for roi in ROIs]
 
        for i, fn in enumerate(image_fns[1:]):

            image = self._imread(fn) - mask_image
            
            for i_roi, ROI in enumerate(ROIs):
                
                if worker_i == False and nexttime < time.time():
                    percentage = int(100*(i*len(ROIs) + i_roi) / (len(ROIs)*len(image_fns)))
                    message = 'Process #1 out of {}, frame {}/{}, in ROI {}/{}. Done {}%'.format(
                            int(self.multiprocess), i+1,len(image_fns),i_roi+1,len(ROIs),int(percentage))
                    messages.append(message)
                    nexttime = time.time() + 2

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
        
        if results_list is not None:
            results_list[worker_i] = results
            return None

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
                
                #fn = image_fns[3] 
                #print(fn)

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


    # --------------------
    # MOVEMENT MEASUREMENT

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
        self.print_callback('Determining stack/ROI relationships in movemeter')
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


    
    def measure_all(self, max_movement=False, message='{}/{}'):
        '''
        Analysing translational movement in stacks and ROIs that has been set in Movemeter's setData.
        '''
        pass
        #for stack_i in range



    def measure_movement(self, stack_i, use_motion_correction=False, max_movement=False):
        '''
        Analysing translational movement in stacks and ROIs that has been set in Movemeter's setData.
        
        INPUT ARGUMETS      DESCRIPTION
        stack_i             Analyse only stack with index stack_i

        Returns
        - [results_stack1, results_stack2, ...]
            where results_stack1 = [results_ROI1_for_stack1, results_ROI2_for_stack1, ...]
            where results_ROI1_for_stack1 = [movement_points_in_X, movement_points_in_Y]

            Ordering is quaranteed to be same as when setting data in Movemeter's setData
        '''
        start_time = time.time()
        self.print_callback('Starting to analyse stack {}/{}'.format(stack_i+1, len(self.stacks)))

        if self.multiprocess:
            
            # Create multiprocessing manager and a inter-processes
            # shared results_list
            manager = multiprocessing.Manager()
            results_list = manager.list()
            messages = manager.list()
            for i in range(self.multiprocess):
                results_list.append([])
            
            # Select target _measure_movement
            if self.subtract_previous == True and self.compare_to_first == False:
                self.print_callback('Targeting to optimized version for xray data')
                target = self._measure_movement_optimized_xray_data
            else:
                target = self._measure_movement
    
            # Create and start workers
            workers = []
            work_chunk = int(len(self.ROIs[stack_i]) / self.multiprocess)
            for i_worker in range(self.multiprocess): 

                if i_worker == self.multiprocess - 1:
                    worker_ROIs = self.ROIs[stack_i][i_worker*work_chunk:]
                else:
                    worker_ROIs = self.ROIs[stack_i][i_worker*work_chunk:(i_worker+1)*work_chunk]
                
                worker = multiprocessing.Process(target=target,
                        args=[self.stacks[stack_i], worker_ROIs],
                        kwargs={'max_movement': max_movement, 'results_list': results_list,
                            'worker_i': i_worker, 'messages': messages} )
                
                workers.append(worker)
                worker.start()

            # Wait until all workers get ready
            for i_worker, worker in enumerate(workers):
                self.print_callback('Waiting worker #{} to finish'.format(i_worker+1))
                while worker.is_alive():
                    if messages:
                        self.print_callback(messages[-1])
                    time.sleep(1)
                worker.join()

            # Combine workers' results
            self.print_callback('Combining results from different workers')
            results = []
            for worker_results in results_list:
                results.extend(worker_results)

        else:
            # No multiprocessing
            if self.subtract_previous == True and self.compare_to_first == False:
                results = self._measure_movement_optimized_xray_data(self.stacks[stack_i], self.ROIs[stack_i], max_movement=max_movement)
            else:
                results = self._measure_movement(self.stacks[stack_i], self.ROIs[stack_i], max_movement=max_movement)
        

        self.print_callback('Finished stack {}/{} in {} secods'.format(stack_i+1, len(self.stacks), time.time()-start_time))

        return results 

    
    def stop():
        self._stop = True


    # ---------------------------------------------------
    # METHODS TO GET INFORMATION ABOUT THE IMAGES LOADED

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

       


