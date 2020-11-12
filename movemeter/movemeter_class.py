'''
Contains the Movemeter class for analysing translational movement from a time series
of images.

Under the hood, it uses OpenCV's template matching (normalized cross-correlation,
cv2.TM_CCOEFF_NORMED). Also other backends are supported (but currently, not implemeted).
'''

import os
import time
import multiprocessing

import exifread
import numpy as np
import cv2


class Movemeter:
    '''Analysing translational movement from time series of images.

    Common work flow:
        1) Create a Movemeter object by
                meter = Movemeter()

        2) set_data: Set images and ROIs (regions of interest)
                meter.set_data(image_stacks, ROIs)

        3) measure_movement: Returns the movement data
                meter.measure_movement()


    Attributes
    ----------
    upscale : int
        Amount to upscale during movement analysing, passed to the cc backend
    cc_backend : string
        Movement analysis backend. Currently only "OpenCV"
    im_backend : string
        Image loading backend. "OpenCV" or "tifffile",
        or a callable that takes in an image filename and returns a 2D numpy array.

    absolute_results : bool
        Return results in absolute image coordinates
    tracking_rois : bool
        If True, ROIs are shifted between frames, following the movement
    compare_to_first : bool
        If True, use the ROI of the first image only to quantify the movement.
        Good for purely translational motion when frame-to-frame displacements are
        very small.
    subtract_previous : bool
        Special treatment for when there's a faint moving feature on a static
        background.
    multiprocess : int
        If 0 then no multiprocessing. Otherwise the number of parallel processes.
        Note that there may be some multiprocessing already at the cc_backend level.
    print_callback : callable
        Print function to convey the progress.
        By default, this is the built-in print function.

    '''    

    def __init__(self, upscale=1, cc_backend='OpenCV', imload_backend='OpenCV',
            absolute_results=False, tracking_rois=False, compare_to_first=True,
            subtract_previous=False, multiprocess=False, print_callback=print):
        
        # See Class docstring for documentation

        # Set the options given in constructor to same name attributes
        self.upscale = upscale
        self.cc_backend = cc_backend
        self.im_backend = imload_backend
        self.absolute_results = absolute_results
        self.tracking_rois = tracking_rois
        self.compare_to_first = compare_to_first
        self.subtract_previous = subtract_previous
        self.multiprocess = multiprocess
        self.print_callback = print_callback


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
            from .cc_backends.opencv import _find_location
            self._find_location = _find_location
       


    @staticmethod
    def _find_location(im, ROI, im_ref):
        '''
        This method is to be overwritten by any cross-correlation backend that is loaded.
        
        Parameters
        ----------
        im
            Image
        ROI : tuple of int
            (x,y,w,f)
        im_ref
            Reference image
        '''
        raise NotImplementedError('_find_location (a method in Movemeter class) needs to be overridden by the selected cc_backend implementation.')
    


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
            self.print_callback("Color image ({}), grayscaling it by dropping dimensions.".format(image.shape))
            image = image[:,:,0]

       
        # Normalize values to interval 0...1000
        # FIXME Is the range 0...1000 optimal?
        image -= np.min(image)
        image = (image / np.max(image)) * 1000


        return image.astype(np.float32)
    

    def _measure_movement_optimized_xray_data(self, image_fns, ROIs,
            max_movement=False, results_list=None, worker_i=0, messages=[]):
        '''
        Optimized version when there's many rois and subtract previous
        is True and compare_to_first is False.
        '''

        results = []

        if worker_i == False:
            nexttime = time.time()

        # Create a mask image that is subtracted from the images to enhance moving features
        mask_image = self._imread(image_fns[0]) 
        for fn in image_fns[1:]:
            mask_image = np.min([mask_image, self._imread(fn)], axis=0)
    

        previous_image = self._imread(image_fns[0]) - mask_image

        X = [[] for roi in ROIs]
        Y = [[] for roi in ROIs]
 
        for i, fn in enumerate(image_fns[0:]):

            image = self._imread(fn) - mask_image
            
            for i_roi, ROI in enumerate(ROIs):
                
                if worker_i == False and nexttime < time.time():
                    percentage = int(100*(i*len(ROIs) + i_roi) / (len(ROIs)*len(image_fns)))
                    message = 'Process #1 out of {}, frame {}/{}, in ROI {}/{}. Done {}%'.format(
                            int(self.multiprocess), i+1,len(image_fns),i_roi+1,len(ROIs),int(percentage))
                    messages.append(message)
                    nexttime = time.time() + 2

                x, y = self._find_location(image, ROI, previous_image,
                        max_movement=int(max_movement), upscale=self.upscale)
                    
                X[i_roi].append(x)
                Y[i_roi].append(y)

            previous_image = image
        
        for x,y in zip(X,Y):
        
            x = np.asarray(x)
            y = np.asarray(y)
            
            if not self.absolute_results:
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
        Generic way to analyse movement using _find_location.
        
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

            X = []
            Y = []

            for i, fn in enumerate(image_fns[0:]):
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
                
                x, y = self._find_location(image, [int(c) for c in ROI], previous_image, 
                        max_movement=int(max_movement), upscale=self.upscale)
                    
                X.append(x)
                Y.append(y)

                if self.tracking_rois:
                    print('roi tracking')
                    #ROI = [ROI[0]+x, ROI[1]+y, ROI[2], ROI[3]]
                
                print('{} {}'.format(x,y))

            X = np.asarray(X)
            Y = np.asarray(Y)

            if not self.absolute_results:
                X = X-X[0]
                Y = Y-Y[0]

                if not self.compare_to_first:
                    X = np.cumsum(X)
                    Y = np.cumsum(Y)

            results.append([X.tolist(), Y.tolist()])
        
        return results



    def set_data(self, stacks, ROIs):
        ''' Set image filenames and regions to be analysed.

        Parameters
        ----------
        stacks : list
            List of filename lists: [ [stack1_im1, stack1_im2...],[stack2_im1, stack2_im2], ...]
        ROIs : list
            [[ROI1_for_stack1, ROI2_for_stack1, ...], [ROI1_for_stack2, ...],...].
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



    def measure_movement(self, stack_i, max_movement=False):
        ''' Run the translational movement analysis.

        Image stacks and ROIs are expected to be set before using set_data method.
        See class attributes.

        Note
        ----
            Ordering is quaranteed to be same as when setting data in Movemeter's setData
        
        Parameters
        ----------
        stack_i : int
            Analyse stack with index stack_i (order according what set to set_data method)
        max_movement : int
            Speed up the computation by specifying the maximum translation between
            subsequent frames, in pixels.

        Returns
        -------
        results_stack_i
            [results_ROI1_for_stack_i, results_ROI2_for_stack_i, ...]
            where results_ROIj_for_stack_i = [movement_points_in_X, movement_points_in_Y]

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

    

    def get_metadata(self, stack_i, image_i=0):
        '''Get metadata for stack number stack_i using exifread.
        
        Parameters
        ----------
        stack_i : int
            Index of the stack (set_data)
        image_i : int
            Index of the image

        Returns
        -------
        tags: dict
            A dictionary of exifread objects. See exifread documentation.
        '''

        with open(self.stacks[stack_i][image_i], 'rb') as fp:
            tags = exifread.process_file(fp)

        return tags
    

    def get_image_resolution(self, stack_i):
        '''
        Returns resolution of the images in stack_i.

        Note
        ----
            Currently opens the first image to see the resolution (slow).
            Would be better to read from the metadata directly
        
        Parameters
        ----------
        stack_i : int
            Index of the stack (set_data)

        Returns
        -------
        width : int
        height : int

        '''
        height, width = self._imread(self.stacks[stack_i][0]).shape
        return width, height

       


