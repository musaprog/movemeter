'''
Contains the Movemeter class for analysing translational movement from a time series
of images.

Under the hood, it uses OpenCV's template matching (normalized cross-correlation,
cv2.TM_CCOEFF_NORMED). Also other backends are supported (but currently, not implemeted).
'''

import os
import time
import multiprocessing
import warnings
import types

import exifread
import numpy as np
import cv2

try:
    # Optional dependency to scipy, used only for preblurring
    # (Movemeter.preblur)
    import scipy.ndimage
except ImportError:
    scipy = None
    warnings.warn('cannot import scipy.ndimage; Movemeter preblur not available')


from movemeter.stacks import stackread

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

        Note:
        tifffile supports multipage images (many frames/images in a single file) and
        OpenCV doesn't.

    absolute_results : bool
        Return results in absolute image coordinates
    tracking_rois : bool
        If True, ROIs are shifted between frames, following the movement
    template_method : string
        How to create the template image used in the motion analysis that
        is based on template matching.

        If string:
        'first': Use the first image (default)
                + Good even when frame-to-frame displacement are small
        'previous': Use the frame before the current frame.
                + Good when the object changes over time
                - Bad when frame-to-frame displacements are small
        'mean': Calculate the mean frame over the whole stack
                + Good when frames are very noise
                - Bad when movement too large

        New since movemeter v0.6.0.
            The prior versions' attribute called `compare_to_first` corresponded
            to template_method='first' when set to True and to
            template_method='previous' when set to False.

    subtract_previous : bool
        Special treatment for when there's a faint moving feature on a static
        background.
    multiprocess : int
        If 0 then no multiprocessing. Otherwise the number of parallel processes.
        Note that there may be some multiprocessing already at the cc_backend level.
        If used, avoid adding any non-pickable or heavy attributes to this class objects.
    print_callback : callable
        Print function to convey the progress.
        By default, this is the built-in print function.
    preblur : False or float
        Standard deviation of the Gaussian blur kernel, see scipy.ndimage.gaussian_filter
        Requires an optional dependency to scipy.
    max_movement : None or int
        If not None, in pixels, the maximum expected per-frame (compare_to_first=False)
        or total (compare_to_first=True) motion analysis result in pixel.
        This cropping the source image can increase performance a lot but too small
        values will truncated and underestimated results.
    ''' 

    measure_brightness_opt = {
            'relative': ['absolute', 'roi', 'roimin'],
            }

    def __init__(self, upscale=1, cc_backend='OpenCV', imload_backend='tifffile',
            absolute_results=False, tracking_rois=False, template_method='first',
            subtract_previous=False, multiprocess=False, print_callback=print,
            preblur=0, max_movement=None, max_rotation=None):
        '''
        See Class docstring for documentation
        '''

        # Set the options given in constructor to same name attributes
        self.upscale = upscale
        self.cc_backend = cc_backend
        self.im_backend = imload_backend
        self.absolute_results = absolute_results
        self.tracking_rois = tracking_rois
        #self.compare_to_first = compare_to_first           # removed in v0.6.0
        self.template_method = template_method
        self.subtract_previous = subtract_previous
        self.multiprocess = multiprocess
        self.print_callback = print_callback
        self.preblur=preblur
        self.max_movement = max_movement
        self.max_rotation = max_rotation

        # IMAGE LOADING BACKEND
        self.imload_args = []
        if imload_backend == 'OpenCV':
            import cv2
            self.imload = cv2.imread
            self.imload_args.append(-1)

        elif imload_backend == 'tifffile':
            import tifffile
            self.imload = tifffile.imread

        elif callable(imload_backend):
            self.imload = imload_backend 

        else:
            raise ValueError('Given backend {} is not "OpenCV" or "tifffile" or a callable'.format(print(imload_backend)))
        
        
        # CROSS CORRELATION BACKEND        
        if cc_backend == 'OpenCV':
            from .cc_backends.opencv import _find_location, _find_rotation
            self._find_location = _find_location
            self._find_rotation = _find_rotation
       


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

        
        Returns dim=3 numpy array where axis=0 is the number of pages (images)
        in the tiff file.
        '''

        # Check the appropriate action

        if isinstance(fn, str):
            return stackread(fn)
        elif isinstance(fn, np.ndarray):
            return fn
        elif hasattr(fn, '__iter__') or hasattr(fn, '__getitem__'):
            return fn
        else:
            print(f'Warning! Unkown fn: {fn}')
            return fn
        # Check if the image file is actually a stack of many images.
        #if len(image.shape) == 3:
        #    pass
        #else:
        #    image = [image]
       
        # Normalize values to interval 0...1000
        # FIXME Is the range 0...1000 optimal?

        #for i in range(len(image)):
        #    image[i] -= np.min(image[i])
        #    image[i] = (image[i] / np.max(image[i])) * 1000
        #    image[i] = image[i].astype(np.float32)

        #    if self.preblur and scipy:
        #        image[i] = scipy.ndimage.gaussian_filter(image[i], sigma=self.preblur)
        
        #if not isinstance(image, list):
        #    image = image.astype(np.float32)

        return image

    
    def create_mask_image(self, image_fns):
        '''
        Create a mask image that is subtracted from the images to enhance moving features
        Seems to work well with X-ray data.
        '''

        mask_image = self._imread(image_fns[0])
        mask_image = np.min(mask_image, axis=0)

        for fn in image_fns[1:]:
            for image in self._imread(fn):
                mask_image = np.min([mask_image, image], axis=0)
        
        return mask_image


    def _measure_movement_optimized_xray_data(self, image_fns, ROIs,
            max_movement=False, results_list=None, worker_i=0, messages=[],
            _rotation=False):
        '''
        Optimized version when there's many rois and subtract previous
        is True and compare_to_first is False.
        
        Optimized version of measure_movement for scenarios when there are
        many ROIs

        '''

        results = []

        if worker_i == False:
            nexttime = time.time()

        if self.subtract_previous:
            mask_image = self.create_mask_image(image_fns)
            previous_image = self._imread(image_fns[0])[0] - mask_image
        else:
            previous_image = self._imread(image_fns[0])[0]

        previous_image = previous_image.astype(np.float32, copy=False)

        if not _rotation:
            X = [[] for roi in ROIs]
            Y = [[] for roi in ROIs]
        else:
            R = [[] for roi in ROIs]

        for i, fn in enumerate(image_fns[0:]):

            for image in self._imread(fn):

                image = image.astype(np.float32, copy=False)
                if self.subtract_previous:
                    image = image - mask_image
                
                for i_roi, ROI in enumerate(ROIs):
                    
                    if worker_i == False and nexttime < time.time():
                        percentage = int(100*(i*len(ROIs) + i_roi) / (len(ROIs)*len(image_fns)))
                        message = 'Process #1 out of {}, frame {}/{}, in ROI {}/{}. Done {}%'.format(
                                int(self.multiprocess), i+1,len(image_fns),i_roi+1,len(ROIs),int(percentage))
                        messages.append(message)
                        nexttime = time.time() + 2

                    
                    if not _rotation:
                        x, y = self._find_location(image, ROI, previous_image, 
                                max_movement=max_movement, upscale=self.upscale)

                        X[i_roi].append(x)
                        Y[i_roi].append(y)
 
                        if self.tracking_rois:
                            ROIs[i_roi] = [x, y, ROI[2], ROI[3]]

                        print('{} {}'.format(x,y))
                    else:
                        r = self._find_rotation(
                                image, [int(c) for c in ROI], previous_image, max_movement=max_movement, upscale=self.upscale, max_rotation=self.max_rotation)
                        R.append(r)
                        print(r)

                       
               
                if self.template_method == 'first':
                    # Shortcut for using the first frame as a template
                    pass
                elif self.template_method == 'previous':
                    # Shortcut for using the previous image as a template
                    previous_image = image
        
        if not _rotation:
            for x,y in zip(X,Y):
            
                x = np.asarray(x)
                y = np.asarray(y)
                
                if not self.absolute_results:
                    x = x-x[0]
                    y = y-y[0]
                    
                    if self.template_method == 'previous':
                        x = np.cumsum(x)
                        y = np.cumsum(y)

                results.append([x.tolist(), y.tolist()])
        else:
            raise NotImplementedError

        if results_list is not None:
            results_list[worker_i] = results
            return None

        return results


    def _measure_movement(self, image_fns, ROIs, max_movement=False, _rotation=False):
        '''
        Generic way to analyse movement using _find_location.
        
        Could be overridden by a cc_backend.
        '''
       
        results = []

        # Number of frames hiding in the stacks, not apparent from the image
        # file count. Can change during the analysis while reading images.
        N_stacked = 0
 
        if self.subtract_previous:
            mask_image = self.create_mask_image(image_fns)

        for i_roi, ROI in enumerate(ROIs):
            print('  _measureMovement: {}/{}'.format(i_roi+1, len(ROIs)))
            if self.template_method == 'first':
                previous_image = self._imread(image_fns[0])[0]
            elif self.template_method == 'mean':
                # Fixme: blows up if huge stack that doesnt fit in RAM
                images = []
                for fn in image_fns:
                    for image in self._imread(fn):
                        images.append(image)
                previous_image = np.mean(images, axis=0)

            previous_image = previous_image.astype(np.float32, copy=False)
            X = []
            Y = []
            R = [] # rotations

            i_frame = 0

            for fn in image_fns[0:]:

                images = self._imread(fn)
                N_stacked += len(images) - 1
                  
                for image in images:

                    image = image.astype(np.float32, copy=False)

                    print('ROI IS {}'.format(ROI))
                    print('Frame {}/{}'.format(i_frame, len(image_fns)+N_stacked))
                    
                    if self.template_method == 'previous':
                        if self.subtract_previous:
                            # FIXME Possible bug here
                            previous_image = image -  mask_image
                        else:
                            previous_image = image
                    
                    if self.subtract_previous:
                        image = image - mask_image
                    
                    if not _rotation:
                        x, y = self._find_location(image, [int(c) for c in ROI], previous_image, 
                                max_movement=max_movement, upscale=self.upscale)
                        X.append(x)
                        Y.append(y)

                        if self.tracking_rois:
                            #print('roi tracking')
                            #raise NotImplementedError
                            ROI = [x, y, ROI[2], ROI[3]]

                        print('{} {}'.format(x,y))
                    else:
                        r = self._find_rotation(
                                image, [int(c) for c in ROI], previous_image, max_movement=max_movement, upscale=self.upscale, max_rotation=self.max_rotation)
                        R.append(r)
                        print(r)

                    i_frame += 1

            if not _rotation:
                X = np.asarray(X)
                Y = np.asarray(Y)

                if not self.absolute_results:
                    X = X-X[0]
                    Y = Y-Y[0]

                    if self.template_method == 'previous':
                        X = np.cumsum(X)
                        Y = np.cumsum(Y)

                results.append([X.tolist(), Y.tolist()])

            else:
                results.append(R)

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

    
    def measure_rotation(self, stack_i):
        '''Runs rotation analysis
        '''
        return self.measure_movement(stack_i, optimized=False, _rotation=True)
   

    def measure_brightness(self, stack_i, relative='roi'):
        '''Measure brightness changes over time within the ROIs

        relative :  None or string
            If None (default) or "absolute", values are absolute.
            If "roimin", report values relative to the minimum of each ROI.
            If "roi", report values relative to each ROI (between 0 and 1)
        '''
        image_fns = self.stacks[stack_i]
        ROIs = self.ROIs[stack_i]

        results = [[[],[]] for roi in ROIs]
        
        for fn in image_fns:
            for image in self._imread(fn):
                for i_roi, roi in enumerate(ROIs):
                    x,y,w,h = roi
                    value = np.mean(image[y:y+h, x:x+w])

                    results[i_roi][0].append(value)
        
        if relative is None or relative == "absolute":
            pass
        elif relative == "roimin":
            for rresults in results:
                divider = np.min(rresults[0])
                rresults[0] = (np.array(rresults[0])/divider).tolist()
        elif relative == "roi":
            for rresults in results:
                minval = np.min(rresults[0])
                maxval = np.max(rresults[0])
                rresults[0] = ((np.array(rresults[0])-minval)/maxval).tolist()
        else:
            raise ValueError(f"Unkown value for 'relative': {relative}")

        # Dirty fix to brightness results to work with XY motion analysis code
        for rresults in results:
            rresults[1] = (np.array(rresults[0])*0).tolist()

        return results


    def measure_movement(self, stack_i, max_movement=False, optimized=False, _rotation=False):
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
            subsequent frames, in pixels. If not specified use self.max_movement
        optimized : bool
            Experimental, if true use reversed roi/image looping order.

        Returns
        -------
        results_stack_i
            [results_ROI1_for_stack_i, results_ROI2_for_stack_i, ...]
            where results_ROIj_for_stack_i = [movement_points_in_X, movement_points_in_Y]

        '''
        if not max_movement:
            max_movement = self.max_movement

        start_time = time.time()
        self.print_callback('Starting to analyse stack {}/{}'.format(stack_i+1, len(self.stacks)))
        
        # Select target _measure_movement
        if optimized:
            self.print_callback('Targeting to the optimized version')
            target = self._measure_movement_optimized_xray_data
        else:
            target = self._measure_movement

        if self.multiprocess:
        
            # Temporary EOFError fix:
            # When starting new processeses the whole
            # Movemeter class ends up pickeld. Because print_callback in tkgui
            # is set to some tkinter object, the whole tkinter session gets
            # pickled. On Windows for some reason, this cannot be pickeld.
            #
            # While this works now, this is not a good fix because if anyone
            # adds something unpickable to a Movemeter object, the same
            # happens again.
            #
            print_callback = self.print_callback
            self.print_callback = None
            # -----------------------------


            # Create multiprocessing manager and a inter-processes
            # shared results_list
            manager = multiprocessing.Manager()
            results_list = manager.list()
            messages = manager.list()
            for i in range(self.multiprocess):
                results_list.append([])
            
   
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
                                'worker_i': i_worker, 'messages': messages, '_rotation': _rotation} )
                
                workers.append(worker)
                worker.start()

            # Wait until all workers get ready
            for i_worker, worker in enumerate(workers):
                print_callback('Waiting worker #{} to finish'.format(i_worker+1))
                while worker.is_alive():
                    if messages:
                        print_callback(messages[-1])
                    time.sleep(1)
                worker.join()

            # Combine workers' results
            print_callback('Combining results from different workers')
            results = []
            for worker_results in results_list:
                results.extend(worker_results)

            # FIX EOFError, see above
            self.print_callback = print_callback
            # ---------------


        else:
            results = target(self.stacks[stack_i], self.ROIs[stack_i], max_movement=max_movement, _rotation=_rotation)
        

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
        height, width = self._imread(self.stacks[stack_i][0])[0].shape
        return width, height

       


