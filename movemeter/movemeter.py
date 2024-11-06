'''Movemeter Python API

The Movemeter class here does cross-correlation (or template matching)
based motion analysis. You can use it in your own Python programs.

Alternatives
------------
For graphical user interface, use tkgui.py
For direct command line use, use cli.py
'''

import os
import time
import multiprocessing
import warnings
import types

try:
    import exifread
except ImportError:
    exifread = None
import numpy as np
import cv2
try:
    # Optional dependency to scipy, used only for preblurring
    # (Movemeter.preblur)
    import scipy.ndimage
except ImportError:
    scipy = None
    warnings.warn('cannot import scipy.ndimage; Movemeter preblur not available')

from .cc_backends import cc_backends, get_cc_backend
from .im_backends import im_backends, get_im_backend
from .version import __version__
from movemeter.stacks import stackread


__license__ = "GPLv3-only"
# __version__ imported below



class Movemeter:
    '''A Python API for motion analysis.

    Movemeter analyses mainly translational motion using 2D
    cross-correlation (aka. template matching). Brightness analysis and
    rotational movement analysis are also supported.

    The actual motion analaysis is performed by an backend (default
    the opencv backend). Similarly, there are backends for image
    loading.


    Example
    -------
        meter = Movemeter()                 # Create instance
        meter.set_data(image_stacks, ROIs)  # Set data and ROIs
        meter.measure_movement(0)           # Run analysis on stack 0

    Many of the attributes can be set at the init or then
    modified after the meter is created.

    The set_data(stacks, ROIs) can be hard to get right - see its
    docstring for more clarification


    Attributes
    ----------
    upscale : int or float
        Image upscaling used in motion analysis. Enables subpixel
        resolution. For example, with the upscale value 10, the smallest
        reportable motion step is 1/10 of a pixel. Higher are slow down
        the analysis and use more RAM memory.
    cc_backend : string
        Motion analysis backend. "OpenCV"
    im_backend : string or callable
        Image loading backend. If string, "stackread",
        "OpenCV" or "tifffile".
    absolute_results : bool
        If True, return results in absolute image coordinates. If false,
        returns results relative to the corresponding ROI.
    tracking_rois : bool
        If True, ROIs are shifted between frames, following the moving
        features. If False, the ROIs are stationary.
    template_method : string
        Template image creation method used in the motion analysis.

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
        If 0 then no multiprocessing. Otherwise, the number of parallel
        processes. Note that there may be some multiprocessing alread
        at the cc_backend level. If used, avoid adding any non-pickable
        or heavy attributes to the instances of this class.
    print_callback : callable
        Print function to convey the progress.
        By default, this is the built-in print function.
    preblur : False or float
        Standard deviation of the Gaussian blur kernel, see
        scipy.ndimage.gaussian_filter Requires an optional dependency
        to scipy to work.
    max_movement : None or int
        If not None, in pixels, the maximum expected per-frame
        (compare_to_first=False) or total (compare_to_first=True) motion
        analysis result in pixel. Essentially crops the source image
        around the ROI. Use leads to increased performance reliability.
        Too small values small values will truncated and underestimated
        motion, too large can wrong matches.
    ''' 

    measure_brightness_opt = {
            'relative': ['absolute', 'roi', 'roimin'],
            }

    def __init__(
            self,
            upscale=1,
            cc_backend='OpenCV',
            imload_backend='stackread',
            absolute_results=False,
            tracking_rois=False,
            template_method='first',
            subtract_previous=False,
            multiprocess=False,
            print_callback=print,
            preblur=0,
            max_movement=None,
            max_rotation=None
            ):

        # Initialize attributes to defaults or given values
        self.upscale = upscale
        self.cc_backend = cc_backend
        self.im_backend = imload_backend
        self.absolute_results = absolute_results
        self.tracking_rois = tracking_rois
        self.template_method = template_method
        self.subtract_previous = subtract_previous
        self.multiprocess = multiprocess
        self.print_callback = print_callback
        self.preblur=preblur
        self.max_movement = max_movement
        self.max_rotation = max_rotation

        # Cross-correlation backend
        self._find_location, self._find_rotation = get_cc_backend(
                cc_backend)


    def _imread(self, fn):
        '''Wraps the image loading backend (self._imload).
        
        Arguments
        ---------
        fn : string
            Filename of the stack or image to read.

        Returns
        -------
        stack : iterable
            Returns something that when you iterate over it, it returns
            valid images
        '''

        # Image loading backend
        if self.im_backend == "stackread":
            imload = stackread
        else: 
            imload = get_im_backend(imload_backend)

        if isinstance(fn, str):
            return imload(fn)
        elif isinstance(fn, np.ndarray):
            return fn
        elif hasattr(fn, '__iter__') or hasattr(fn, '__getitem__'):
            return fn
        else:
            print(f'Warning! Unkown fn: {fn}')
            return fn

        return image

    def _im_preprocess(image):
        '''Peform preprocessing on the image before the cc backend

        FIXME: Not currently called from anywhere in the code
        '''
        # FIXME Normalization not sure if needed
        normalize = False
        if normalize:
            image[i] -= np.min(image[i])
            image[i] = (image[i] / np.max(image[i])) * 1000
            image[i] = image[i].astype(np.float32)

        if self.preblur and scipy:
            image[i] = scipy.ndimage.gaussian_filter(
                    image[i], sigma=self.preblur)
        
        return image
       
    
    def create_mask_image(self, image_fns):
        '''Creates a min-mask image

        The problem sometimes with cross-correlation analysis is that
        the moving features are semi-transparent or faint while there's
        a stationary, strong-featured background.

        Subtracting the min-mask image of a stack can be used to remove
        some stationary features -> Working motion detection.

        Seems to work well with microsaccades X-ray projections data.

        Arguments
        ---------
        image_fns : list of strings
            Filenames of the images in the stack

        Returns
        -------
        mask_image : 2D numpy array
            Min image of the stack
        '''

        mask_image = self._imread(image_fns[0])
        mask_image = np.min(mask_image, axis=0)

        for fn in image_fns[1:]:
            for image in self._imread(fn):
                mask_image = np.min([mask_image, image], axis=0)
        
        return mask_image


    def _measure_movement_optimized_xray_data(
            self, image_fns, ROIs, max_movement=False,
            results_list=None, worker_i=0, messages=[],
            _rotation=False):
        '''Optimized version for many ROIs (once iterate over images)
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
                        x, y = self._find_location(
                                image, ROI,
                                previous_image, 
                                max_movement=max_movement,
                                upscale=self.upscale)

                        X[i_roi].append(x)
                        Y[i_roi].append(y)
 
                        if self.tracking_rois:
                            ROIs[i_roi] = [x, y, ROI[2], ROI[3]]

                    else:
                        r = self._find_rotation(
                                image, ROI,
                                previous_image,
                                max_movement=max_movement,
                                upscale=self.upscale,
                                max_rotation=self.max_rotation)

                        R.append(r)

                       
               
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
        '''Measure movement for 1 or few ROIs.

        This vanilla version is there just to check that the optimized
        version works.
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
                        x, y = self._find_location(
                                image, ROI,
                                previous_image, 
                                max_movement=max_movement,
                                upscale=self.upscale)

                        X.append(x)
                        Y.append(y)

                        if self.tracking_rois:
                            #print('roi tracking')
                            #raise NotImplementedError
                            ROI = [x, y, ROI[2], ROI[3]]

                    else:
                        r = self._find_rotation(
                                image, ROI,
                                previous_image,
                                max_movement=max_movement,
                                upscale=self.upscale,
                                max_rotation=self.max_rotation)

                        R.append(r)

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
        ''' Set stack filenames and ROIs (regions of interest) to be analysed.

        Arguments
        ---------
        stacks : list of lists of filenames
            List of filename lists. In format
            [
              [stack1_im1, stack1_im2...],
              [stack2_im1, stack2_im2], ...
            ]
        ROIs : list of lists of ROIs
            [
              [ROI1_for_stack1, ROI2_for_stack1, ...],
              [ROI1_for_stack2, ...], ...
            ]

            ROI format: (x, y, w, h)
        
            If the list has length, it means use the same provided
            ROIs for all stacks.

        Example 1 - One stack made separate images, one roi
        ----------------------------------------------------
            stacks = [['frame1.tif', 'frame2.tif', ...]]
            ROIs = [[(0,0,24,24)]]

        Example 2 - One stack actually one stack on disk, two rois
        ----------------------------------------------------------
            stacks = [['stack.tif']]
            ROIs = [[(0,0,24,24), (24,24,24,24)]]

        '''
        N_rois = len(ROIs)
        N_stacks = len(stacks)
 
        # A) Stacks
        # FIXME: Validity checks for stacks
        self.stacks = stacks
        
        # B) Determine ROIs relationship to the stacks
        self.print_callback('Determining stack/ROI relationships in movemeter')
        if N_rois > 1:
            # Separate ROIs for each stack
            self.ROIs = ROIs
        elif N_rois == 1:
            # Same ROIs for all the stacks
            self.ROIs = [ROIs[0] for i in range(len(stacks))]    
        elif N_rois != N_stacks:
            raise ValueError(
                    f"Stacks and ROIs lenghts do not match ({N_stacks} vs {N_rois})")
        
        # Make all ROIs into ints
        self.ROIs = [[[int(x), int(y), int(w), int(h)] for x,y,w,h in ROI] for ROI in self.ROIs]

    
    def measure_rotation(self, stack_i, optimized=False):
        '''Measures rotation changes over time within the ROIs

        Arguments
        ---------
        stacks_i : int
            The index of the stack to analyse (order as in set_data)
        optimized : bool
            Run the optimized version for many ROIs (experimental)
        '''
        return self.measure_movement(
                stack_i, optimized=optimized, _rotation=True)
   

    def measure_brightness(self, stack_i, relative='roi'):
        '''Measures brightness changes over time within the ROIs
        
        Brightness measurement is independent of the cross-correlation
        backends.

        Arguments
        ---------
        stack_i : int
            The index of the stack to analyse (order as in set_data)
        relative : string or None
            How to report the brightness change calues.
            If None or "absolute", values are absolute.
            If "roimin", values relative to the minimum of each ROI.
            If "roi", values relative to each ROI (between 0 and 1)
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

        # Dirty fix to brightness results to work with XY motion
        # analysis based code elsewhere (add zero y component)
        for rresults in results:
            rresults[1] = (np.array(rresults[0])*0).tolist()

        return results


    def measure_movement(self, stack_i, max_movement=False, optimized=False, _rotation=False):
        '''Measures translational movement over time within the ROIs.

        Arguments
        ---------
        stack_i : int
            The index of the stack to analyse (order as in set_data)
        max_movement : int
            Speed up the computation (and increase reliability) by
            specifying the maximum translation between subsequent frames,
            in pixels. If not specified, uses self.max_movement
        optimized : bool
            Run the optimized version for many ROIs (experimental)

        Returns
        -------
        results_stack_i : list of lists
            [
              results_ROI1, results_ROI2, ...
            ]
            
            results_ROIj = [movement_points_in_X, movement_points_in_Y]

        '''
        if not max_movement:
            max_movement = self.max_movement

        start_time = time.time()
        self.print_callback(
                'Starting to analyse stack {}/{}'.format(
                    stack_i+1, len(self.stacks)))
        
        # Select target _measure_movement
        if optimized:
            self.print_callback('Targeting to the optimized version')
            target = self._measure_movement_optimized_xray_data
        else:
            target = self._measure_movement

        if self.multiprocess:
        
            # -----------------------------
            # Temporary EOFError fix:
            # When starting new processes, the whole
            # Movemeter class ends up pickled. Because print_callback in
            # tkgui is set to some tkinter object, the whole tkinter
            # session gets pickled. On Windows, for some reason, this
            # object cannot be pickeld.
            #
            # While this works now, this is not a good fix because if
            # anyone adds something unpickable to a Movemeter object,
            # the same error happens again.
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
                
                worker = multiprocessing.Process(
                        target=target,
                        args=[self.stacks[stack_i], worker_ROIs],
                        kwargs={
                            'max_movement': max_movement,
                            'results_list': results_list,
                            'worker_i': i_worker,
                            'messages': messages,
                            '_rotation': _rotation}
                        )
                
                workers.append(worker)
                worker.start()

            # Wait until all workers get ready
            for i_worker, worker in enumerate(workers):
                print_callback('Waiting worker #{} to finish'.format(
                    i_worker+1))
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

            # -----------------------------
            # FIX EOFError, latter part, see above
            self.print_callback = print_callback
            # -----------------------------


        else:
            # So nice and simple without multiprocessing ;)
            results = target(
                    self.stacks[stack_i],
                    self.ROIs[stack_i],
                    max_movement=max_movement,
                    _rotation=_rotation
                    )
        
        self.print_callback(
                'Finished stack {}/{} in {} seconds'.format(
                    stack_i+1, len(self.stacks), time.time()-start_time))

        return results 

    

    def get_metadata(self, stack_i, image_i=0):
        '''Get metadata for stack number stack_i.
        
        Arguments
        ---------
        stack_i : int
            The index of the stack to analyse (order as in set_data)
        image_i : int
            Optionally, the index of the image from which to read
            the metadata from. Can be different especially if stack
            is made of separate images on disk.

        Returns
        -------
        tags: dict
            A dictionary of exifread objects. See exifread documentation.
        '''
        # FIXME - Instead of using exifread here, we should
        # implement metadata reading (using exifread or something else)
        # in the image loading backends
        if exifread is None:
            return {}
        with open(self.stacks[stack_i][image_i], 'rb') as fp:
            tags = exifread.process_file(fp)

        return tags
    

    def get_image_resolution(self, stack_i):
        '''Returns resolution of the images in stack_i.

        Currently opens the first image to see the resolution (slow).
        Would be better to read from the metadata directly...
        
        Arguments
        ----------
        stack_i : int
            The index of the stack to analyse (order as in set_data)

        Returns
        -------
        width : int
            Width of the image, in pixels
        height : int
            Height of the image, in pixels

        '''
        height, width = self._imread(self.stacks[stack_i][0])[0].shape
        return width, height

       


