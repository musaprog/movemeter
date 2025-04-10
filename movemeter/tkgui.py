'''A tkinter/tk GUI for Movemeter motion analysis.

Features in short
-----------------
- Load, view and exclude images
- Draw variously shaped ROIs that are made from small,
    rectangular (square) cross-correlation windows,
    and allow grouping these ROIs
- Perform motion analysis and save results
- View and save motion analysis heatmaps


This file contains most of the GUI elements extra logic (such as saving results)
that are not present in the movemeter.py file.
'''

import os
import csv
import json
import datetime
import zipfile
import inspect

import numpy as np
import tifffile
import tkinter as tk
from tkinter import filedialog, simpledialog
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.cm
import matplotlib.colors
import matplotlib.transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from tk_steroids.elements import (
        Listbox,
        Tabs,
        TickboxFrame,
        ButtonsFrame,
        DropdownList,
        )
from tk_steroids.matplotlib import CanvasPlotter

from movemeter import __version__
from movemeter.directories import MOVEDIR
from movemeter.roi import (
        gen_grid,
        grid_along_ellipse,
        grid_along_line,
        grid_arc_from_points,
        grid_radial_line_from_points,
        _workout_circle,
        )
from movemeter import Movemeter
from movemeter.tk_heatmap import popup as open_httool


class ColormapSelector(tk.Frame):
    '''
    Widget to preview and select a matplotlib colormap.
    '''
    def __init__(self, tk_parent, callback, startmap=None):
        '''
        tk_parent : object
            Tkinter parent widget
        callback : callable
            When selected, the colormap passed to this callback function
        startmap : string
            Name of the colormap to start with.
        '''
        tk.Frame.__init__(self, tk_parent)
        
        self._callback = callback

        # Dict of all availbale colormap objects
        self.colormaps = {name: getattr(matplotlib.cm, name) for name in dir(matplotlib.cm) if isinstance(
            getattr(matplotlib.cm, name), matplotlib.colors.Colormap)}

       
        self.listbox = Listbox(self, list(self.colormaps.keys()), callback=self.on_selection)
        self.listbox.grid(row=1, column=1, sticky='NSWE')

        self.plotter = CanvasPlotter(self, text='Preview', figsize=(0.5,5))
        self.plotter.grid(row=1, column=2, sticky='NSWE')
        
        data = np.linspace(0,10)[:, np.newaxis]
        self.plotter.imshow(data)
        if startmap:
            self.on_selection(startmap)

        self.select_button = tk.Button(self, text='Ok', command=self.on_ok)
        self.select_button.grid(row=2, column=1, columnspan=2, sticky='NSWE')

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=10)
        self.grid_columnconfigure(1, weight=1)


    def on_selection(self, name):
        self._current = name
        self.plotter.imshow_obj.cmap = self.colormaps[name]
        self.plotter.update()


    def on_ok(self):
        self._callback(self.colormaps[self._current])



class MovemeterSettings(tk.Frame):
    '''
    Movemeter settings widget, semi-automatically inspected from
    the Movemeter.__init__ method.

    Use get_current method to retrive the setting dictionary.

    Attributes
    ----------
    tickboxes : object
        tk_steroids TickboxFrame containing False/True options.
    maxmovement_slider, blur_slider, cores_slider, upscale_slider: object
        tkinter Slider widgets
    '''
    def __init__(self, tk_parent):
        '''
        tk_parent       Tkinter parent widget
        '''
        tk.Frame.__init__(self, tk_parent)
        self.columnconfigure(2, weight=1)

        # Movemeter True/False options; Automatically inspect from Movemeter.__init__
        moveinsp = inspect.getfullargspec(Movemeter.__init__)

        moveargs = []
        movedefaults = []
        for i in range(1, len(moveinsp.args)):
            arg = moveinsp.args[i]
            default = moveinsp.defaults[i-1]
            if isinstance(default, bool) and arg not in ['multiprocess']:
                moveargs.append(arg)
                movedefaults.append(default)
        

        # GUI elements next
        # True/false - motion analysis options
        self.tickboxes = TickboxFrame(self, moveargs, defaults=movedefaults)
        self.tickboxes.grid(row=2, column=1, columnspan=2)
        
        # Preprocessing options
        tk.Label(self, text='Gaussian blur').grid(row=3, column=1)
        self.blur_slider = tk.Scale(self, from_=0, to=32, orient=tk.HORIZONTAL)
        self.blur_slider.set(0)
        self.blur_slider.grid(row=3, column=2, sticky='NSWE')

        # Numerical value - motion analysis options
        tk.Label(self, text='Maximum movement').grid(row=4, column=1)
        self.maxmovement_slider = tk.Scale(self, from_=1, to=100,
                orient=tk.HORIZONTAL)
        self.maxmovement_slider.set(10)
        self.maxmovement_slider.grid(row=4, column=2, sticky='NSWE')

        tk.Label(self, text='Upscale').grid(row=5, column=1)
        self.upscale_slider = tk.Scale(self, from_=0.1, to=10,
                orient=tk.HORIZONTAL, resolution=0.1)
        self.upscale_slider.set(5)
        self.upscale_slider.grid(row=5, column=2, sticky='NSWE')

        tk.Label(self, text='Parallel processes').grid(row=6, column=1)
        self.cores_slider = tk.Scale(self, from_=1, to=os.cpu_count(),
                orient=tk.HORIZONTAL)
        self.cores_slider.set(max(1, int(os.cpu_count()/2)))
        self.cores_slider.grid(row=6, column=2, sticky='NSWE')


    def get_current(self):
        '''
        Returns a dictionary of the current settings that can be directly
        passed to the Movemeter.__init__ method.
        '''
        settings = {'upscale': float(self.upscale_slider.get()),
                'max_movement': int(self.maxmovement_slider.get()),
                'multiprocess': int(self.cores_slider.get())}

        if settings['multiprocess'] == 1:
            settings['multiprocess'] = False

        return {**self.tickboxes.states, **settings}


class MovemeterMenubar:
    '''Add the movemeter top menu to the application

    The parent widget has to be MovemeterTkGUI widget with methods
    like open_stack, open_directory etc.
    '''

    def __init__(self, parent):

        self.parent = parent

        self.menu = tk.Menu(parent)
        
        filemenu = tk.Menu(parent)
        filemenu.add_command(
                label='Add stack...', command=parent.folview.open_stack)
        filemenu.add_command(
                label='Add folder...', command=parent.folview.open_directory)
        filemenu.add_separator()
        filemenu.add_command(
                label='Load ROIs',
                command=lambda: parent.apply_movzip(rois=True))
        filemenu.add_command(
                label='Save ROIs',
                command=lambda: parent._save_movzip(
                    only=['rois', 'selections']))
        
        filemenu.add_separator()
        filemenu.add_command(label='Save ROI view',
                command=parent.save_roiview)
        
        filemenu.add_command(label='Save ROIs only view',
                command=lambda: parent.save_roiview(only_rois=True))
        filemenu.add_separator()
        filemenu.add_command(label='Quit', command=parent.parent.destroy) 
        
        self.menu.add_cascade(label='File', menu=filemenu)
        

        editmenu = tk.Menu(parent)
        editmenu.add_command(
                label='Undo (latest ROI)', command=parent.roidrawer.undo)
        editmenu.add_separator()
        editmenu.add_command(
                label='Global settings', command=parent.open_settings)
        self.menu.add_cascade(label='Edit', menu=editmenu)

        viewmenu = tk.Menu(parent)
        viewmenu.add_command(
                label='Show image controls',
                command=parent.imview.toggle_controls)
        self.menu.add_cascade(label='View', menu=viewmenu)
        
        batchmenu = tk.Menu(parent)
        batchmenu.add_command(
                label='Batch measure & save all',
                command=parent.batch_process)
        batchmenu.add_separator()
        batchmenu.add_command(
                label='Reprocess rectangular selection (with current block settings)',
                command=parent.recalculate_old)
        batchmenu.add_command(
                label='Replot heatmap', command=parent.replot_heatmap)
 
        self.menu.add_cascade(label='Batch', menu=batchmenu)

        toolmenu = tk.Menu(parent)
        toolmenu.add_command(
                label='Heatmap tool', command=lambda: open_httool(parent))
        self.menu.add_cascade(label='Tools', menu=toolmenu)
        
        parent.parent.config(menu=self.menu)



class DataInputWidget(tk.Frame):
    '''Imports user data and sets the active stack
    
    Attributes
    ----------
    folders : list
        List of opened directories and image stack files
    folders_listbox : object
        tk_steroids Listbox of opened directories
    current_folder : string
        The currently selected folder from self.folder
    image_fns : string
        List of image filenames in th current folder.
    images : list of Nones or list of ndarray
        Initially list of Nones, as long as many images there are.
        Incrimentally, becomes a list of images (numpy array).
    fs : int or float
        Sampling rate of the images, in Hz (1/s). Global for all data.
    filename_extensions : tuple of strings
        Accepted filename extensions for images (or videos).
    self.N_frames : dict
        For video files of image stacks, contains the amount of frames
        per each file and the filenames are the keys.
    '''
        
    def __init__(self, parent, movemeter, callback, statusbar=None):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.statusbar = statusbar
        self.movemeter = movemeter
        self.callback = callback

        # Data and images
        self.folders = []
        self.current_folder = None
        self.image_fns = []
        self.images = None
        self.fs = 100
       
        self.exclude_images = []

        self.filename_extensions = ('.tiff', '.tif', '.mp4')
        self.N_frames = {}

        # Other
        self.use_mask_image = False
        self.mask_image = None
 

        # Input folders

        self.folders_listbox = Listbox(self, ['No folders selected'], self.folder_selected)
        self.folders_listbox.listbox.config(height=10)
        self.folders_listbox.grid(row=2, column=1, columnspan=2, sticky='NSWE')

        self.folview_buttons = ButtonsFrame(self,
                ['Add stack...', 'Add folder...', 'Remove', 'FS'],
                [self.open_stack, self.open_directory, self.remove_directory, self.set_fs])
        self.folview_buttons.grid(row=0, column=1) 
        self.fs_button = self.folview_buttons.buttons[3]
        self.set_fs(fs=self.fs)

   
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2,weight=1)

    def set_status(self, *args, **kwargs):
        if self.statusbar:
            self.statusbar.set_status(*args, **kwargs)


    def set_fs(self, fs=None):
        '''
        Opens a dialog to set the image sampling frequency (frame rate) so that
        time axises come correctly.
        '''
        if fs is None:
            fs = simpledialog.askfloat('Imaging frequency (Hz)', 'How many images were taken per second')

        if fs:
            self.fs = fs
            self.fs_button.configure(text='fs = {} Hz'.format(self.fs))


    def _get_previous_directory(self):
        try: 
            with open(os.path.join(MOVEDIR, 'last_directory.txt'), 'r') as fp:
                previous_directory = fp.read().rstrip('\n')
        except FileNotFoundError:
            previous_directory = os.getcwd()

        if os.path.exists(previous_directory):
            return previous_directory
        return None


    def _set_previous_directory(self, directory):
        if not os.path.isdir(MOVEDIR):
            os.makedirs(MOVEDIR)
        with open(os.path.join(MOVEDIR, 'last_directory.txt'), 'w') as fp:
            fp.write(directory)
         

    def open_stack(self, stack_fn=None):
        '''Add a stack on the list of input folders

        Similar to open_directory but let's to select a stack
        from folder instead of all the folder's contents
        '''

        if stack_fn is None:
            previous_dir = self._get_previous_directory()

            stack_fn = filedialog.askopenfilename(
                title='Select an image stack',
                initialdir=previous_dir)
        
        if stack_fn:
            self._set_previous_directory(
                    os.path.dirname(stack_fn))
            
            self.set_status(f'Added a new stack {stack_fn}')
            
            self.folders.append(stack_fn)
            self.folders_listbox.set_selections(self.folders)
            self.folder_selected(stack_fn, usermade=False)



    def open_directory(self, directory=None):
        '''
        Open a dialog to select a data directory and adds it to the
        list of open directories.
        '''
        if directory is None:
            previous_dir = self._get_previous_directory()

            directory = filedialog.askdirectory(
                    title='Select directory with the images',
                    initialdir=previous_dir)
            
            
        if directory:
           
            self._set_previous_directory(directory)

            # Check if folder contains any images; If not and it contains folders, append
            # The folders in this folder

            contents = os.listdir(directory)
            noimages = [fn for fn in os.listdir(directory) if fn.endswith(self.filename_extensions)] == []
            has_subfolders = any([os.path.isdir(os.path.join(directory, fn)) for fn in contents])
            
            if noimages and has_subfolders:
                directories = [os.path.join(directory, fn) for fn in os.listdir(directory)]
                self.set_status('Added {} new directories'.format(len(directories)))
            else:
                directories = [directory]
                self.set_status('Added directory {}'.format(directory))
            
            for directory in directories:
                self.folders.append(directory)
                self.folders_listbox.set_selections(self.folders)
                self.folder_selected(directory, usermade=False)

    
    def remove_directory(self):
        '''
        Closes a directory from the list of open data directories.
        '''
        self.folders.remove(self.current_folder)
        self.folders_listbox.set_selections(self.folders)

        self.set_status('Closed directory {}'.format(self.current_folder))
   

    def clear_directories(self):
        '''Clear all added stacks and folders
        '''
        self.folders = []
        self.folders_listbox.set_selections([])

    
    def _imread(self, fn):
        '''
        Use Movemeter to open image/video.
        '''
        images = self.movemeter._imread(fn)
        return images

    @property
    def image_shape(self):
        #slider_value = int(self.image_slider.get())
        #image_i = int(slider_value) -1
        #if self.images[image_i] is None:
        #    self.images[image_i] = self._imread(self.image_fns[image_i])[0]
        #return self.images[image_i].shape
        for image in self.images:
            if image is not None:
                return self.images[image_i].shape
        return self._imread(self.image_fns[0])[0].shape

    def folder_selected(self, folder, usermade=True):
        '''
        When the user selects a folder from the list of open data
        directories (that is self.folders_listbox)

        Arguments
        ---------
        usermade : bool
            If true, update the status bar
        '''
        
        self.current_folder = folder

        if os.path.isdir(folder):
            # Folder of separate image frame files
            if usermade:
                self.set_status('Selected folder {}'.format(folder))
            self.image_fns = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith(self.filename_extensions)]
            self.image_fns.sort()
        else:
            # One image stack files
            if usermade:
                self.set_status(f'Selected stack {folder}')
            self.image_fns = [folder]

        self.N_frames = {}
        total_frames = 0
        for fn in self.image_fns:
            N = len(self._imread(fn))
            total_frames += N
            if N > 1:
                self.N_frames[fn] = N

        N_images = total_frames

        self.images = [None for i in range(N_images)]
        self.mask_image = None
        
        if callable(self.callback):
            self.callback()

    
    def _get_fn_and_frame(self, i_image):
        '''
        Workaround needed for video/stack files, getting the correct
        filename and frame for the ith image.

        Arguments
        ---------
        i_image : int
            Index of the image.

        Returns
        -------
        i_fn : int
            Index of the file name in self.datainput.image_fns
        i_frame : int
            Index of the frame in the video/stack file.
        '''
        total_frames = 0
        for i_fn, fn in enumerate(self.image_fns):
            frames = self.N_frames.get(fn, 1)
            total_frames += frames

            if total_frames >= i_image:
                return i_fn, frames - (total_frames - i_image)
 
 
    def get_image(self, i_image):

        if not 0 <= i_image < len(self.images):
            return None
        
        if self.use_mask_image:
            if self.mask_image is None:
                for i in range(len(self.images)):

                    self.datainput.images[i] = self.datainput._imread(self.datainput.image_fns[i])
                
                self.mask_image = np.inf * np.ones(self.image_shape)
                
                for image in self.images:
                    self.mask_image = np.min([self.mask_image, image], axis=0)


        i_fn, i_frame = self._get_fn_and_frame(i_image)
        
        if self.images[i_image] is None:
            self.images[i_image] = self._imread(self.image_fns[i_fn])[i_frame] 
 
        if self.use_mask_image:
            showimage = self.images[i_image] - self.mask_image
        else:
            showimage = self.images[i_image]

        return showimage

    def _included_image_fns(self):
        return [fn for i_fn, fn in enumerate(self.image_fns) if fn not in self.exclude_images and i_fn not in self.exclude_images]
    
    def _len_included_frames(self):
        return sum([self.N_frames.get(fn, 1) for fn in self._included_image_fns()])


class ColorsWidget(tk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        
        self.colors = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.tab10)
        self.colors.set_clim(0,10)

        self.buttons = ButtonsFrame(
                self,
                ['Set colormap'],
                [self.open_colormap_selection],
                )
        self.buttons.grid()
    
    def connect_widgets(self, roidrawer=None):
        if roidrawer:
            self.roidrawer = roidrawer

    def open_colormap_selection(self):
        '''
        Start ColormapSelector widget in a toplevel window.
        '''
        top = tk.Toplevel(self)
        top.title('Select colormap')
        sel = ColormapSelector(top, callback=self.apply_colormap,
                startmap=self.colors.get_cmap().name)
        sel.grid(row=0, column=0, sticky='NSWE')
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        top.mainloop()


    def apply_colormap(self, colormap):
        if hasattr(colormap, 'colors'):
            self.colors.set_clim(0, len(colormap.colors))
        else:
            self.colors.set_clim(0, 10)

        self.colors.set_cmap(colormap)

        if self.roidrawer:
            self.roidrawer.update_grid()


class ImageROIWidget(tk.Frame):
    '''Displays the image and ROIs
    '''

    def __init__(self, tk_parent):
        tk.Frame.__init__(self, tk_parent)
        self.parent = tk_parent


        self.show_controls = False

        self.datainput = None
        self.roidrawer = None
        self.statusbar = None
        

        # Images view: Image looking and ROI selection
        # -------------------------------------------------
        
        self.imview_buttons = ButtonsFrame(self,
                ['Exclude image', 'Exclude index'],
                [self.toggle_exclude,
                 lambda: self.toggle_exclude(by_index=True),
                 ])
        
        self.imview_buttons.grid(row=1, column=1)


        self.image_slider = tk.Scale(self, from_=0, to=0,
                orient=tk.HORIZONTAL, command=self.change_image)
        
        self.image_slider.grid(row=2, column=1, sticky='NSWE')

        self.images_plotter = CanvasPlotter(self)
        self.images_plotter.grid(row=3, column=1, sticky='NSWE') 
        
        ax = self.images_plotter.ax
        self.excludetext = ax.text(0.5, 0.5, '', transform=ax.transAxes,
                fontsize=24, ha='center', va='center', color='red')

        tk.Label(self.imview_buttons, text='Line width').grid(row=2, column=0)
        self.patch_lw_slider = tk.Scale(self.imview_buttons, from_=0, to_=10,
                orient=tk.HORIZONTAL)
        self.patch_lw_slider.set(1)
        self.patch_lw_slider.grid(row=2, column=1, sticky='NSWE')


        tk.Label(self.imview_buttons, text='Fill strength').grid(row=2, column=3)
        self.patch_fill_slider = tk.Scale(self.imview_buttons, from_=0, to=100,
                orient=tk.HORIZONTAL)
        self.patch_fill_slider.grid(row=2, column=4, sticky='NSWE')
        self.patch_fill_slider.set(40)




    
    def connect_widgets(self, datainput=None, roidrawer=None,
                        statusbar=None, colors=None):
        if datainput is not None:
            self.datainput = datainput
            self.datainput.callback = self.reset_slider
        if roidrawer is not None:
            self.roidrawer = roidrawer
        if statusbar is not None:
            self.statusbar = statusbar
        if colors is not None:
            self.colors = colors
   
    def reset_slider(self):
        self.change_image(slider_value=1)
        self.image_slider.config(from_=1, to=len(self.datainput.images))
       
        #self.export_name.delete(0, tk.END)
        #self.export_name.insert(0, os.path.basename(folder.rstrip('/')))


    def toggle_exclude(self, by_index=False):
        '''
        Look at the currently shown image and toggle its excludance.
        
        Arguments
        ---------
        by_index : bool
            If true, toggle exclude for all images with this index.
            If false, exclude the filename only.
        '''

        indx = int(self.image_slider.get()) - 1
        if by_index:
            fn = indx
        else:
            fn = self.datainput.image_fns[indx]

        if fn not in self.datainput.exclude_images:
            self.datainput.exclude_images.append(fn)
            self.set_status('Removed image {} from the analysis'.format(fn))
        else:
            self.datainput.exclude_images.remove(fn) 
            self.set_status('Added image {} back to the analysis'.format(fn))
        
        self.datainput.mask_image = None
        self.change_image(slider_value=self.image_slider.get())
    

    def toggle_controls(self):
        '''
        Show/hide image brightness/contrast controls.
        '''
        self.show_controls = not(self.show_controls)
        self.change_image()


    def change_image(self, slider_value=None):
        '''
        Change the currently shown data image.
        '''
        slider_value = int(self.image_slider.get())

        image_i = int(slider_value) -1

        image = self.datainput.get_image(image_i)
        if image is None:
            return

        i_fn, i_frame = self.datainput._get_fn_and_frame(image_i)
        
        if image_i in self.datainput.exclude_images or self.datainput.image_fns[i_fn] in self.datainput.exclude_images:
            self.excludetext.set_text('EXCLUDED')
        else: 
            self.excludetext.set_text('')

        self.images_plotter.imshow(image, roi_callback=self.roidrawer.set_roi,
                cmap='gray', slider=self.show_controls,
                roi_drawtype=self.roidrawer.roi_drawtypes[self.roidrawer.roitype_selection.ticked[0]])


   
    def set_status(self, *args, **kwargs):
        if self.statusbar:
            self.statusbar.set_status(*args, **kwargs)




class BoxRoiDrawer:
    '''Draw various box ROIs for movemeter use
    '''

    def __init__(self, imview, settings_frame):

        self.imview = imview
        self.settings_frame = settings_frame
        self.statusbar = None

        self.roiview = self.settings_frame

        # Selections and ROIs
        self.selections = []
        self.roi_groups = []
        self.current_roi_group = 0
        self.roi_patches = []
       

        self.roitypes = [
                'free', 'box', 'ellipse', 'line', 'polygon',
                'arc_from_points', 'concentric_arcs_from_points',
                'radial_lines_from_points'
                ]

        self.roi_drawtypes = {
                'free': 'box',
                'box': 'box',
                'ellipse': 'ellipse',
                'line': 'line',
                'polygon': 'polygon',
                'arc_from_points': 'polygon',
                'concentric_arcs_from_points': 'polygon',
                'radial_lines_from_points': 'polygon'}

        tk.Label(self.roiview, text='Selection mode').grid(row=1, column=1)
        self.selmode_frame = tk.Frame(self.roiview)
        self.selmode_frame.grid(row=1, column=2)
        
        self.roitype_selection = DropdownList(
                self.selmode_frame, self.roitypes,
                ['Free', 'Box', 'Ellipse', 'Line', 'Polygon', 'Arc from points',
                    'Concentric Arcs (++RG)',
                    'Radial lines (++RG)'],
                single_select=True, callback=self.update_roitype_selection)
        self.roitype_selection.grid(row=1, column=2)

        self.drawmode_selection = TickboxFrame(self.selmode_frame,
                ['add', 'remove'], ['Add', 'Remove'],
                single_select=True
                )
        self.drawmode_selection.grid(row=1, column=1)


        self.blocksize_label = tk.Label(self.roiview, text='Block size')
        self.blocksize_label.grid(row=3, column=1)
        self.blocksize_slider = tk.Scale(self.roiview, from_=16, to=128,
                orient=tk.HORIZONTAL)
        self.blocksize_slider.set(32)
        self.blocksize_slider.grid(row=3, column=2, sticky='NSWE')

        self.overlap_label = tk.Label(self.roiview, text='Block distance')
        self.overlap_label.grid(row=4, column=1)
        self.overlap_slider = tk.Scale(self.roiview, from_=1, to=128,
                orient=tk.HORIZONTAL, resolution=1)
        self.overlap_slider.set(32)
        self.overlap_slider.grid(row=4, column=2, sticky='NSWE')
        
        self.distance_label = tk.Label(self.roiview, text='Line-block distance')
        self.distance_label.grid(row=5, column=1)
        self.distance_slider = tk.Scale(self.roiview, from_=1, to=128,
                orient=tk.HORIZONTAL, resolution=1)
        self.distance_slider.set(32)
        self.distance_slider.grid(row=5, column=2, sticky='NSWE')

        
        self.nroi_label = tk.Label(self.roiview, text='Count')
        self.nroi_label.grid(row=6, column=1)
        self.nroi_label.grid_remove()
        self.nroi_slider = tk.Scale(self.roiview, from_=1, to=128,
                orient=tk.HORIZONTAL, resolution=1)
        self.nroi_slider.grid(row=6, column=2, sticky='NSWE')

        self.radial_len_label = tk.Label(self.roiview, text='Radial line length')
        self.radial_len_label.grid(row=7, column=1)
        self.radial_len_label.grid_remove()
        self.radial_len_slider = tk.Scale(self.roiview, from_=1, to=1024,
                orient=tk.HORIZONTAL, resolution=1)
        self.radial_len_slider.grid(row=7, column=2, sticky='NSWE')

        # Set what sliders are needed for the default roi selection type
        self.update_roitype_selection(change_image=False)

        self.roi_buttons = ButtonsFrame(self.roiview, ['Update', 'Max grid', 'Clear', 'Undo', 'New group'],
                [self.update_grid, self.fill_grid, self.clear_selections, self.undo, self.new_group])

        self.roi_buttons.grid(row=8, column=1, columnspan=2)

        self.statusbar = None
        self.colors = None

    def connect_widgets(self, statusbar=None, colors=None, imageview=None):
        if statusbar is not None:
            self.statusbar = statusbar
        if colors is not None:
            self.colors = colors
        if imageview is not None:
            self.imageview = imageview

    def set_roi(self, x1=None,y1=None,x2=None,y2=None, params=None, user_made=True,
            recursion_data=None):
        '''
        Add (or "remove") a ROI based on user selection.

        Arguments
        ---------
        x1, y1, x2, y2 : None or int
        params : none or dict
        user_made : bool
            Is this an user made selection.
        recursion_data : None or something
            Internal use.
        '''

        if params is None:
            params = {}
            params['roitype'] = [s for s, b in self.roitype_selection.states.items() if b][0]
            params['blocksize'] = 2*[self.blocksize_slider.get()]
            params['distance'] = self.distance_slider.get()
            params['relstep'] = float(self.overlap_slider.get())/params['blocksize'][0]
            params['count'] = self.nroi_slider.get()
            params['rlen'] = self.radial_len_slider.get()
            params['i_roigroup'] = int(self.current_roi_group)
            params['mode'] = self.drawmode_selection.ticked[0]

       
        roitype, block_size, distance, rel_step, i_roigroup, count, mode, rlen = [
                params[key] for key in ['roitype','blocksize','distance','relstep', 'i_roigroup', 'count', 'mode', 'rlen']]

        if user_made:
            self.selections.append( (x1, y1, x2, y2, params) )   
        
        if roitype in ['polygon', 'arc_from_points', 'concentric_arcs_from_points',
                'radial_lines_from_points']:
            vertices = x1

            if roitype == 'polygon':
                rois = []
                for i_vertex in range(len(vertices)-1):
                    pA, pB = vertices[i_vertex:i_vertex+2]
                    rois.extend( grid_along_line(pA, pB, distance, block_size, step=rel_step) )
            elif roitype == 'arc_from_points':
                rois = grid_arc_from_points((0,0,*reversed(self.image_shape)), block_size, step=rel_step, points=vertices)
            elif roitype in ['concentric_arcs_from_points', 'radial_lines_from_points']:
                if recursion_data is None:
                    recursion_data = _workout_circle(vertices)
                
                if int(self.current_roi_group) < count-1:
                    self.current_roi_group += 1
                    cp, R = recursion_data
                    
                    if roitype == 'concentric_arcs_from_points':
                        new_recursion_data = (cp, R-distance)
                    elif roitype == 'radial_lines_from_points':
                        new_recursion_data = (cp, R)

                    self.set_roi(x1=x1,y1=y1,x2=x2,y2=y2,
                            params={**params, **{'i_roigroup': self.current_roi_group}},
                            user_made=False,
                            recursion_data=new_recursion_data)
                    self.current_roi_group -= 1 
                
                if roitype == 'concentric_arcs_from_points':
                    rois = grid_arc_from_points((0,0,*reversed(self.image_shape)), block_size, step=rel_step,
                            circle=recursion_data, lw=distance)
                elif roitype == 'radial_lines_from_points':
                    rois = grid_radial_line_from_points((0,0,*reversed(self.image_shape)), block_size, step=rel_step,
                            circle=recursion_data, line_len=rlen,
                            i_segment=self.current_roi_group, n_segments=count)

            else:
                raise ValueError('unkown roitype {}'.format(roitype))

        else:
            w = x2-x1
            h = y2-y1
       
            if roitype == 'line':
                rois = grid_along_line((x1, y1), (x2, y2), distance, block_size, step=rel_step)
            elif roitype == 'ellipse':
                rois = grid_along_ellipse((x1,y1,w,h), block_size, step=rel_step)
            elif roitype == 'free':
                # Single user-made box similar to roimarker
                rois = [[int(x1), int(y1), int(w), int(h)]]
            else:
                rois = gen_grid((x1,y1,w,h), block_size, step=rel_step)
            
        while len(self.roi_groups) <= i_roigroup:
            self.roi_groups.append([])

        if mode == 'add':
            self.roi_groups[i_roigroup].extend(rois)
     
            # Draw ROIs

            if len(rois) < 3000:
                self.set_status('Plotting all ROIs...')
            else:
                self.set_status('Too many ROIs, plotting only 3 000 first...')
            
            fig, ax = self.imview.images_plotter.get_figax()
            
            color = self.colors.colors.to_rgba(i_roigroup%self.colors.colors.get_clim()[1])
            
            patches = [] 
            lw = self.imview.patch_lw_slider.get()
            fill = self.imview.patch_fill_slider.get()/100
            fcolor = (color[0], color[1], color[2], color[3]*fill)
            for roi in rois[:3000]:

                patch = matplotlib.patches.Rectangle((float(roi[0]), float(roi[1])),
                        float(roi[2]), float(roi[3]), fill=True, edgecolor=color, facecolor=fcolor,
                        lw=lw)
                patches.append(patch)
                ax.add_patch(patch)
            
            self.roi_patches.append(patches)

        elif mode == 'remove':
            
            def _overlaps(a, b):
                return not (a[0]+a[2] < b[0] or b[0]+b[2] < a[0] or a[1]+a[3] < b[1] or b[1]+b[3] < a[1])

            for i_rgroup in range(len(self.roi_groups)) :
                
                # Remove ROIs
                remove_indices = []
                for i_old, old_roi in enumerate(self.roi_groups[i_rgroup]):    
                    for new_roi in rois:
                        if _overlaps(old_roi, new_roi):
                            remove_indices.append(i_old)
                            break
                
                print('removing {} in rg {}'.format(remove_indices, i_rgroup))

                for i_rm in remove_indices[::-1]:
                    self.roi_groups[i_rgroup].pop()

                    #self.roi_patches[i_rgroup].pop()
                
            # Remove patches separetly
            # Potential optimization if needed: Not sure if this is faster or
            #    slower than the own _overlaps
            # Anyway quite risky if rois and patches become unsynced
            #    (should be made in one-to-one correspondence)
            new_bboxes = [matplotlib.transforms.Bbox([[x, y],[x+w,y+h]]) for x,y,w,h in rois]
            for patches, selections in zip(self.roi_patches, self.selections):
                
                remove_indices = []
                for i_patch, patch in enumerate(patches):
                    if patch.get_bbox().count_overlaps(new_bboxes):
                        patch.remove()
                        remove_indices.append(i_patch)

                for i_rm in remove_indices[::-1]:
                    patches.pop(i_rm)


        else:
            raise ValueError('unkown mode {}'.format(mode))
        self.imview.images_plotter.update()
        self.set_status('ROIs plotted :)')
    

    def update_roitype_selection(self, change_image=True):
        '''
        When user selects a certain ROI type (box, circle, ...) to draw
        some of the sliders can be hidden.

        Arguments
        ---------
        change_image : bool
            If true, calls the change_image method at the end
        '''
        selected = self.roitype_selection.ticked[0] 

        elements = [
                [self.blocksize_slider, self.blocksize_label],
                [self.overlap_slider, self.overlap_label],
                [self.distance_slider, self.distance_label],
                [self.nroi_slider, self.nroi_label],
                [self.radial_len_slider, self.radial_len_label],
                ] 
        
        # Multiboxes all but the free selection
        multiboxes = self.roitypes.copy()
        multiboxes.remove('free')
    
        needs = [
                multiboxes,
                multiboxes,
                multiboxes,
                ['concentric_arcs_from_points', 'radial_lines_from_points'],
                ['radial_lines_from_points'],
                ]
        
        for needing, (slider, label) in zip(needs, elements):
            if selected in needing:
                slider.grid()
                label.grid()
            else:
                slider.grid_remove()
                label.grid_remove()
        
        if change_image:
            self.change_image()

    def update_grid(self, *args):

        # Updating the image also needed now to update the selector
        # type drawn while selecting (box or line)
        self.imageview.change_image()
        
        # Clear any previous patches
        for group in self.roi_patches:
            for patch in group:
                patch.remove()
        self.roi_patches = []

        self.roi_groups = []
        
        if self.selections:
            for selection in self.selections:
                self.set_roi(*selection, user_made=False)
        else:
            self.imageview.images_plotter.update()

    def new_group(self):
        '''
        Advance to the next ROI group.
        '''
        self.current_roi_group += 1


    def fill_grid(self):
        '''
        Create a selection spanning the whole image and distribute
        cross-correlation windows everywhere.
        '''
        self.set_roi(0,0,*reversed(self.image_shape))


    def clear_selections(self):
        '''
        Clear current user selections and ROIs (fresh start)
        '''
        self.selections = []
        self.update_grid()

        self.roi_groups = []
        self.current_roi_group = 0

    
    def undo(self):
        '''
        Undo a ROI selection made by the user.
        '''
        if len(self.selections) == 0:
            self.set_status('Nothing to undo')
            return None

        # Index of the roigroup to be undone
        i_roigroup = self.selections[-1][-1]['i_roigroup']

        # Clear the previous selection data
        self.selections = self.selections[:-1]
        
        # Clear the corresponding ROI patches
        N_rois_remove = len(self.roi_patches[-1])
        for patch in self.roi_patches[-1]:
            patch.remove()
        self.roi_patches = self.roi_patches[:-1]
        
        # Clear the actual ROIs
        self.roi_groups[i_roigroup] = self.roi_groups[i_roigroup][:-N_rois_remove]
        
        self.imview.images_plotter.update()
        self.set_status('Undone windows {} in ROI group {}'.format(N_rois_remove, i_roigroup))
    

    def set_status(self, *args, **kwargs):
        if self.statusbar:
            self.statusbar.set_status(*args, **kwargs)





class StatusBar(tk.Frame):

    def __init__(self, tk_parent):
        tk.Frame.__init__(self, tk_parent)
        self.parent = tk_parent
        self.status = tk.Label(self, text='Nothing to do')

    def set_status(self, text):
        '''
        Shows info text at the window bottom.
        '''
        self.status.config(text=text)
        self.status.update_idletasks()
 

class MovemeterAnalyser(tk.Frame):
    
    def __init__(self, tk_parent, movemeter, settings_tab):
        super().__init__(tk_parent)
        self.parent = tk_parent
        
        self.movemeter = movemeter

        self.calculate_button = tk.Button(self, text='Movement',
                command=self.measure_movement)
        self.calculate_button.grid(row=1, column=1)
        
        self.brightness_do_button = tk.Button(self, text='Brightness',
                command=self.measure_brightness)
        self.brightness_do_button.grid(row=1, column=2)

        self.stop_button = tk.Button(self, text='Stop',
                command=self.stop)
        self.stop_button.grid(row=1, column=3)


       
        # Movement parameters
        self.parview = settings_tab
        self.parview.columnconfigure(1, weight=1)
        self.movemeter_settings = MovemeterSettings(self.parview)
        self.movemeter_settings.grid(column=1,sticky='NSWE')

 
        # Brightness parameters
        self.brightness_view = settings_tab
        self.brightness_view.columnconfigure(1, weight=1)
        
        self.brightness_tickboxes = {}
        for name, options in self.movemeter.measure_brightness_opt.items():
            frame = TickboxFrame(
                    self.brightness_view, options, single_select=True)
            frame.grid()
            self.brightness_tickboxes[name] = frame

        
        self.datainput = None
        self.roidrawer = None
        self.plotter = None
        self.statusbar = None
        self.settings = None


    def connect_widgets(self, datainput=None, roidrawer=None,
                       plotter=None, statusbar=None):
        if datainput is not None:
            self.datainput = datainput
        if roidrawer is not None:
            self.roidrawer = roidrawer
        if plotter is not None:
            self.plotter = plotter
        if statusbar is not None:
            self.statusbar = statusbar

    def stop():
        '''
        Stop any ongoing motion analysis.
        '''
        self.exit=True
        if self.movemeter:
            self.movemeter.stop()

    
    def measure_movement(self, target=None):
        '''
        Run motion analysis for the images in the currently selected
        directory, using the drawn ROIs.
        '''

        if not self.datainput or not self.roidrawer:
            return # Nothing to do
        
        if target is None:
            target = lambda: self.movemeter.measure_movement(0, optimized=True)

        if self.datainput.image_fns and self.roidrawer.roi_groups:
            print('Started roi measurements')
           
            self.results = []
            
            self.movemeter = Movemeter(print_callback=self.set_status,
                    **self.movemeter_settings.get_current())
           
            for rois in self.roidrawer.roi_groups:
                # Set movemeted data
                images = [self.datainput._included_image_fns()]
                self.movemeter.set_data(images, [rois])
                
                self.results.append( target() )
            
            self.plotter.plot_results(self.results)

            # FIXME heatmap plotting
            #self.plotter.calculate_heatmap()
            #self.plotter.change_heatmap(1)

        else:
            self.set_status('No images or ROIs selected')
    

    def measure_brightness(self):
        kwargs = {}
        for name, frame in self.brightness_tickboxes.items():
            kwargs[name] = frame.ticked[0]

        bmes = lambda: self.movemeter.measure_brightness(0, **kwargs)
        self.measure_movement(target=bmes)

    
    def set_status(self, *args, **kwargs):
        if self.statusbar:
            self.statusbar.set_status(*args, **kwargs)




class ResultsPlotter(tk.Frame):

    def __init__(self, tk_parent):
        tk.Frame.__init__(self, tk_parent)
        self.parent = tk_parent
 
        # Results view: Analysed traces
        # ------------------------------------
        
        self.tabs = Tabs(self, ['Displacement', 'Heatmap'])
        self.tabs.grid(row=0, column=0, sticky='NSWE')
        self.resview = self.tabs.pages[0]
        self.heatview = self.tabs.pages[1]

        self.resview.rowconfigure(2, weight=1)
        self.resview.columnconfigure(1, weight=1)
        self.heatview.columnconfigure(2, weight=1)
        self.heatview.rowconfigure(2, weight=1)

        self.results_plotter = CanvasPlotter(self.resview)
        self.results_plotter.grid(row=2, column=1, sticky='NSWE')
       
        # Results show options
        self.results_plotter_opts = TickboxFrame(
                self.resview,
                ['show_individual', 'show_mean', 'show_toolbar'],
                defaults=[True,True,False],
                callback=self.plot_results)
        self.results_plotter_opts.grid(row=1, column=1, sticky='NSWE')

        self.heatmap_plotter = CanvasPlotter(self.heatview)
        self.heatmap_plotter.grid(row=2, column=2, sticky='NSWE') 
        
        self.heatmap_slider = tk.Scale(self.heatview, from_=0, to=0,
            orient=tk.HORIZONTAL, command=self.change_heatmap)
        self.heatmap_slider.grid(row=0, column=1, sticky='NSWE')
        
        self.heatmapcap_slider = tk.Scale(self.heatview, from_=0.1, to=100,
            orient=tk.HORIZONTAL, resolution=0.1, command=self.change_heatmap)
        self.heatmapcap_slider.set(20)
        self.heatmapcap_slider.grid(row=0, column=2, sticky='NSWE') 
        
        self.heatmap_firstcap_slider = tk.Scale(self.heatview, from_=0.1, to=100,
            orient=tk.HORIZONTAL, resolution=0.1, command=self.change_heatmap)
        self.heatmap_firstcap_slider.set(20)
        self.heatmap_firstcap_slider.grid(row=1, column=2, sticky='NSWE') 
        
        self.colors = None

    def connect_widgets(self, colors=None):
        if colors is not None:
            self.colors = colors


    def calculate_heatmap(self):
        '''
        Produce minimum size heatmap.
        '''
        self.heatmap_images = []
        
        # FIXME Heatmap for ROI groups not implemented properly
        # Currently just take the first nonempty ROI group
        i_roigroup = [i for i, rois in enumerate(self.roi_groups) if len(rois) != 0]
        if not i_roigroup:
            return None
        else:
            i_roigroup = i_roigroup[0]
        rois = self.roi_groups[i_roigroup]
        results = self.results[i_roigroup]

        roi_w, roi_h = rois[0][2:]

        roi_max_x = np.max([z[0] for z in rois])
        roi_min_x = np.min([z[0] for z in rois])
        roi_max_y = np.max([z[1] for z in rois])
        roi_min_y = np.min([z[1] for z in rois])
        
        step = int(self.overlap_slider.get())
        
        max_movement = float(self.movemeter_settings.maxmovement_slider.get())

        N = self._len_included_frames()

        for i_frame in range(N):
            image = np.zeros( (int((roi_max_y-roi_min_y)/step)+1, int((roi_max_x-roi_min_x)/step)+1) )
            for ROI, (x,y) in zip(rois, results):
                values = (np.sqrt(np.array(x)**2+np.array(y)**2))
                
                value = values[i_frame]
               
                cx = int((ROI[0]-roi_min_x)/step)
                cy = int((ROI[1]-roi_min_y)/step)
                
                try:
                    image[cy, cx] = value
                except:
                    print(image.shape)
                    print('cx {} cy {}'.format(cx, cy))
                    raise ValueError
            if np.max(image) < 0.01:
                image[0,0] = 1
            self.heatmap_images.append(image)

        self.heatmap_slider.config(from_=1, to=len(self.heatmap_images))
        self.heatmap_slider.set(1) 
        
        maxcapval = np.max(self.heatmap_images)
        self.heatmapcap_slider.config(from_=0, to=maxcapval)
        self.heatmapcap_slider.set(maxcapval)


    def change_heatmap(self, slider_value=None, only_return_image=False):
        '''
        When moving the slider to select the heatmap frame to show.
        '''
        #if slider_value == None:
        slider_value = int(self.heatmap_slider.get())

        i_image = int(slider_value) - 1
        image = np.copy(self.heatmap_images[i_image])
        
        # Total max value cap
        allframemax = np.max(self.heatmap_images, axis=0)
        image[allframemax > float(self.heatmapcap_slider.get())] = 0
        
        # First value max cap
        firstframemax = np.max(self.heatmap_images[0:3], axis=0)
        #image[firstframemax > float(self.heatmap_firstcap_slider.get())] = 0
        
        #image = image / float(self.heatmapcap_slider.get())
        #image[np.isnan(image)] = 0
        image = image / np.max(image)
        if np.isnan(image).any():
            image = np.ones(image.shape)
            image[0][0] = 0

        if only_return_image:
            return image
        else:
            self.heatmap_plotter.imshow(image, normalize=False)
 
    @staticmethod
    def get_displacements(results):
        '''
        Returns the directionless mangitude of the motion (displacement).
        '''
        return [np.sqrt(np.array(x)**2+np.array(y)**2) for x,y in results]


    @staticmethod
    def get_destructive_displacement_mean(results):
        '''
        Takes first the mean of the x and y components separately, and then
        calculates the directionless magnitude (displacement).

        This way the "random walk" does not pollute the mean so much as when
        taking the mean of the directionless magnitudes.
        '''
        x = [x for x,y in results]
        y = [y for x,y in results]
        return np.sqrt(np.mean(x, axis=0)**2 + np.mean(y, axis=0)**2)


    def plot_results(self, results):
        '''
        Plots (time, displacement).
        '''

        self.results_plotter.set_toolbar_visibility(
                'show_toolbar' in self.results_plotter_opts.ticked)

        self.results_plotter.ax.clear()

        for i_roi_group, result in enumerate(results):
            color = self.colors.colors.to_rgba(i_roi_group%self.colors.colors.get_clim()[1])
            displacements = [np.sqrt(np.array(x)**2+np.array(y)**2) for x,y in result]
            
            if 'show_individual' in self.results_plotter_opts.ticked:
                N_toplot = max( len(displacements), 50 )
                for d in displacements[0:N_toplot]:
                    self.results_plotter.plot(d, ax_clear=False, color=color, lw=0.5)
            
            if 'show_mean' in self.results_plotter_opts.ticked:
                self.results_plotter.plot(self.get_destructive_displacement_mean(result), ax_clear=False, color=color, lw=2)



class ResultsExporter(tk.Frame):   
    def __init__(self, tk_parent):
        tk.Frame.__init__(self, tk_parent)
        self.parent = tk_parent
 
        self.export_button = tk.Button(self, text='Export results',
                command=self.export_results)
        self.export_button.grid(row=2, column=1)
        
        self.export_name = tk.Entry(self, width=50)
        self.export_name.insert(0, "enter export name")
        self.export_name.grid(row=2, column=2)
 
        self.colors = None


    def connect_widgets(self, colors=None):
        if colors is not None:
            self.colors = colors

    def _save_movzip(self, fn=None, only=None):
        '''
        Saves a movzip containg data/settings about the ran motion analysis.
        
        Arguments
        ---------
        fn : string or None
            If None, ask the filename.
        only : bool, string or list of strings
            Select to save only certain parts.
            Possible values are 'metadata', 'image_filenames', 'selections',
            'rois', 'movements' or any list combinations of these.
        '''

        if isinstance(only, str):
            only = [only]

        if fn is None:
            if only:
                title = 'Save '+','.join(only)
            else:
                title = 'Save movzip'
            fn = filedialog.asksaveasfilename(parent=self, title=title,
                    initialdir=MOVEDIR)
            
            if not fn.endswith('.zip'):
                fn = fn+'.zip'

        # Dump GUI settings
        settings = {}
        settings['block_size'] = self.blocksize_slider.get()
        settings['block_distance'] = self.overlap_slider.get()
        settings['movemeter_settings'] = self.movemeter_settings.get_current()
        settings['export_time'] = str(datetime.datetime.now())
        settings['movemeter_version'] = __version__
        settings['exclude_images'] = self.exclude_images

        if self.images:
            settings['images_shape'] = self.image_shape
        
        
        movzip = {'metadata': settings,
                'image_filenames': self._included_image_fns(),
                'selections': self.selections,
                'rois': self.roi_groups,
                'movements': self.results}

        self.set_status('Saving movzip...')
        
        with zipfile.ZipFile(fn, 'w') as savezip:
            for pfn, obj in movzip.items():

                if only and pfn not in only:
                    continue

                with savezip.open(pfn+'.json', 'w') as fp:
                    fp.write(json.dumps(obj).encode('utf-8'))
        
        self.set_status('Mozip saved.')
        



    def export_results(self, batch_name=None):
        '''
        Creates a folder containing motion analysis results
        - movzip
        - csv files
        - images
        '''
        savename = self.export_name.get()
        zipsavename = savename

        save_root = MOVEDIR
        if batch_name is not None:
            save_root = os.path.join(save_root, 'batch', batch_name)
        
        save_directory = os.path.join(save_root, savename)
        os.makedirs(save_directory, exist_ok=True)
    
        self._save_movzip(os.path.join(save_directory, 'movemeter_{}.zip'.format(zipsavename)))
        
        means = []

        for i_roigroup, results in enumerate(self.results):
            fn = os.path.join(save_directory, 'movements_{}_rg{}.csv'.format(zipsavename, i_roigroup))
            
            displacements = self.get_displacements(results)
            
            if not displacements:
                continue

            dm_displacement = self.get_destructive_displacement_mean(results)

            with open(fn, 'w') as fp:
                writer = csv.writer(fp, delimiter=',')
                
                writer.writerow(['time (s)', 'mean displacement (pixels)', 'destructive mean displacement (pixels)'] + ['ROI{} displacement (pixels)'.format(k) for k in range(len(displacements))])

                for i in range(len(displacements[0])):
                    row = [displacements[j][i] for j in range(len(displacements))]
                    row.insert(0, dm_displacement[i])
                    row.insert(0, np.mean(row))
                    row.insert(0, i/self.fs)
                    writer.writerow(row)

                if i_roigroup == 0:
                    N = len(dm_displacement)
                    means.append(np.linspace(0, (N-1)/self.fs, N))
                means.append(dm_displacement)

        with open(os.path.join(save_directory, 'summary_desctructive_{}.csv'.format(zipsavename)), 'w') as fp:
            writer = csv.writer(fp,  delimiter=',')

            writer.writerow(['time (s)'] +['roi group {} (pixels)'.format(i) for i in range(len(means)-1)])

            for i in range(len(means[0])):
                row = [m[i] for m in means]
                writer.writerow(row)


        slider_i = int(self.image_slider.get())
        self.image_slider.set(int(len(self._included_image_fns()))/2)

        # Image of the ROIs
        self.set_status('Saving the image view')
        fig, ax = self.images_plotter.get_figax()
        fig.savefig(os.path.join(save_directory, 'movemeter_imageview.jpg'), dpi=400, pil_kwargs={'optimize': True})
        
        self.image_slider.set(slider_i)
        
        # Image of the result traces
        self.set_status('Saving the results view')
        fig, ax = self.results_plotter.get_figax()
        fig.savefig(os.path.join(save_directory, 'movemeter_resultsview.jpg'), dpi=400, pil_kwargs={'optimize': True})


        def save_heatmaps(heatmaps, image_fns, savedir):
            
            for fn, image in zip(image_fns, heatmaps):
                tifffile.imsave(os.path.join(savedir, 'ht_{}'.format(os.path.basename(fn))), image.astype('float32'))
            
            # Save mean heatmap image with scale bar using matplotlib
            # FIXME Expose option for how many last images to save the mean for
            meanimage = np.mean(heatmaps[-min(5, len(heatmaps)):], axis=0)
            
            if False:
                # This was used to clip heatmap values
                # FIXME Expose option in the GUI
                if 'musca' in save_directory:
                    meanimage = np.clip(meanimage, 0, 50)
                    if np.max(meanimage) < 50:
                        meanimage[0,0] = 50
                else:
                    meanimage = np.clip(meanimage, 0, 6)
                    if np.max(meanimage) < 6:
                        meanimage[0,0] = 6

            fig, ax = plt.subplots()
            imshow = ax.imshow(meanimage)
            ax.set_axis_off()

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(imshow, cax=cax)
            
            fig.savefig(os.path.join(savedir, 'ht_mean.png'), dpi=800)
            
            plt.show(block=False)
            plt.pause(0.01)
            plt.close(fig)
        
        self.set_status('Saving heatmaps')
        subsavedir = os.path.join(save_directory, 'heatmap_tif')
        os.makedirs(subsavedir, exist_ok=True)
         
        save_heatmaps(self.heatmap_images, self.image_fns, subsavedir)
        
        self.set_status('DONE Saving :)')
     

class MovemeterTkGui(tk.Frame):
    '''Main widget for the Movemeter tkinter GUI.
    
    ATTRIBUTES
    -----------
    self.parent : object
        tkinter parent widget


    exclude_images : list
        List of image filenames or indices to exclude from the analysis.

   
   
    selections : list
    
    '''

    def __init__(self, tk_parent):
        tk.Frame.__init__(self, tk_parent)
        self.parent = tk_parent

        # Motion analysis
        self.movemeter = Movemeter()
        self.results = []
        self.heatmap_images = []
        self.batch_name = 'batch_name'


        self.statusbar = StatusBar(self)
        self.statusbar.grid(row=3, column=0, columnspan=2)


        self.folview = DataInputWidget(
                self, self.movemeter, self.statusbar)
        self.folview.grid(row=0, column=0, sticky='NSWE')
        
        self.imview = ImageROIWidget(self)
        self.imview.grid(row=1, column=0, sticky='NSWE')

        self.colors = ColorsWidget(self.imview)
        self.colors.grid(row=0, column=1)


        # Operations view
        # -------------------------
        self.opview = tk.LabelFrame(self, text='Command center')
        self.opview.grid(row=0, column=1, sticky='NSWE')
        
        self.tabs = Tabs(self.opview,
                ['ROI creation', 'Analyser settings'],
                draw_frame = True)
        self.tabs.grid(row=0, column=0, sticky='NSWE')
        self.tabs.set_page(1) 
       
        self.roidrawer = BoxRoiDrawer(self.imview, self.tabs.tabs[0])
        self.roiview = self.tabs.tabs[0]
        self.imview.connect_widgets(roidrawer = self.roidrawer)


        self.actframe = MovemeterAnalyser(self.opview, self.movemeter, self.tabs.tabs[1])
        self.actframe.grid(row=1, column=0)
        
        self.results = ResultsPlotter(self)
        self.results.grid(row=1, column=1)

        self.columnconfigure(1, weight=1)    
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)

        # Connect widgets
        self.colors.connect_widgets(
                roidrawer = self.roidrawer,
                )

        self.imview.connect_widgets(
                datainput = self.folview,
                colors = self.colors,
                )
        
        self.roidrawer.connect_widgets(
                colors = self.colors,
                imageview = self.imview,
                )

        self.actframe.connect_widgets(
                datainput = self.folview,
                roidrawer = self.roidrawer,
                plotter = self.results,
                statusbar = self.statusbar)
        
        self.results.connect_widgets(
                colors = self.colors,
                )


    def open_settings(self):
        '''
        Placeholder for the settings dialog.
        '''
        raise NotImplementedError


    def recalculate_old(self, directory=None):
        '''
        Load old movzip, look the ROI extremes, and draw a new ROI
        but using the current block settings (block size and distance).

        Useful for testing how the results change when the selected
        area remains approximately the same but the block settings change.
        '''

        if directory == None:
            directory = filedialog.askdirectory()
            if not directory:
                return None
        
        if not self._ask_batchname():
            return None
 
        self.exit = False
        for root, dirs, fns in os.walk(directory):
            
            if self.exit:
                break

            movzip = [fn for fn in os.listdir(root) if fn.startswith('movemeter') and fn.endswith('.zip')]
            
            if movzip:
                settings, filenames, selections, rois, movements = self._load_movzip(os.path.join(root, movzip[0]))
                
                self.folder_selected(
                        os.path.dirname(filenames[0]), usermade=False)
                
                x1, y1 = np.min(rois, axis=0)[0:2]
                x2, y2 = np.max(rois, axis=0)[0:2] + rois[0][3]
                self.set_roi(x1,y1,x2,y2)

                self.measure_movement()

                self.export_results(batch_name=self.batch_name)

        self.set_status('Results recalculated :)')


    def replot_heatmap(self, directory=None):
        '''
        Like recalculate old, but relies in the old movement analysis results
        '''
        if directory == None:
            directory = filedialog.askdirectory()
            if not directory:
                return None
        
        if not self._ask_batchname():
            return None
 
        self.exit = False
        for root, dirs, fns in os.walk(directory):
            
            if self.exit:
                break

            movzip = [fn for fn in os.listdir(root) if fn.startswith('movemeter') and fn.endswith('.zip')]
            if movzip:
                settings, filenames, self.selections, self.roi_groups, self.results = self._load_movzip(os.path.join(root, movzip[0])) 
                
                self.folder_selected(
                        os.path.dirname(filenames[0]), usermade=False)
                self.set_settings(settings)

                self.plot_results()
                self.calculate_heatmap()
                self.change_heatmap(1)

                self.export_results(batch_name=self.batch_name)

        self.set_status('Heatmaps replotted :)')

    
    def _ask_batchname(self):
        name = simpledialog.askstring('Batch name', 'Name new folder')
        if name:
            self.batch_name = name
            return True
        else:
            return False


    def batch_process(self, fill_maxgrid=False):
        '''
        fill_maxgrid : bool
            If True, ignore current ROIs and fill a full frame grid
            using the current slider options.
        '''

        if not self._ask_batchname():
            return None
        
        self.exit = False
        for folder in self.folders:
            if self.exit:
                break
            self.folder_selected(folder, usermade=False)
            
            if fill_maxgrid:
                self.fill_grid()
            
            self.measure_movement()
            self.export_results(batch_name=self.batch_name)


 


  

    def set_settings(self, settings):
        '''
        Apply the given settings.

        Arguments
        ----------
        settings : dict
            A dictionary of settings.
        '''
        for key, value in settings.items():
            if key == 'block_size':
                self.blocksize_slider.set(value)
            elif key == 'block_distance':
                self.overlap_slider.set(value)
            elif key == 'maximum_movement':
                self.movemeter_settings.maxmovement_slider.set(value)
            elif key == 'upscale':
                self.movemeter_settings.upscale_slider.set(value)
            elif key == 'cpu_cores':
                self.movemeter_settings.cores_slider.set(value)
            elif key == 'exclude_images':
                self.exclude_images = value
            elif key == 'measurement_parameters':
                self.movemeter_settings.tickboxes.states = value


   

    def apply_movzip(self, fn=None, rois=False):
        '''
        Load parts of a movzip and apply settings from it
        to the current session.
        '''
        if fn is None:
            fn = filedialog.askopenfilename(parent=self, title='Select a movzip',
                    initialdir=MOVEDIR)

        settings, filenames, selections, roi_groups, movements = self._load_movzip(fn)
        
        if rois:
            self.selections = selections
            self.rois_groups = roi_groups
            self.update_grid()


    def _load_movzip(self, fn):
        '''
        Load a movzip, returning its contents.

        Returns
        -------
        settings, image_filenames, selections, rois, movements
        '''

        movzip = []

        with zipfile.ZipFile(fn, 'r') as loadzip:
            
            for pfn in ['metadata', 'image_filenames', 'selections', 'rois', 'movements']:
                try:
                    with loadzip.open(pfn+'.json', 'r') as fp:
                        movzip.append( json.loads(fp.read()) )
        
                except KeyError:
                    movzip.append(None)

        return (*movzip,)

    
    def save_roiview(self, only_rois=False):
        '''
        Save the current image view with ROIs.

        Arguments
        ---------
        only_rois : bool
            If True, hide the image and show ROIs in the
            saved image.
        '''
        savefn = filedialog.asksaveasfilename()
        if savefn:
            fig = self.images_plotter.figure
            
            if only_rois:
                self.images_plotter.imshow_obj.set_visible(False)
            
            fig.savefig(savefn, dpi=600, transparent=only_rois)
            
            if only_rois:
                self.images_plotter.imshow_obj.set_visible(True)


     

def main():
    '''
    Initialize tkinter and start the Movemeter GUI.
    '''
    root = tk.Tk()
    root.title('Movemeter - Tkinter GUI - {}'.format(__version__))
    gui = MovemeterTkGui(root)
    gui.grid(row=1, column=1, sticky='NSWE')
    root.columnconfigure(1, weight=1)
    root.rowconfigure(1, weight=1)

    menu = MovemeterMenubar(gui)

    root.mainloop()


if __name__ == "__main__":
    main()
