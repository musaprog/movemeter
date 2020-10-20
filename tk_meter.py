'''
A tkinter GUI for Movemeter.
'''

import os
import json

import numpy as np
import tifffile
import tkinter as tk
from tkinter import filedialog
import matplotlib.patches
from PIL import Image

from tk_steroids.elements import Listbox
from tk_steroids.matplotlib import CanvasPlotter

from movemeter import gen_grid
from movemeter import Movemeter


class MovemeterTkGui(tk.Frame):
    '''
    Class documentation TODO.
    '''

    def __init__(self, tk_parent):
        tk.Frame.__init__(self, tk_parent)
        self.parent = tk_parent

        self.folders = []
        self.image_fns = []
        self.images = None
        self.exclude_images = []

        self.mask_image = None
        self.rois = []
        self.roi_patches = []
        self.results = []
        
        self.heatmap_images = []


        # Top menu
        # --------------------------------
        self.menu = tk.Menu(self)
        
        filemenu = tk.Menu(self)
        filemenu.add_command(label='Open directory', command=self.open_directory)
        self.menu.add_cascade(label='File', menu=filemenu)

        self.parent.config(menu=self.menu)

        # Input folders

        self.folview = tk.LabelFrame(self, text='Input folders')
        self.folview.rowconfigure(1, weight=1)
        self.folview.columnconfigure(1, weight=1)
        self.folview.grid(row=0, column=1, sticky='NSWE')

        self.folders_listbox = Listbox(self.folview, ['No folders selected'], self.folder_selected)
        self.folders_listbox.listbox.config(height=10)
        self.folders_listbox.grid(row=1, column=1, sticky='NSWE')

        # Operations view
        # -------------------------
        self.opview = tk.LabelFrame(self, text='Command center')
        self.opview.grid(row=0, column=2, sticky='NSWE')
        
        self.blocksize_slider = tk.Scale(self.opview, from_=16, to=512,
                orient=tk.HORIZONTAL)
        self.blocksize_slider.grid(row=2, column=1, sticky='NSWE')

        self.maxmovement_slider = tk.Scale(self.opview, from_=1, to=100,
                orient=tk.HORIZONTAL)
        self.maxmovement_slider.set(10)
        self.maxmovement_slider.grid(row=2, column=2, sticky='NSWE')

        self.overlap_slider = tk.Scale(self.opview, from_=0.1, to=2,
                orient=tk.HORIZONTAL, resolution=0.1)
        self.overlap_slider.set(1)
        self.overlap_slider.grid(row=2, column=3, sticky='NSWE')




        self.calculate_button = tk.Button(self.opview, text='Measure movement',
                command=self.measure_movement)
        self.calculate_button.grid(row=1, column=1)


        self.export_button = tk.Button(self.opview, text='Export results',
                command=self.export_results)
        self.export_button.grid(row=3, column=1)
        
        self.export_name = tk.Entry(self.opview, width=50)
        self.export_name.insert(0, "enter export name")
        self.export_name.grid(row=3, column=2)


        # Images view: Image looking and ROI selection
        # -------------------------------------------------
        self.imview = tk.LabelFrame(self, text='Images and ROI')
        self.imview.grid(row=1, column=1)
        
        self.toggle_exclude_button = tk.Button(self.imview, text='Exclude',
                command=self.toggle_exclude)
        self.toggle_exclude_button.grid(row=1, column=1)

        self.image_slider = tk.Scale(self.imview, from_=0, to=0,
                orient=tk.HORIZONTAL, command=self.change_image)
        
        self.image_slider.grid(row=2, column=1, sticky='NSWE')

        self.images_plotter = CanvasPlotter(self.imview)
        self.images_plotter.grid(row=3, column=1) 


        # Results view: Analysed traces
        # ------------------------------------
        self.resview = tk.LabelFrame(self, text='Results')
        self.resview.grid(row=1, column=2)
       
        self.results_plotter = CanvasPlotter(self.resview)
        self.results_plotter.grid(row=1, column=1) 
        
        self.heatmap_plotter = CanvasPlotter(self.resview)
        self.heatmap_plotter.grid(row=1, column=2) 
        
        self.heatmap_slider = tk.Scale(self.resview, from_=0, to=0,
            orient=tk.HORIZONTAL, command=self.change_heatmap)
        self.heatmap_slider.grid(row=0, column=1) 
        
        self.heatmapcap_slider = tk.Scale(self.resview, from_=0.1, to=100,
            orient=tk.HORIZONTAL, resolution=0.1)
        self.heatmapcap_slider.set(20)
        self.heatmapcap_slider.grid(row=0, column=2) 
       


    def folder_selected(self, folder):
        '''
        When the user selects a folder from the self.folders_listbox
        '''
        print('Selected folder {}'.format(folder))

        self.image_fns = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith('.tiff')]

        self.images = [None for fn in self.image_fns]
        self.mask_image = None

        self.change_image(slider_value=1)
        N_images = len(self.image_fns)
        self.image_slider.config(from_=1, to=N_images)
       
        self.export_name.delete(0, tk.END)
        self.export_name.insert(0, os.path.basename(folder.rstrip('/')))


    def toggle_exclude(self):
        fn = self.image_fns[int(self.image_slider.get())-1]
        if fn not in self.exclude_images:
            self.exclude_images.append(fn)
        else:
            self.exclude_images.pop(fn)
        
        self.mask_image = None
        self.change_image(slider_value=self.image_slider.get())
        print(self.exclude_images)


    def measure_movement(self):
        if self.image_fns and self.rois:
            print('Started roi measurements')
            self.movemeter = Movemeter(upscale=5)
            
            self.movemeter.subtract_previous = True
            self.movemeter.compare_to_first = False

            self.movemeter.set_data([[fn for fn in self.image_fns if fn not in self.exclude_images]], [self.rois])
            self.results = self.movemeter.measure_movement(0, max_movement=int(self.maxmovement_slider.get()))
            self.plot_results()

            self.calculate_heatmap()
            self.change_heatmap(1)

            print('Finished roi measurements')
        else:
            print('No rois')


    def set_roi(self, x1,y1,x2,y2):
        w = x2-x1
        h = y2-y1
        block_size = self.blocksize_slider.get()
        block_size = (block_size, block_size)
        self.rois = gen_grid((x1,y1,w,h), block_size, step=float(self.overlap_slider.get()))
        
        print(len(self.rois))

        fig, ax = self.images_plotter.get_figax()
        
        # Clear any previous patches
        for patch in self.roi_patches:
            patch.remove()
        self.roi_patches = []

        for roi in self.rois:
            patch = matplotlib.patches.Rectangle((float(roi[0]), float(roi[1])),
                    float(roi[2]), float(roi[3]), fill=True, color='red',
                    alpha=0.2)
            self.roi_patches.append(patch)

            ax.add_patch(patch)
        
        self.images_plotter.update()
        print('FInished plotting rois')
        

    def change_image(self, slider_value=None):
        
        image_i = int(slider_value) -1
       
        if not 0 <= image_i < len(self.image_fns):
            return None

        if self.mask_image is None:
            for i in range(len(self.images)):
                self.images[i] = tifffile.imread(self.image_fns[i])
            
            self.mask_image = np.inf * np.ones(self.images[0].shape)
            
            for image in self.images:
                self.mask_image = np.min([self.mask_image, image], axis=0)


        if self.images[image_i] is None:
            self.images[image_i] = tifffile.imread(self.image_fns[image_i])
        

        self.images_plotter.imshow(self.images[image_i]-self.mask_image, roi_callback=self.set_roi, cmap='gray')
        self.images_plotter.update()
   

    def plot_results(self):
        self.results_plotter.ax.clear()
        for x,y in self.results[:50]:
            self.results_plotter.plot(np.sqrt(np.array(x)**2+np.array(y)**2), ax_clear=False, color='red')


    def calculate_heatmap(self):

        self.heatmap_images = []
        
        for i_frame in range(len(self.image_fns)):
            if i_frame == 0:
                continue
            image = np.zeros(self.images[0].shape)
            for ROI, (x,y) in zip(self.rois, self.results):
                values = (np.sqrt(np.array(x)**2+np.array(y)**2))
                value = abs(values[i_frame] - values[i_frame-1])
                xx,yy,w,h = ROI
                step = float(self.overlap_slider.get())
                cx = xx+int(w/2)
                cy = yy+int(h/2)
                #image[yy:yy+h, xx:xx+w] = value
                image[cy-int(step*(h/2)):cy+int(step*(h/2)), cx-int(step*(w/2)):cx+int(step*(w/2))] = value
            
            if np.max(image) < 0.01:
                image[0,0] = 1
            self.heatmap_images.append(image)

        self.heatmap_slider.config(from_=1, to=len(self.heatmap_images))
        self.heatmap_slider.set(1) 


    def change_heatmap(self, slider_value=None):
        i_image = int(slider_value) -1
        image = self.heatmap_images[i_image]
        allframemax =np.max(self.heatmap_images, axis=0)
        image[allframemax > float(self.heatmapcap_slider.get())] = 0
        self.heatmap_plotter.imshow(image)


    def export_results(self):

        savename = self.export_name.get()

        save_root = 'exports'
        save_directory = os.path.join(save_root, savename)
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Dump exact used filenames
        with open(os.path.join(save_directory, 'image_filenames.json'), 'w') as fp:
            json.dump(self.image_fns, fp)

        # Dump ROIs
        with open(os.path.join(save_directory, 'rois.json'), 'w') as fp:
            json.dump(self.rois, fp)

        # Dump analysed movements
        with open(os.path.join(save_directory, 'movements.json'), 'w') as fp:
            json.dump(self.results, fp)

        # Image of the ROIs
        fig, ax = self.images_plotter.get_figax()
        fig.savefig(os.path.join(save_directory, 'image_view.jpg'), dpi=600, optimize=True)

        # Image of the result traces
        fig, ax = self.results_plotter.get_figax()
        fig.savefig(os.path.join(save_directory, 'results_view.jpg'), dpi=600, optimize=True)

        # Image of the result traces
        #fig, ax = self.heatmap_plotter.get_figax()
        #fig.savefig(os.path.join(save_directory, 'heatmap_view.jpg'), dpi=600, optimize=True)

        maxval = np.max(self.heatmap_images)
        heatmaps = [np.copy(image)/maxval for image in self.heatmap_images]
        
        # Save heatmap images
        #subsavedir = os.path.join(save_directory, 'heatmap_npy')
        #os.makedirs(subsavedir, exist_ok=True)
        #for fn, image in zip(self.image_fns, self.heatmap_images):
        #    np.save(os.path.join(subsavedir, 'heatmap_{}.npy'.format(os.path.basename(fn))), image)

        subsavedir = os.path.join(save_directory, 'heatmap_matplotlib')
        os.makedirs(subsavedir, exist_ok=True)
        for fn, image in zip(self.image_fns, heatmaps):
            self.heatmap_plotter.imshow(image, normalize=False)
            fig, ax = self.heatmap_plotter.get_figax()
            fig.savefig(os.path.join(subsavedir, 'heatmap_{}.jpg'.format(os.path.basename(fn))), dpi=300, optimize=False)

        # Save heatmap images
        #subsavedir = os.path.join(save_directory, 'heatmap_pillow')
        #os.makedirs(subsavedir, exist_ok=True)
       
        #for fn, image in zip(self.image_fns, heatmaps):
        #    pimage = Image.fromarray(image)
        #    pimage.save(os.path.join(subsavedir, 'heatmap_{}.png'.format(os.path.basename(fn))))

            
          
    def open_directory(self, directory=None):
        
        if directory is None:
            directory = filedialog.askdirectory(title='Select directory with the images')
        
        if directory:
            self.folders.append(directory)
            self.folders_listbox.set_selections(self.folders)
            self.folder_selected(directory)


def main():
    '''
    Initialize tkinter and place the Movemeter GUI
    on the window.
    '''
    root = tk.Tk()
    gui = MovemeterTkGui(root)
    gui.grid()
    root.mainloop()


if __name__ == "__main__":
    main()
