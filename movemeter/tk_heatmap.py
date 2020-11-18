import os
import zipfile
import json

import numpy as np
import tkinter as tk
from tkinter import filedialog
from tk_steroids.matplotlib import CanvasPlotter

from movemeter import __version__
import movemeter.heatmap as ht


class HeatmapPlotter(tk.Frame):

    def __init__(self, tk_parent, standalone=False):
        tk.Frame.__init__(self, tk_parent)

        self.loaded_data = []
        self.heatmaps = []

        self.heatmap_plotter = CanvasPlotter(self)
        self.heatmap_plotter.grid(row=5, column=0, sticky='NSWE')

        self.image_slider = tk.Scale(self, from_=0, to=0,
                orient=tk.HORIZONTAL, command=self.update_image)
        self.image_slider.grid(row=4, column=0, sticky='NSWE')

        self.type_frame = tk.LabelFrame(text='Heatmap type')
        self.type_frame.grid(row=1, column=0, sticky='NSWE')
        
        self.map_types = ['speed', 'displacement', 'direction']
        
        self.selected_type = tk.StringVar()
        self.selected_type.set(self.map_types[0])

        for i_button, mtype in enumerate(self.map_types):

            button = tk.Radiobutton(self.type_frame, text=mtype,
                    command=self.update_heatmaps, variable=self.selected_type,
                    value=mtype)
            
            button.grid(column=i_button, row=1)

        self.save_button = tk.Button(text='Save', command=self.save_images)
        self.save_button.grid(row=6, column=0)

        if standalone:
            menu = tk.Menu(self)

            filemenu = tk.Menu(menu)
            filemenu.add_command(label='Load movemeter zip file...', command=self.select_file)
            menu.add_cascade(label='File', menu=filemenu)
            
            tk_parent.config(menu=menu)
        
        # Modify hetmaps showing sliders
        self.filter_frame = tk.LabelFrame(text='Filtering options')
        self.filter_frame.grid(row=3, column=0, sticky='NSWE')
        self.filter_frame.columnconfigure(2, weight=1)

        tk.Label(self.filter_frame, text='Upcap value').grid(row=1, column=1)
        self.upcap_slider = tk.Scale(self.filter_frame, from_=0, to=100,
                orient=tk.HORIZONTAL, command=self.update_image,
                resolution=0.2)
        self.upcap_slider.set(10)
        self.upcap_slider.grid(row=1, column=2, sticky='NSWE')



    def set_data(self, image_shape, rois, movements):
        self.image_shape = image_shape
        self.rois = rois
        self.movements = movements


    def select_file(self):
        
        files = filedialog.askopenfilenames()
        
        for fn in files:
            with zipfile.ZipFile(fn) as savezip:
                with savezip.open('rois.json') as fp:
                    rois = json.loads(fp.read().decode('utf-8'))
                with savezip.open('movements.json') as fp:
                    movements = json.loads(fp.read().decode('utf-8'))
                with savezip.open('metadata.json') as fp:
                    image_shape = json.loads(fp.read().decode('utf-8'))['images_shape']
            self.savedir = os.path.dirname(fn)
            break    
            

        self.rois = rois
        self.movements = movements
        self.image_shape = image_shape
        
        self.update_heatmaps()


    def update_heatmaps(self):
        mtype = self.selected_type.get()

        self.cmap = 'viridis'
        if mtype == 'speed':
            self.heatmaps = ht.speed_heatmaps(self.image_shape, self.rois, self.movements)
        elif mtype == 'displacement':
            self.heatmaps = ht.displacement_heatmaps(self.image_shape, self.rois, self.movements)
        elif mtype == 'direction':
            self.heatmaps = ht.displacement_direction_heatmaps(self.image_shape, self.rois, self.movements)
            self.cmap = 'hsv'
        self.image_slider.config(from_=1, to=len(self.heatmaps))
        self.image_slider.set(1)
        
        self.update_image()


    def update_image(self, slider_val=None):
        #if slider_val == None:
        #    slider_val = 1
        #i_image = int(slider_val) - 1
        
        i_image = int(self.image_slider.get()) - 1

        image = np.copy(self.heatmaps[i_image])
        #image = image / np.max(self.heatmaps)

        # Do the filtering
        allimagesmax = np.max(self.heatmaps, axis=0)
        image[allimagesmax > float(self.upcap_slider.get())] = 0
        
        
        image = image / float(self.upcap_slider.get())
        self.heatmap_plotter.imshow(image, normalize=False, cmap=self.cmap)
    
    def save_images(self):
        
        savedir = os.path.join(self.savedir, 'tk_heatmap_exports')
        
        os.makedirs(savedir, exist_ok=True)

        for i_image in range(len(self.heatmaps)):
            self.image_slider.set(i_image+1)
            self.update_image()

            fig, ax = self.heatmap_plotter.get_figax()
            fig.savefig(os.path.join(savedir, '{:03d}.jpg'.format(i_image)), dpi=600)


def main():
    
    root = tk.Tk()
    root.title('Heatmap - Tkinter GUI - {}'.format(__version__))
    gui = HeatmapPlotter(root, standalone=True)
    gui.grid()
    root.mainloop()

if __name__ == "__main__":
    main()

