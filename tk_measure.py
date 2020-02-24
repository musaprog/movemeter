'''

'''
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from marker import Marker
from imalyser.movement import Movemeter


def load(fn):
    with open(fn, 'r') as fp:
        data = json.load(fp)
    return data

def plot(movement):
    fs = 50
    time = np.linspace(0, len(movement[0][0])/fs, len(movement[0][0]))
    for trace in movement:
        plt.plot(time, trace[0], label='x')
        plt.plot(time, trace[1], label='y')
    
    plt.legend()

    plt.show()



def plotOld(fn):
    
    movement = load(fn)
    plot(movement)    




def runSingle(folder=None, chdir=None):

    if not chdir is None:
        if not os.path.exists(chdir):
            os.makedirs(chdir, exist_ok=True)
        os.chdir(chdir)
	
    print('Input for movement measuring') 
    
    #folder = '/home/joni/smallbrains-nas1/array1/pseudopupil_imaging/DrosoMuscle2/in_darkness_50fps'
    
    if folder is None:
        folder = input('folder: ')
    #ending = input('File ending: ')
    ending = '.tiff'

    image_fns = [os.path.join(folder,fn) for fn in os.listdir(folder) if fn.endswith(ending)]
    image_fns.sort()
    #image_fns = image_fns[0:500]
    
    if len(image_fns) == 0:
        raise FileNotFoundError('Could not find any images')

    tag = os.path.split(folder)[1]

    fig, ax = plt.subplots()
    markings_savefn = 'tmp_crops_{}.json'.format(tag)

    marker = Marker(fig, ax, [image_fns[int(len(image_fns)/2)]], markings_savefn)
    marker.run()
    markings = list(marker.getMarkings().values())

    m = Movemeter(upscale=5)
    

    m.set_data([image_fns], markings)
    
    print('Starting movement measurements...')
    movement = m.measure_movement(0, max_movement=10)

    print(movement)
    
    print('Saving the results...')
    with open('tmp_movements_{}.json'.format(tag), 'w') as fp:
        json.dump(movement, fp)
    
    print('Starting plotting the results...')
    plot(movement)

if __name__ == "__main__":
    runSingle()

