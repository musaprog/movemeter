
import unittest
import os

from movemeter import Movemeter

class MovemeterTest(unittest.TestCase):
    '''
    Creates movemeter with all possible backend combinations
    and sets data and measures movements and compares the
    movements to predefined results.
    '''

    def setUp(self):
        
        # Get images and ROIs
        images = [os.path.join('data', fn) for fn in os.listdir('data')
                        if fn.endswith('tiff')]

        ROIs = [[[130,500,133,114]]]
                
        
        # Test all combinations of the following cc and im backends
        test_cc_backends = ['OpenCV']
        test_im_backends = ['OpenCV', 'tifffile']
        
        self.movemeters = []

        for cc in test_cc_backends:
            for im in test_im_backends:
                movemeter = Movemeter(cc_backend=cc, imload_backend=im)               
                movemeter.set_data([images], ROIs)
                
                self.movemeters.append( movemeter )



    def test_measure_movement(self):

        for movemeter in self.movemeters:
            movements = movemeter.measure_movement(0)
            print(movements)
            self.assertEqual(movements, [[[0.0, 0.0, 0.0, -1.0, -1.0, 0.0], [0.0, 5.0, 9.0, -13.0, -16.0, -6.0]]], 'Different results in movement measurement')


