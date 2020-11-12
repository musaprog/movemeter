
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
            self.assertEqual(movements, [[[0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 3.0, 10.0, 16.0, 21.0, 25.0]]], 'Different results in movement measurement')

    def test_absolute_measure_movement(self):
        
        for movemeter in self.movemeters:
            movemeter.absolute_results = True
            movements = movemeter.measure_movement(0)
            self.assertEqual(movements, [[[130.0, 130.0, 131.0, 131.0, 131.0, 131.0], [500.0, 503.0, 510.0, 516.0, 521.0, 525.0]]], 'Different results in movement measurement')
            movemeter.absolute_results = False

