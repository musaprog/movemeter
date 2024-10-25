
import tifffile

def load_image(fn):
    return tifffile.imread(fn)
