
class im_backends:
    tifffile = 0
    opencv = 1

def get_im_backend(name):
    '''Returns a image loading backend matching the name

    Arguments
    ---------
    name : string, int or callable
        If callable, returns the argument as is

    Returns
    load_image : callable
        The image loading backend function
    '''
    if callable(name):
        return name

    if name == 'tifffile' or name == im_backends.tifffile:
        from .tifffile import load_image
    elif name == 'opencv' or name == im_backends.opencv:
        from .opencv import load_image
    else:
        raise ValueError(
                f"Unkown im backend {name} (available {im_backends})")

    return load_image
