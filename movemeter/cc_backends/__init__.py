
class cc_backends:
    opencv = 0


def get_cc_backend(name):
    '''Return a backend

    Arguments
    ---------
    name : string, int or callable

    Returns
    -------
    find_location : callable
    find_rotation : callable
    '''

    if name.lower() == 'opencv' or name == cc_backends.opencv:
        from .opencv import find_location, find_rotation
        return find_location, find_rotation

    raise ValueError(f'Unkown backend: "{name}" (available {cc_backends})')
        
