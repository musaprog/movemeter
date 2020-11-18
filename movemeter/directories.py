'''
Module setting the default directories.

Attributes
----------
MOVEDIR : string
    Where to save results and any temporal data.
    By default, a folder Movemeter (Windows) or .movemeter (other platforms)
    in the user home directory.
'''


import os
import platform

USER_HOMEDIR = os.path.expanduser('~')

if platform.system() == "Windows":
    MOVEDIR = os.path.join(USER_HOMEDIR, 'Movemeter')
else:
    MOVEDIR = os.path.join(USER_HOMEDIR, '.movemeter')


