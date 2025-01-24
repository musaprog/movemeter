'''Create one file script of embed tk
'''

import os

from .version import __version__

from .embedtk import __doc__ as dok

backends = '''
class im_backends:
    tifffile = 0
    opencv = 1

def get_im_backend(name):
    if callable(name):
        return name    
    return load_image

class cc_backends:
    opencv = 0

def get_cc_backend(name):
    return find_location, find_rotation
'''

header = f'''embedtk - Scriptified Embdedded Movemeter

Based on selected code from movemeter v{__version__}.

Example 1 - Making an application

    import tkinter as tk
    from movemeter_embedtk import MovemeterWidget

    # Construct your tkinter application as usual
    root = tk.Tk()
    root.title('My application title')

    # Add movemeter on it, in this example, using the grid
    # layout engine of tkinter and make it fill the window
    gui = MovemeterWidget(root)
    gui.grid(row=0, column=0, sticky='NSWE')
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Set the stack to analyse
    gui.set_stack('mystack.tiff', fs=100)

    # Run tkinter
    root.mainloop()

    # Get the data analysis out
    data = gui.get_results()
    print(data)
'''


def fetch_sources(fns):
    src = []
    for fn in fns:
        print(f'Loading {fn}')
        with open(fn, 'r') as fp:
            source = fp.read()
        src.append(source)
    
    return src

def removelines(sources, toremove):
    
    newsource = []


    for source in sources:
        newsource.append([])
        removenext=False
        for line in source.split('\n'):
            
            if removenext:
                if ')' in line:
                    removenext=False
                print(f'removing line: {line}')
                continue

            keep = True
            for torem in toremove:
                if line.strip(' ').startswith(torem):
                    keep = False
                    continue
            if keep:
                newsource[-1].append(line)
            else:
                print(f'removing line: {line}')

                if line.strip(' ').endswith('('):
                    removenext=True

        newsource[-1] = '\n'.join(newsource[-1])

    return newsource

def replacelines(sources, toreplace):
    newsource = []

    for source in sources:
        newsource.append([])
        
        for line in source.split('\n'):
            rep = None
            for torep in toreplace:
                if line.strip(' ').startswith(torep[0]):
                    rep = torep[1]
                    continue

            if rep:
                newsource[-1].append(rep)
            else:
                newsource[-1].append(line)

        newsource[-1] = '\n'.join(newsource[-1])

    return newsource


def purge_main(sources):
    '''Remove main sections for the given code
    '''
    newsource = []
    for source in sources:
        newsource.append([])
        for line in source.split('\n'):
            if line.startswith('def main(') and line.endswith('):'):
                break
            if '__name__' in line and '__main__' in line:
                break

            newsource[-1].append(line)

        newsource[-1] = '\n'.join(newsource[-1])
    return newsource
    
def main():
    
    codedir = os.path.dirname(__file__)
    
    target = 'movemeter_embedtk.py'
    
    movsources = [
            'cc_backends/opencv.py',
            'im_backends/opencv.py',
            'im_backends/tifffile.py',
            'stacks.py',
            'movemeter.py',
            'tkgui.py',
            'embedtk.py',
            ]

    movsources = [os.path.join(codedir, fn) for fn in movsources]

    tk_sources = [
            'elements.py',
            'matplotlib.py',
            ]
    tk_sources = [os.path.join(codedir, '../../tk_steroids/tk_steroids',
                               fn) for fn in tk_sources]

    fns = tk_sources + movsources
    sources = fetch_sources(fns)

    to_removelines = [
            'from movemeter',
            'from tk_steroids',
            'from .tkgui',
            'from .cc_backends',
            'from .im_backends'
            ]

    to_replacelines = [
            ['from .version', f'__version__ = "{__version__}"']
            ]
    
    sources = removelines(sources, to_removelines)
    sources = replacelines(sources, to_replacelines)


    mainindex = fns.index(os.path.join(codedir, 'embedtk.py'))
    mainsource = sources.pop(mainindex)

    sources = purge_main(sources)
    sources = ['"""'+header, '"""', backends] + sources + [mainsource]

    with open(target, 'w') as fp:
        fp.write('\n'.join(sources))


if __name__ == "__main__":
    main()
