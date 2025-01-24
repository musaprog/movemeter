
import tkinter as tk
import numpy as np


from .tkgui import MovemeterTkGui

class MovemeterWidget(tk.Frame):
    '''Movemeter embedded tkinter widget

    Attributes
    ----------
    parent : obj
        Tkinter parent widget or root window
    gui : obj
        The MovementerTkGui instance
    '''

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.gui = MovemeterTkGui(self)
        self.gui.grid(row=0, column=0, sticky='NSWE')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.gui.folview.grid_remove()
        self.gui.export_button.grid_remove()
        self.gui.export_name.grid_remove()

        self.gui.opview.grid_remove()
        self.gui.opview.grid(row=1, column=0, sticky='NSWE')

        self.gui.tabs.grid_remove()

    def set_stack(self, stack, fs):
        '''Set input image stack

        Arguments
        ---------
        stack : string or obj
            Stack to analyse. Can be a filename pointing to a file
            on disk or in-Python-memory image (numpy array).
        fs : int or float
            Stack imaging frequency in frames per second.
        '''
        self.clear_directories()
        if isinstance(stack, str):
            self.gui.open_stack(stack_fn=stack)
        elif isinstance(stack, np.ndarray):
            raise NotImplementedError('Setting loaded images not yet supported')
        else:
            type_ = type(stack)
            raise TypeError(f"Unkown image format {type_}")

    def get_results(self):
        '''Returns the current analysed results if any available

        Returns None if no results

        Returns
        -------
        results : list
            [roi_group_1, roi_group_2, ...]

            roi_group_i = [roi_1, roi_2, roi_3, ...]

            roi_i = [xmovements, ymovements]
        '''
        if not self.results:
            return None

        return self.results


def main():
    
    root = tk.Tk()
    root.title('Movemeter - Embedded Tkinger GUI')
    gui = MovemeterWidget(root)
    gui.grid(sticky='NSWE')
    root.columnconfigure(1, weight=1)
    root.rowconfigure(1, weight=1)

    root.mainloop()

if __name__ == "__main__":
    main()
