
import cv2


class MovieIterator:
    '''
    Iteratively return video file (.mp4 or other) frames
    using OpenCV.

    Attributes
    ----------
    i_frame : int
        the index of the frame to be captured next, starting from zero
    '''
    def __init__(self, fn, post_process=None):
        self.i_frame = 0
        self.post_process = post_process
        self.video_capture = cv2.VideoCapture(fn)

    def __iter__(self):
        return self

    def __next__(self, i_frame=None):

        if i_frame is None:
            index = self.i_frame
        
        self.video_capture.open()
        self.video_capture.set(CAP_PROP_POS_FRAMES, float(index))
        
        reval, frame = self.video_capture.read() 
        
        self.video_capture.release()

        if not retval:
            self.i_frame = 0
            raise StopIteration

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if post_process is not None:
            frame = self.post_process(frame)
        
        if i_frame is None:
            self.i_frame += 1

        return frame

    def __len__(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


    def __getitem__(self, index):
        return self.__next__(i_frame=index)
