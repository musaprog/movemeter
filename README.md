<h1>Movemeter</h1>
A motion analysis tool that uses template matching
on upscaled images to reach subpixel resolution.

The template matching is performed using
the cv2.matchTemplate function (normalised cross-correlation) from OpenCV.

Results are reported in units of pixels in x and y,
in square root displacement values sqrt(x^2+y^2),
or as heatmap images of the latter.


<h2>Installing</h2>

The latest version from PyPi can be installed with the command

```
pip install movemeter
```


<h2>How to use</h2>

To open the graphical user interface

```
python -m movemeter.tk_meter
```


<h2>Other</h2>
This is an early, unfinished piece of software.


