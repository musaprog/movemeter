<h1>Movemeter</h1>

Movemeter is a motion analysis tool to quantify how much arbitrary
*image features* move in pixels over time. It currently uses template matching
from the OpenCV library (cv2.matchTemplate)
on upscaled images to reach subpixel resolution.

*The features* are selected by drawing regions of interest (ROIs),
which consist of small rectangular *windows*, on the images.
These analysis *windows* can be set to

1. follow the selected feature
to quantify how much and where it moves
2. to stay stationary
to quantify overal movement within that image area.

Results are reported in the units of pixels in x and y,
in square root displacement values sqrt(x^2+y^2),
or as heatmap images, and they can be exported as CSV
files for easy import into external plotting and further analysis software.


<h2>Installing</h2>

The latest version from [PyPi](https://pypi.org/)
can be installed with the command

```
pip install movemeter
```


<h2>How to use</h2>

To open the graphical user interface, simply

```
python -m movemeter.tkgui
```

For further details, please see our [Online Documentation](https://musaprog.github.io/docs/movemeter/)

<h2>Other</h2>
Movemeter is still a rather early and unfinished program.
There may be bugs and some parts can be better optimized,
especially the motion analysis calculation can be faster.
