'''
Parametric functions to generate ROIs or grids or
other combinations of ROIs
'''

import math
import numpy as np
import scipy.optimize


def _get_cps(blocks):
    return [np.array((bx+bw/2, by+bh/2)) for bx, by, bw, bh in blocks]


def gen_grid(gridpos, blocksize, step=1):
    '''
    Fill a large ROI (gridpos) with smaller ROIs to create
    a grid of ROIs.

    gridpos         (x,y,w,h) in pixels
    blocksize       (x,y) in pixels
    step            Relative step between grids, in blocks

    Returns a list of ROIs (x,y,w,h)
    '''
    
    blocks = []

    # Grid coordinates
    gx, gy, gw, gh = gridpos
    bw, bh = blocksize

    xblocks = int(gw/(bw*step))
    yblocks = int(gh/(bh*step))

    for j in range(0, yblocks):
        for i in range(0, xblocks):
            bx = gx + i*(bw*step)
            by = gy + j*(bh*step)
            
            blocks.append([int(bx),int(by),bw,bh])
    
    return blocks


def _square_to_ellipse(blocks, Rx, Ry, CP, lower):
    cps = _get_cps(blocks)
    blocks = [block for block, cp in zip(blocks, cps) if lower**2 <= (cp[0]-CP[0])**2+( (cp[1]-CP[1])*(Rx/Ry) )**2 <= Rx**2]
    
    return blocks


def grid_along_ellipse(gridpos, blocksize, step=1, lw=None, fill=False):
    '''
    Variation of gen_grid but on circle
    '''
    blocks = gen_grid(gridpos, blocksize, step=step)
    
    # block cp distances from
    CP = _get_cps([gridpos])[0]
    Rx = gridpos[2]/2
    Ry = gridpos[3]/2

    if fill:
        lower = 0
    else:
        if lw is None:
            lw = blocksize[0] * 1.1
        lower = Rx - lw

    return _square_to_ellipse(blocks, Rx, Ry, CP, lower)


def grid_along_line(p0, p1, d, blocksize, step=1):
    '''
    Fill a grid along a line segment, where each window
    is maximally at distance d from the line.

    p0, p1 : array-like or tuple
        Start and end points (x,y)
    d : int
        Maximal window center-linepoint distance
    '''
    
    p0 = np.array(p0)
    p1 = np.array(p1)
    
    w = p0[0] - p1[0]
    h = p0[1] - p1[1]
    
    if w < 0:
        w = -w
        x = p0[0]
    else:
        x = p1[0]
    
    if h < 0:
        h = -h
        y = p0[1]
    else:
        y = p1[1]
    
    # Draw horizontal and vertical line properly
    if w < d:
        w = d
    if h < d:
        h = d

    blocks = gen_grid((x-int(w/2), y-int(h/2), 2*w, 2*h),
            blocksize=blocksize, step=step)
    
    cps = _get_cps(blocks)

    n = p1 - p0
    blocks = [block for block, cp in zip(blocks, cps) if abs(np.cross(n, np.array(cp)-p0)/np.linalg.norm(n)) < d]
    
    # Cut to the line segment (dont let past p0 and p1)
    if abs(p0[0] - p1[0]) > abs(p0[1] - p1[1]):
        blocks = [block for block in blocks if x <= block[0]+block[2]/2 <= x+abs(p0[0]-p1[0])]
    else:
        blocks = [block for block in blocks if y <= block[1]+block[3]/2 <= y+abs(p0[1]-p1[1])]

    return blocks



def _workout_circle(points):
    npoints = np.array(points)

    def distances_to_cp(cp):
        return np.sqrt((npoints[:,0]-cp[0])**2 + (npoints[:,1]-cp[1])**2)

    def residual(cp):
        cp_i = distances_to_cp(cp)
        return cp_i - np.mean(cp_i)

    cp, err = scipy.optimize.leastsq(residual, np.mean(npoints, axis=0))
    R = np.mean(distances_to_cp(cp))
    return cp, R


def grid_arc_from_points(gridpos, blocksize, step=1, points=None, circle=None, lw=None):
    '''
    Give a points that make up a circle
    '''
    if lw is None:
        lw = blocksize[0]*1.1

    if circle is None and points is not None:
        cp, R = _workout_circle(points)
    else:
        cp, R = circle

    blocks = gen_grid(gridpos, blocksize, step=step)
    
    return _square_to_ellipse(blocks, R, R, cp, R-lw)



def grid_radial_line_from_points(gridpos, blocksize, step=1,
        points=None, circle=None, line_len=0,
        i_segment=0, n_segments=10):
    '''
    Giving either the points or the circle parameters, create radial lines or
    '''
    if circle is None and points is not None:
        cp, R = _workout_circle(points)
    else:
        cp, R = circle
    
    blocks = gen_grid(gridpos, blocksize, step=step)
    blocks = _square_to_ellipse(blocks, R, R, cp, R-line_len)
    
    # blocks angles
    angles = [math.atan2(y-cp[1], x-cp[0])+math.pi for x,y in _get_cps(blocks)]
    
    # Angle limits
    A = i_segment * ((2*math.pi) / n_segments)
    B = (i_segment+1) * ((2*math.pi) / n_segments)

    blocks = [block for block, angle in zip(blocks, angles) if A < angle <= B]

    return blocks



