'''
Parametric functions to generate ROIs or grids or
other combinations of ROIs
'''

import numpy as np


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
    
    cps = [np.array((bx+bw/2, by+bh/2)) for bx, by, bw, bh in blocks]

    n = p1 - p0
    blocks = [block for block, cp in zip(blocks, cps) if abs(np.cross(n, np.array(cp)-p0)/np.linalg.norm(n)) < d]
    
    # Cut to the line segment (dont let past p0 and p1)
    if abs(p0[0] - p1[0]) > abs(p0[1] - p1[1]):
        blocks = [block for block in blocks if x <= block[0]+block[2]/2 <= x+abs(p0[0]-p1[0])]
    else:
        blocks = [block for block in blocks if y <= block[1]+block[3]/2 <= y+abs(p0[1]-p1[1])]

    return blocks




