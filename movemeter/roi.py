'''
Parametric functions to generate ROIs or grids or
other combinations of ROIs
'''


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
