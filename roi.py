'''
Different ways to autogenerate ROIs.
'''


def gen_grid(gridpos, blocksize, step=1):
    '''
    Generate a regular grid of blocks.

    gridpos         (x,y,w,h)
    blocksize        (x,y)
    step            in blocks
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



