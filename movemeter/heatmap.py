import math
import numpy as np

def get_step(rois):
    step = np.linalg.norm(np.array(rois[0][0:2]) - np.array(rois[1][0:2]))
    step = step / (rois[1][0]-rois[0][0])
    return step

def speed_heatmaps(image_shape, rois, movements):
    '''
    Calculate speed heatmap images

    image_shape     Shape of the original images, (height, width)
    rois            List of ROIs, each ROI [x1,x2,w,h]
    movements       List of movements for each ROIs,
                        each [[x1,x2,...],[y1,y2,...]]
    
    Returns a list of 2D numpy arrays
    '''
    heatmap_images = []
    
   
    for i_frame in range(len(movements[0][0])):
        if i_frame == 0:
            continue
        image = np.zeros(image_shape)
        
        step = get_step(rois)

        for ROI, (x,y) in zip(rois, movements):
            values = (np.sqrt(np.array(x)**2+np.array(y)**2))
            value = abs(values[i_frame] - values[i_frame-1])
            xx,yy,w,h = ROI
            
            cx = xx+int(round(w/2))
            cy = yy+int(round(h/2))
            
            image[cy-int(step*(h/2)):cy+int(round(step*(h/2))), cx-int(round(step*(w/2))):cx+int(round(step*(w/2)))] = value
        
        if np.max(image) < 0.01:
            image[0,0] = 1
        heatmap_images.append(image)

    return heatmap_images




def displacement_heatmaps(image_shape, rois, movements):
    '''
    Calculate speed heatmap images

    image_shape     Shape of the original images, (height, width)
    rois            List of ROIs, each ROI [x1,x2,w,h]
    movements       List of movements for each ROIs,
                        each [[x1,x2,...],[y1,y2,...]]
    
    Returns a list of 2D numpy arrays
    '''
    heatmap_images = []
    
   
    for i_frame in range(len(movements[0][0])):
        image = np.zeros(image_shape)
        
        step = get_step(rois)

        for ROI, (x,y) in zip(rois, movements):
            values = (np.sqrt(np.array(x)**2+np.array(y)**2))
            value = values[i_frame]
            xx,yy,w,h = ROI
            
            cx = xx+int(round(w/2))
            cy = yy+int(round(h/2))
            
            image[cy-int(step*(h/2)):cy+int(round(step*(h/2))), cx-int(round(step*(w/2))):cx+int(round(step*(w/2)))] = value
        
        if np.max(image) < 0.01:
            image[0,0] = 1
        heatmap_images.append(image)

    return heatmap_images


def displacement_direction_heatmaps(image_shape, rois, movements):
    heatmap_images = []
    for i_frame in range(len(movements[0][0])):
        image = np.zeros(image_shape)
        
        step = get_step(rois)

        for ROI, (x,y) in zip(rois, movements):
            #values = (np.sqrt(np.array(x)**2+np.array(y)**2))
            
            value = math.atan2(y[i_frame], x[i_frame])
            
            if value < 0:
                value += 2*math.pi

            value /= 2*math.pi

            xx,yy,w,h = ROI
            
            cx = xx+int(round(w/2))
            cy = yy+int(round(h/2))
            
            image[cy-int(step*(h/2)):cy+int(round(step*(h/2))), cx-int(round(step*(w/2))):cx+int(round(step*(w/2)))] = value
        
        if np.max(image) < 0.01:
            image[0,0] = 1
        heatmap_images.append(image)
    
    return heatmap_images







