import cv2

def resize_frames(frames,dims=[128,128]):
    ret_frames = []
    for frame in frames:
        ret_frames.append( cv2.resize(frame, (dims[0],dims[1])) )
    return ret_frames

def str_label_to_numeric(emotions,labels):
    """Convert text labels to numeric labels"""
    return [emotions.index(label) for label in labels]
    
