import numpy as np
import cv2 as cv

def get_cv_capture(id_or_file):
    return cv.VideoCapture(0)

def get_frame(capture, resize_to=None, grayscale=True):
    has_frame, frame = capture.read()
    frame_r = frame
    if has_frame:
        if resize_to is not None:
            frame_r = frame = cv.resize(frame, resize_to, interpolation=cv.INTER_AREA)
        if grayscale:
            frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    return has_frame, frame, frame_r

def get_flow(capture, frame_count, resize_to):
    has_frame, frame1, frame1_ori = get_frame(capture, resize_to=resize_to)
    if not has_frame:
        return None, None, None

    for i in range(frame_count):
        has_frame, frame2, frame2_ori = get_frame(capture, resize_to=resize_to)
        if not has_frame:
            return None, None, None
        flow = cv.calcOpticalFlowFarneback(frame1, frame2, 
            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        yield frame1_ori, frame2_ori, (flow[...,0], flow[...,1])
        frame1 = frame2

def flow2image(flow, image_buf)->None:
    mag, ang = cv.cartToPolar(flow[0], flow[1])
    image_buf[...,0] = ang*180/np.pi/2
    image_buf[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    return cv.cvtColor(image_buf,cv.COLOR_HSV2BGR)

def show_image(image, delay=30)->bool:
    cv.imshow('frame2',image)
    return wait_key()

def wait_key():
    k = cv.waitKey(30) & 0xff
    return k == 27   

def get_flow_amount(capture, frame_count=120, resize_to=None)->float:
    amts = []
    for frame1, frame2, flow in get_flow(capture, frame_count=frame_count, resize_to=resize_to):
        x, y = flow
        amts.append(np.mean(np.abs(x)) + np.mean(np.abs(y)))
    return np.mean(amts), np.var(amts)

def get_flow_image(capture, frame_count=120, resize_to=None):
    image_buf = None
    for frame1, frame2, flow in get_flow(capture, frame_count=frame_count, resize_to=resize_to):
        if frame1 is None or frame2 is None:
            continue
        if image_buf is None:
            image_buf = np.zeros_like(frame1)
            image_buf[...,1] = 255    
        image = flow2image(flow, image_buf)
        yield image