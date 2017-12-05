from ctypes import *
import math
import random
import cv2
import numpy as np
import timeit 
import sys

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

resize_image = lib.resize_image
resize_image.argtypes = [IMAGE, c_int, c_int]
resize_image.restype = IMAGE

make_capture = lib.get_capture
make_capture.argtypes = [c_char_p]
make_capture.restype = c_void_p 

load_image_from_stream = lib.get_image_from_stream
load_image_from_stream.argtypes = [c_void_p]
load_image_from_stream.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res

def image_to_array(im):
    array = np.zeros((im.h, im.w, im.c), np.float32) 
    for c in range(im.c):
        for y in range(im.h):
            for x in range(im.w):
                assert x < im.w and y < im.h and c < im.c 
                array[y,x,c] = im.data[c*im.h*im.w + y*im.w + x]
    return array[:,:,[2,1,0]]

if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    
    cap = cv2.VideoCapture(1)    
    ret, frame = cap.read()
    print type(frame.shape[0]) 
    frame = frame.astype(np.float32)/255
    print frame.shape
    frame = np.swapaxes(frame, 0, 1)
    frame = np.swapaxes(frame, 0, 2)
    frame = frame[[2,1,0],:,:]
    print frame.shape
    im = IMAGE(frame.shape[2],frame.shape[1],frame.shape[0],frame.ctypes.data_as(POINTER(c_float)))
    disp = np.ctypeslib.as_array(im.data, shape=(im.c, im.h, im.w))
    disp = np.swapaxes(disp, 0, 2)
    disp = np.swapaxes(disp, 0, 1)
    disp = disp[:,:,[2,1,0]]
    cv2.imshow('test',disp)
    cv2.waitKey(0)

    sys.exit(-1)
    #net = load_net("cfg/yolo9x9.cfg", "yolo_9*9_4000-20000.weights", 0)
    meta = load_meta("cfg/fall_detection.data")
       
    cap = make_capture("/home/l301/old/hxy/fall_detection/databese/video/positive/1.5/p1.mp4")
    while(1):
        im = load_image_from_stream(cap)
        start_time = timeit.default_timer()
        channels = im.c
        if channels == 0:
                print "stream closed"
                sys.exit(-1)
        disp = np.ctypeslib.as_array(im.data, shape=(im.c, im.h, im.w))
        disp = np.swapaxes(disp, 0, 2)
        disp = np.swapaxes(disp, 0, 1)
        disp = disp[:,:,[2,1,0]]
        r = detect(net, meta, resize_image(im, 448, 448))
        print timeit.default_timer() - start_time
        print r
        cv2.imshow("cam",disp)
        cv2.waitKey(1)
        del disp
        free_image(im)

