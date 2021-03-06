#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import argparse
import imutils
import time
import cv2
import os


# In[2]:


FLAGS = []
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model-path',
	type=str,
	default='./yolov3-coco/',
	help='The directory where the model weights and \
		  configuration files are.')

parser.add_argument('-w', '--weights',
	type=str,
	default='./yolov3-coco/yolov3-tiny.weights',
	help='Path to the file which contains the weights \
		 	for YOLOv3.')

parser.add_argument('-cfg', '--config',
	type=str,
	default='./yolov3-coco/yolov3-tiny.cfg',
	help='Path to the configuration file for the YOLOv3 model.')

parser.add_argument('-i', '--image-path',
	type=str,
	help='The path to the image file')

parser.add_argument('-v', '--video-path',
	type=str,
	help='The path to the video file')


parser.add_argument('-vo', '--video-output-path',
	type=str,
        default='./output.avi',
	help='The path of the output video file')

parser.add_argument('-l', '--labels',
	type=str,
	default='./yolov3-coco/coco-labels',
	help='Path to the file having the \
				labels in a new-line seperated way.')

parser.add_argument('-c', '--confidence',
	type=float,
	default=0.5,
	help='The model will reject boundaries which has a \
			probabiity less than the confidence value. \
			default: 0.5')

parser.add_argument('-th', '--threshold',
	type=float,
	default=0.3,
	help='The threshold to use when applying the \
			Non-Max Suppresion')

parser.add_argument('--download-model',
	type=bool,
	default=False,
	help='Set to True, if the model weights and configurations \
			are not present on your local machine.')

parser.add_argument('-t', '--show-time',
	type=bool,
	default=False,
	help='Show the time taken to infer each image.')

FLAGS, unparsed = parser.parse_known_args()


# In[3]:


# Download the YOLOv3 models if needed
if FLAGS.download_model: subprocess.call(['./yolov3-coco/get_model.sh'])

# load the COCO class labels our YOLO model was trained on
LABELS = open(FLAGS.labels).read().strip().split('\n')

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


# In[4]:


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# In[5]:


FLAGS.video_path = '/home/madhur/Downloads/4.mp4'


# In[6]:


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(FLAGS.video_path)
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2()         else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1


# In[7]:


box = []
kick = 0


# In[8]:


# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    kickCount = 0

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > FLAGS.confidence:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    #np.concatenate(box,boxes)
    
    #print(type(boxes))
    
    #print(type(classIDs))
    #res = {frame: boxes} 
    
    
            
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence,
        FLAGS.threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(FLAGS.video_output_path, fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    writer.write(frame)
    
#print("[INFO] cleaning up...")
#writer.release()
#vs.release()


# In[ ]:


kick = 0
acc = []
for i in range(len(box)):
    x2 = box[i+1][0]
    y2 = box[i+1][1]
    x1 = box[i][0]
    y1 = box[i][1]
    distsquared = np.square(x2 - x1) + np.square(y2 - y1)
    dist = np.sqrt(distsquared)
    time = total/30
    acc[i] = ((2*dist)/np.square(time))
    
for i in range(len(acc)):
    netAcc = acc[i+1]-acc[i]
    if (netAcc>5):
        kick+=1
        
print(kick)

