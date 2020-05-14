# USAGE
# python detect_faces.py --image rachit.jpeg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
#python detect_faces.py --image david.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel




# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
#in this case we have to construct a argrument parser which takes image caffee model and protext and image as input

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
#blob is created by preprocessing the image
# pass the blob through the network and obtain the detections and
# predictions
#now what we do is pass blob which is obtained by preprocessing and
# pass it to net which is our dnn model and it gives the result
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
#here we draw boxes over each edge
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	#detections is an array which contains a lot of information in a form
	# of array here detection [0,0,i,2]contains informtion about the probality
	confidence = detections[0, 0, i, 2]
	#print(detections[0,0,i,2])

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		#the detections[0, 0, i, 3:7] contains value between zero to 1
		#we convert into points by multyplying by array which contains dimension of image

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

		#convert to int
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 255, ), 4)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 4)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)