import cv2
from imageai.Detection import ObjectDetection
from keras import backend as K
import numpy as np

#read video frame

# detector = ObjectDetection()
#
# model_path = "../DetectionModel/yolo-tiny.h5"
# input_path = "../InputData/testImage.png"
# output_path = "../OutputData/outputImage.jpg"
#
# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath(model_path)
# detector.loadModel()
# detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)
#
# for eachItem in detection:
#     print(eachItem["name"] , " : ", eachItem["percentage_probability"])


# Load Yolo
print("LOADING YOLO")
net = cv2.dnn.readNet("../YOLOPreTrainedModel/yolov3.weights", "../YOLOPreTrainedModel/yolov3.cfg")
#save all the names in file o the list classes
classes = []
with open("../YOLOPreTrainedModel/yolov3.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#get layers of the network
layer_names = net.getLayerNames()
#Determine the output layer names from the YOLO model
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")


videoFile = cv2.VideoCapture('../InputData/cc2f(0).avi')
count = 0
while 1:
    ret, frame = videoFile.read()
    if not ret:
        break
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    gray = cv2.resize(frame, None, fx=0.4, fy=0.4)
    #cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Capture frame-by-frame
    #img = cv2.imread("../InputData/o1.PNG")
    #     img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = gray.shape

    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(gray, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    # Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # We use NMS function in opencv to perform Non-maximum Suppression
    # we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(gray, (x, y), (x + w, y + h), color, 2)
            cv2.putText(gray, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        1 / 2, color, 2)

    #cv2.imshow("Image", gray)
    count += 1
    fileName = "../OutputData/image_" + str(count) + ".png"
    result = cv2.imwrite(fileName, gray)
    print(count)
    #cv2.waitKey(0)

print("Ended")
videoFile.release()
cv2.destroyAllWindows()