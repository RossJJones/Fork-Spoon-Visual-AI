import cv2
import numpy as np

YoloNet = cv2.dnn.readNet("YoloNet\yolov3.weights", "yoloNet\yolov3.cfg") #Loads in the Yolov3 pretrined weights
classes = [] #Holds the classes that the Yolov3 net is trained on

with open("YoloNet\coco.names", "r") as file: #Short algorithm to retrieve the classes from a txt file
    classes = [line.strip() for line in file.readlines()]

layers = YoloNet.getLayerNames() #Gets the layers of the Yolov3 network
OutLayers = [layers[out[0]-1] for out in YoloNet.getUnconnectedOutLayers()] #Gets the output layers form the Yolov3

def SetupCamera(): #Setsup a camera object 
    capture = cv2.VideoCapture(0) #Gets the first camera availible
    return capture

def ReadImage(Dir): #Reads in image from given directory
    img = cv2.imread(Dir)
    return img

def GetCurrentFrame(capture): #Gets the next frame from the given camera object
    ret,frame = capture.read()
    frame = cv2.flip(frame,1)
    return frame

def GrayscaleFrame(frame): #Turns the given image grayscale
    grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return grayscale

def ResizeFrame(frame): #Resizes the given image to 256x256
    frame = cv2.resize(frame,(256,256))
    return frame

def QuitCam(capture): #Ends the given camera object
    capture.release()

def ComputeYoloFrame(frame,Bool): #Runs the Yolov3 net and adds the bounding boxes to the given image
    DetectBoxes =[] #Holds bounding box data of detected objects
    Detected = [] #Holds all detected objects
    Classes = [] #Holds the class IDs
    FrameHeight, FrameWidth, c = frame.shape #Gets the size of the image
    BlobbedFrame = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True) #Gets the RGB versions of the image and puts them into an array
    YoloNet.setInput(BlobbedFrame) #Passes the RGB images through the Yolov3 net
    Outputs = YoloNet.forward(OutLayers) #Gets the detected bounding box data from the Yolov3 net
    for outs in Outputs: #Gets a single output
        for detection in outs: #Gets one object detection from the output
            score = detection[5:] #Retrives various scores from the detection
            ClassID = np.argmax(score) #Gets the class, in the form of a number, from the detection
            confidence = score[ClassID] #Gets the confidence score from the detection

            #Checks if the confidence is higher than 50% and if the object is a fork or spoon
            if confidence > 0.5 and (classes[ClassID] == "fork" or classes[ClassID] == "spoon"):
                CenterX = int(detection[0]*FrameWidth) #Gets the x coordinate of the object
                CenterY = int(detection[1]*FrameHeight) #Gets the y coordinate of the object
                Width = int(detection[2]*FrameWidth) #Gets the width of the object
                Height = int(detection[3]*FrameHeight) #gets the height of the object

                x = int(CenterX - Width / 2) #Gets the bounding box x coordinate
                y = int(CenterY - Height / 2) #Gets the bounding box y coordinate

                DetectBoxes.append([x,y,Width,Height]) #Adds the detected bounding box to an array
                Detected.append(float(confidence)) #Adds the confidence score to an array
                Classes.append(ClassID) #Adds the class number to an array
    if Bool: #Checks if the camera is running
        frame = GrayscaleFrame(frame) #Makes the image grayscale
    index = cv2.dnn.NMSBoxes(DetectBoxes,Detected,0.5,0.4) #Gets rid of duplicated bounding boxes and marks only the bounding boxes needed
    for box in range(len(DetectBoxes)): #Loops for the number of bounding boxes detected
        if box in index: #If the current bounding box is a valid bounding box
            x,y,w,h = DetectBoxes[box] #Gets the coordinates of the bounding box
            Class = str(classes[Classes[box]]) #Gets the string version of the object's class
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #Draws the bounding box on the image
            cv2.putText(frame,Class,(x,y+30),cv2.FONT_HERSHEY_PLAIN,3,(255,170,170),3) #Adds the class name to the bounding box

    return frame #Returns the computed image



        








                
    
