from NeuralNet import *
from CameraSetup import *
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
Dir = 'Datasets/spoon-vs-fork/spoon-vs-fork'
class1 = 'fork'
class2 = 'spoon'

train,validate = Dataset_Setup(Dir,class1,class2) #Used to get training sets



class Loop(): #Superclass for both GUI camera loops
    def __init__(self):
        self.LOOP=None #Used to hold the GUI loop iterator
        self._CurrentFrame=None #Used to hold the current frame from the camera
        self.capture=None #Used to hold the camera object
        self._img = tk.Label(frame,image=None) #Sets up the image frame in the GUI window
        

    def RunCamera(self): #Basic blueprint for methods, which will be polymorphed in later classes
        print("Loop parent method called")

    def RunImage(self):
        print("Loop parent method called")




class ObjectDetectLoop(Loop): #Camera loop class for Yolov3 detection
    def __init__(self):
        super().__init__() #Calls superclass constructor

    def RunCamera(self): #Method for running camera on GUI
        self._img.destroy() #Resets the image in image GUI frame
        if self.capture == None: #If there's no camera ready
            self.capture = SetupCamera() #Setup a new camera object
        self._CurrentFrame = GetCurrentFrame(self.capture) #Retrives the next frame from the camera object
        self._CurrentFrame = ComputeYoloFrame(self._CurrentFrame,True) #Puts the frame data through the Yolov3 network
        self._CurrentFrame = Image.fromarray(self._CurrentFrame) #Creates an image from a 3D array of pixels
        self._CurrentFrame = ImageTk.PhotoImage(self._CurrentFrame) #Prepares the new image for GUI display
        self._img = tk.Label(frame,image=self._CurrentFrame) #Sets the image to the image GUI label
        self._img.image = self._CurrentFrame #Validates the image so that it appears
        self._img.pack() #Displays the new image

        self.LOOP = root.after(20,self.RunCamera) #Allows for the camera to keep running. Loops through this method until cancelled

    def RunImage(self): #Runs the image classifier method
        self._CurrentFrame = tk.filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(('JPG files','*.jpg'),('JPEG files','*.jpeg'),('PNG files','*.png'))) #Runs the image upload GUI
        self._CurrentFrame = ReadImage(self._CurrentFrame) #Loads in image pixel data
        self._CurrentFrame = ComputeYoloFrame(self._CurrentFrame,False) #Put's uploaded image data through Yolov3 network
        self._CurrentFrame = ResizeFrame(self._CurrentFrame) #Reshapes the new image to fit the GUI
        self._CurrentFrame = Image.fromarray(self._CurrentFrame) #Loads in the new image from the 3D array of image data
        self._CurrentFrame = ImageTk.PhotoImage(self._CurrentFrame) #Prepares image to be displayed on GUI
        self._img = tk.Label(frame,image=self._CurrentFrame) #Places image into image GUI frame
        self._img.image = self._CurrentFrame #Validates the image so that is appears
        self._img.pack() #Displays the image

class ImageClassifierLoop(Loop): #Camera loop class for my CNN 
    def __init__(self):
        super().__init__() #Calls superclass constructor
        self.text = tk.Scale(frame2,showvalue=0,state="active",from_=0,to=100,orient="horizontal",length=500) #Setsup the scrollbar on the GUI
        self._prediction = None #Holds the returned prediction from the CNN
        self._image=None #Holds the uploded image
        self._frame=None #Holds data for computed frame 

    def RunCamera(self): #Camera method
        self._img.destroy() #Resets the frame on the GUI
        if self.capture == None: #If the camera isn't setup
            self.capture = SetupCamera() #Setup a new camera object
        self._CurrentFrame = GetCurrentFrame(self.capture) #Gets the next frame from camera
        self._CurrentFrame = GrayscaleFrame(self._CurrentFrame) #Turns the frame grayscale so the CNN can read it
        self.__ComputeFrame() #Gets predictions on frame
        self._CurrentFrame = Image.fromarray(self._CurrentFrame) #Makes image from 3D array of pixel data
        self._CurrentFrame = ImageTk.PhotoImage(self._CurrentFrame) #Prepared image to display on GUI
        self._img = tk.Label(frame,image=self._CurrentFrame) #Puts image on image GUI frame
        self._img.image = self._CurrentFrame #Validates the image so that it appears
        self.text.set(self._prediction) #Sets scroll bar to predicted number
        
        self._img.pack() #Displays current frame on GUI
        self.LOOP = root.after(20,self.RunCamera) #Allows the camera to loop.

    def __ComputeFrame(self): #Puts the frame through the CNN
        self._frame = ResizeFrame(self._CurrentFrame) #Resizes the frame to fit CNN
        self._frame = SetupFrameData(self._frame) #Remakes frame data to fit CNN
        self._prediction = Test_Model(model,self._frame) #Gets predictions from CNN
        self._prediction = self._prediction[0][0] #Gets the prediction out of the 2D array
        self._prediction *= 100 #Makes the prediction user friendly to read

    def RunImage(self): #Runs the image classifier
        self._CurrentFrame = tk.filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(('JPG files','*.jpg'),('JPEG files','*.jpeg'),('PNG files','*.png'))) #Runs image upload GUI
        self._image = ReadImage(self._CurrentFrame) #Loads uploaded image
        self._CurrentFrame = GrayscaleFrame(self._image) #Turns image grayscale
        self.__ComputeFrame() #Gets prediction from CNN
        self._image = ResizeFrame(self._image) #Resizes image to fit GUI frame
        self._image = Image.fromarray(self._image) #Loads image from 3D array of pixel data
        self._image = ImageTk.PhotoImage(self._image) #Prepares image for GUI placement
        self._img = tk.Label(frame,image=self._image) #Places image in GUI frame
        self._img.image = self._CurrentFrame #Validates the image so that it appears
        self.text.set(self._prediction) #Sets scrollbar to prediction
        self.text.pack() #Displays scrollbar
        self._img.pack() #Displays image

        
def ImageClassifier(): #Runs the upload image classifier
    HideMainMenu()
    ShowOtherMenu(True)
    ImageLoop.RunImage()

def ObjectIdentifier(): #Runs the upload object identifier
    HideMainMenu()
    ShowOtherMenu(False)
    ObjectLoop.RunImage()


def ShowMainMenu(): #Displays the main menu
    ImageCam.pack()
    ObjectCam.pack()
    ImageClass.pack()
    ObjIdent.pack()
    Eval.pack()
    end.pack()

def HideMainMenu(): #Hides the main menu
    ImageCam.pack_forget()
    ObjectCam.pack_forget()
    ImageClass.pack_forget()
    ObjIdent.pack_forget()
    Eval.pack_forget()
    end.pack_forget()
    for item in frame2.pack_slaves():
        item.pack_forget()

def BackToMenu(): #Hides current menu and returns to main menu
    if ImageLoop.LOOP != None: #Checks if the camera loop has run for the CNN class
        root.after_cancel(ImageLoop.LOOP)
        QuitCam(ImageLoop.capture)
        ImageLoop.capture = None
        ImageLoop.LOOP = None
    elif ObjectLoop.LOOP != None: #Checks if the camera loop has run for the Yolov3 class
        root.after_cancel(ObjectLoop.LOOP)
        QuitCam(ObjectLoop.capture)
        ObjectLoop.capture = None
        ObjectLoop.LOOP = None
    for item in frame.pack_slaves():
        item.pack_forget()
    for item in frame2.pack_slaves():
        item.pack_forget()
    for item in frame3.pack_slaves():
        item.pack_forget()
    spoon.pack_forget()
    fork.pack_forget()
    ShowMainMenu()

def StartImageCamera(): #Starts the CNN live camera detection
    HideMainMenu()
    ShowOtherMenu(True)
    ImageLoop.text.pack()
    ImageLoop.RunCamera()

def StartObjectCamera(): #Starts the Yolov3 live camera detection
    HideMainMenu()
    ShowOtherMenu(False)
    ObjectLoop.RunCamera()

def ShowOtherMenu(Bool): #Displays the main menu button on other screens
    menu.pack()
    if Bool:
        spoon.pack()
        fork.pack()

def ModelAccuracy():
    for item in frame2.pack_slaves():
        item.pack_forget()
    accuracy = Evaluate_Model(model,validate)
    text = tk.Label(frame2,text="Model accuracy: {:5.2f}%".format(100*accuracy))
    text.pack()

def Quit(): #Quits the GUI
    root.destroy()


model = Load_Model() #Loads in the pretrained CNN
root = tk.Tk() #Starts the GUI loop
bg = tk.Canvas(root, height=500, width=600, bg='#000000') #Sets GUI background
bg.pack() #Shows GUI background

#Sets up various frames on GUI
frame = tk.Frame(root, bg="black")
frame.place(relwidth=0.8,relheight=0.5,relx=0.1,rely=0.25)
frame2 = tk.Frame(root, bg="black")
frame2.place(relwidth=0.8,relheight=0.1,relx=0.1,rely=0.1)
frame3 = tk.Frame(root, bg="black")
frame3.place(relwidth=0.8,relheight=0.1,relx=0.1,rely=0.8)
frame4 = tk.Frame(root,bg="black")
frame4.place(relwidth=0.07,relheight=0.05,relx=0.03,rely=0.1)
frame5 = tk.Frame(root,bg="black")
frame5.place(relwidth=0.07,relheight=0.05,relx=0.9,rely=0.1)

#Sets up various labels on GUI
fork = tk.Label(frame4,text="Fork",padx=0,justify="left",bg="black",fg="white")
spoon = tk.Label(frame5,text="Spoon",padx=0,justify="right",bg="black",fg="white")
ImageLoop=ImageClassifierLoop()
ObjectLoop=ObjectDetectLoop()

#Sets up the GUI buttons
ImageCam = tk.Button(frame, text="Start Real Time Image Classifier",padx=10,pady=5,fg="white",bg="black",command=StartImageCamera)
ObjectCam = tk.Button(frame, text="Start Real Time Object Classifier",padx=10,pady=5,fg="white",bg="black",command=StartObjectCamera)
end = tk.Button(frame, text="Exit",padx=10,pady=5,fg="white",bg="black",command=Quit)
menu = tk.Button(frame3, text="Main Menu",padx=10,pady=5,fg="white",bg="black",command=BackToMenu)
ImageClass = tk.Button(frame, text="Upload Image To Image Classifer",padx=10,pady=5,fg="white",bg="black",command=ImageClassifier)
ObjIdent = tk.Button(frame, text="Upload Image To Object Identifier",padx=10,pady=5,fg="white",bg="black",command=ObjectIdentifier)
Eval = tk.Button(frame, text="Evaluate Model",padx=10,pady=5,fg="white",bg="black",command=ModelAccuracy)

ShowMainMenu() #Starts the main menu

root.mainloop() #Defines the end of the GUI loop
