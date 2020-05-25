import tensorflow
from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
import os
from os.path import join
from sklearn.model_selection import train_test_split

def Dataset_Setup(Dir,class1,class2): #Setsup dataset with given directory and classes
    directory = Dir #Sets the directory
    class1_Direc = join(directory, class1) #Adds the class1 path to the directory
    class2_Direc = join(directory, class2) #Adds the class2 path to the directory

    folders = os.listdir(directory) #Gets the class folders
    image_Frame = pd.DataFrame(columns=['filename','class']) #Setsup the dataframe to hold the images data
    for folder in folders: #Loops through both class folders
        files = [join(directory,folder,file) for file in os.listdir(join(directory,folder))] #Gets all the image files in the folder
        data_frame = pd.DataFrame({'filename':files, 'class':folder}) #Adds the images to the dataframe and gives them the class name
        image_Frame = pd.concat([image_Frame,data_frame]) #Combines the dataframes

    allowed = ['jpg','jpeg','png'] #Allowed file types
    for image in image_Frame.filename: #Loops through each image in the dataframe
        fileType = str.lower(os.path.splitext(image)[1])[1:] #Finds the image's file type
        if fileType not in allowed: #Chacks if the file type is valid
            image_Frame = image_Frame[image_Frame.filename != image] #Removes the image from the dataframe

    #Splits the dataframe into train and validation data    
    x_train, x_test, y_train, y_test = train_test_split(image_Frame.filename, image_Frame['class'], test_size=0.2,random_state=42)

    datagen = ImageDataGenerator() #Starts up the generator object

    #Creates the train and validation generators 
    train = datagen.flow_from_dataframe(pd.DataFrame({'filename':x_train,'class':y_train}),target_size=(256,256),class_mode='binary',batch_size=32,color_mode='grayscale')
    validate = datagen.flow_from_dataframe(pd.DataFrame({'filename':x_test,'class':y_test}),target_size=(256,256),class_mode='binary',batch_size=32,color_mode='grayscale')
    

    image_Frame = None #Resets the dataframe
    return(train, validate) #Returns the generators


def Setup_Model(): #Creates a new untrained CNN
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(256,256,1))) #Reformats data
    model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256,activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    sgd = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
    return(model)

def Evaluate_Model(model,validate): #Returns the accuracy of the given model
   loss,accuracy = model.evaluate_generator(validate,verbose=0)
   return accuracy
    
def Train_Model(model,train,validate): #Trains a given model and saves it when finished
    model.fit_generator(train,validation_data=validate,steps_per_epoch=100,epochs=10)
    model.save('trained.h5')
    return(model)
    
def Test_Model(model,test): #Gets predictions from a given model with a given test set
    prediction = model.predict(test)
    return prediction

def Load_Model(): #Loads in a saved trained CNN
    model = keras.models.load_model('trained.h5')
    return(model)

def SetupFrameData(frame): #Prepares given image data to be put through the CNN
    FrameData = np.array([[frame]])
    FrameData = FrameData.reshape(FrameData.shape[0],256,256,1)
    FrameData = FrameData.astype('float32')
    return FrameData
