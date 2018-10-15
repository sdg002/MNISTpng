import numpy as np
import glob
import cv2
from keras.utils import np_utils
import os.path

#
#Load an image using the specified path. opencv needed
#Do not resize
#
def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    return img;
    # Reduce size
    #resized = cv2.resize(img, (128, 96))
    #return resized
#
#Load images and labels. Returns a tuple of image data,label
#
def load_images(path_in):
        filenames = glob.glob(path_in)
        images=[] 
        labels=[] #labels for each training file
        filenames = glob.glob(path_in)
        for filename in filenames:
                #get the parent folder from the full path of the file /mnist/blah/training/3/34348.png
                fulldir=os.path.dirname(filename)
                parentfolder=os.path.basename(fulldir)
                imagelabel=int(parentfolder)
                labels.append(imagelabel)
                img = get_im(filename)
                images.append(img)
                # if ("\\2\\" in filename):
                #         labels.append(0)			
                #         img = get_im(filename)
                #         images.append(img)          
                # elif ("\\3\\" in filename):
                #         labels.append(1)
                #         img = get_im(filename)
                #         images.append(img)
        return images,labels

#
#The output from load_images() is further refined
#
def ReShapeData(data,target,numclasses):
        data_out = np.array(data, dtype=np.uint8)
        target_out = np.array(target, dtype=np.uint8)
        data_out = data_out.reshape(data_out.shape[0],  28,28)
        data_out = data_out[:, :, :, np.newaxis]
        data_out = data_out.astype('float32')
        data_out /= 255
        target_out = np_utils.to_categorical(target_out, numclasses)
        return data_out,target_out
