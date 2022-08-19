## --- DATA MODIFIERS --- ##

## Functions for list sorting ##
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    import re

    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    import re
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

## Channel-appending function ##
def combine(a,b):
    ''' Takes two (2) lists with binary footprint images. Joins two binary maps into one based
    on their position on the list. Adjusts so values are either 0 or 255.
    Returns a new (1) list of binary map images '''

    import numpy as np
    from PIL import Image

    combined=[]
    for i in range(len(a)):
        c=np.asarray(a[i])+np.asarray(b[i])
        c[np.where(c!=0)]=255
        c=Image.fromarray(c)

        combined.append(c)
    return combined

## Augmenting function ##
def Augment(X, Y):

    ''' Takes images, returns augmented X and Y (applies rotations and flips to image to augment data x8)
    and/or turns images to numpy matrices. Only augments if X > 30 in length (train set)'''

    from PIL import Image, ImageOps
    import numpy as np

    rotations=[0,90,180,270]

    #Lists of transformed
    r0=[]
    r90=[]
    r180=[]
    r270=[]

    rh0=[]
    rh90=[]
    rh180=[]
    rh270=[]

    for image in X:
        image1=image
        image2=ImageOps.mirror(image)

        r0.append(np.asarray(image1.rotate(0)))
        r90.append(np.asarray(image1.rotate(90)))
        r180.append(np.asarray(image1.rotate(180)))
        r270.append(np.asarray(image1.rotate(270)))

        rh0.append(np.asarray(image2.rotate(0)))
        rh90.append(np.asarray(image2.rotate(90)))
        rh180.append(np.asarray(image2.rotate(180)))
        rh270.append(np.asarray(image2.rotate(270)))

    X_new=r0+r90+r180+r270+rh0+rh90+rh180+rh270
    Y_new=Y*8

    if len(Y)<30:
        X_new=r0
        Y_new=Y

    return X_new, Y_new

## Set splitting function ##
def train_test_split1(X,Y,test_size,seed):

    ''' Splits a given X data and Y data according to split size returns 4 variables'''

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test =train_test_split(X, Y,test_size=test_size,shuffle=True,random_state=seed)

    return X_train, X_test, Y_train, Y_test

## Standardizing Functions ##
def standardize(X_train,X_val,X_test):
    ''' Takes list of numpy matrices and standardizes based on ImageNet
    means and stds for RGB channels. Returns numpy matrices list with standardized values'''

    import os
    import pandas as pd
    from PIL import Image
    from numpy import asarray
    import numpy as np

    #Active code is to scale everything from 0 to 1
    image_stack=X_train,X_val,X_test

    #Code below is to standardize based on mean and std.

    mean_list1=[]
    mean_list2=[]
    mean_list3=[]

    std_list1=[]
    std_list2=[]
    std_list3=[]

    for image in image_stack:
        pixels = asarray(image)
        pixels = pixels/250
        mean_list1.append(np.mean(pixels[:,:,0]))
        mean_list2.append(np.mean(pixels[:,:,1]))
        mean_list3.append(np.mean(pixels[:,:,2]))

        std_list1.append(np.std(pixels[:,:,0]))
        std_list2.append(np.std(pixels[:,:,1]))
        std_list3.append(np.std(pixels[:,:,2]))

    mean1=0.485
    mean2=0.456
    mean3=0.406

    std1=0.229
    std2=0.224
    std3=0.225

    X_train2=[]
    X_val2=[]
    X_test2=[]

    for image in X_train:
        pixels = asarray(image)
        pixels = pixels/250
        pixels[:,:,0] = (pixels[:,:,0] - mean1) / std1
        pixels[:,:,1] = (pixels[:,:,1] - mean2) / std2
        pixels[:,:,2] = (pixels[:,:,2] - mean3) / std3
        X_train2.append(pixels)

    for image in X_val:
        pixels = asarray(image)
        pixels = pixels/250
        pixels[:,:,0] = (pixels[:,:,0] - mean1) / std1
        pixels[:,:,1] = (pixels[:,:,1] - mean2) / std2
        pixels[:,:,2] = (pixels[:,:,2] - mean3) / std3
        X_val2.append(pixels)

    for image in X_test:
        pixels = asarray(image)
        pixels = pixels/250
        pixels[:,:,0] = (pixels[:,:,0] - mean1) / std1
        pixels[:,:,1] = (pixels[:,:,1] - mean2) / std2
        pixels[:,:,2] = (pixels[:,:,2] - mean3) / std3
        X_test2.append(pixels)

    return X_train2, X_val2, X_test2

def standardize2(X_train,X_val,X_test):
    '''Standardizes binary maps (numpy matrices list) so its only 0 & 1 values.
    Returns standardized list of numpy matrices'''

    import numpy as np
    new_train=[]
    new_val=[]
    new_test=[]

    for matrix in X_train:
        matrix=matrix.copy()
        matrix[np.where(matrix!=0)]=255
        matrix=matrix/255
        new_train.append(matrix)

    for matrix in X_val:
        matrix=matrix.copy()
        matrix[np.where(matrix!=0)]=255
        matrix=matrix/255
        new_val.append(matrix)

    for matrix in X_test:
        matrix=matrix.copy()
        matrix[np.where(matrix!=0)]=255
        matrix=matrix/255
        new_test.append(matrix)

    return new_train,new_val,new_test

## Channel-appending function ###
def join(orig,context):
    ''' Takes two lists with numpy matrices. Stacks them as additional channels
    to create 5- and 6- channel images. Returns list of 5- and 6-channel numpy
    matrices '''

    import numpy as np

    joined=[]

    for i in range(len(orig)):
        joined_image=np.dstack((orig[i],context[i]))
        joined.append(joined_image)

    return joined

## -- Model Creators -- ##

## -- 1-branched models -- ##

def model_creator():

    ''' Creates a 1-branch CNN model using pre-trained weights and architecture from VGG16.
    The CNN has a 3-channel input. It also changes the last layer into
    a linear activation for regression. '''

    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, InputLayer
    import tensorflow as tf
    tf.random.set_seed(2)

    #Setting reference model
    reference_model=VGG16(weights='imagenet')

    #Creating modified VGG-16 model for this project.
    model=Sequential()
    #model.add(Conv2D(input_shape=(224,224,channels),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    for layer in reference_model.layers[0:-1]: #Copy VGG-16 layers except last
        model.add(layer)

    #Add last regression layer
    model.add(Dense(name="regression",units=1, activation="linear")) #Setting up the output layer as regression layer

    return model

def model_creator2(channels):

    ''' Creates a 1-branch CNN model using pre-trained weights and architecture from VGG16.
    It takes in a channel input, which is used to change the input layer of the CNN
    to allow for more channels in an image. It also changes the last layer into
    a linear activation for regression. '''

    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, InputLayer
    import tensorflow as tf
    tf.random.set_seed(2)

    #Setting reference model
    reference_model=VGG16(weights='imagenet')

    #Creating modified VGG-16 model for this project.
    model=Sequential()
    model.add(Conv2D(input_shape=(224,224,channels),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

    for layer in reference_model.layers[2:-1]: #Copy VGG-16 layers except last
        model.add(layer)

    tf.random.set_seed(2)
    model.add(Dense(name="regression",units=1, activation="linear")) #Setting up the output layer as regression layer

    return model

## -- 2-branched models -- ##

def concat_model1():

    ''' Creates a 2-branch CNN model using pre-trained weights and architecture from VGG16.
    The CNN has a 3-channel input on each branch. It also changes the last layer into
    a linear activation for regression. '''

    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, InputLayer
    import tensorflow as tf

    #Creating reference VGG16 models for later copying
    tf.random.set_seed(2)
    reference_model=VGG16(weights='imagenet')
    reference_model2=VGG16(weights='imagenet')

    #Branch1
    model1=Sequential()
    for layer in reference_model.layers[0:-4]: #Copy VGG-16 layers except last layers
        model1.add(layer)

    #Renaming layers prior to copying
    for layer in reference_model2.layers:
        layer._name = layer.name + "_branch2"

    #Branch2
    model2=Sequential()
    for layer in reference_model2.layers[0:-4]: #Copy VGG-16 layers except last layers
        model2.add(layer)

    #Joining both branches
    combinput=concatenate([model1.output,model2.output])

    #Defining last layers as per VGG16
    x=Flatten()(combinput)
    x=Dense(4096, activation="relu")(x)
    x=Dense(4096, activation="relu")(x)
    x=Dense(1, activation="linear")(x) #Setting last layer as dense,linear for regression
    model=Model(inputs=[model1.input, model2.input], outputs=x)

    return model

def concat_model2():
    ''' Creates a 2-branch CNN model. The first branch uses pre-trained weights,
    architecture from VGG16, and a 3-channel input. The second branch uses randomly
    intialized weights, architecture from VGG16, and a 1-channel input.
    The last layer is changed to a linear activation for regression. Finally,
    the RGB branch is frozen in this function.'''

    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, InputLayer
    import tensorflow as tf

    #Creating reference VGG16 models for later copying
    tf.random.set_seed(2)
    reference_model=VGG16(weights='imagenet')
    reference_model2=VGG16(weights=None)

    #Creating Branch1
    model1=Sequential()
    for layer in reference_model.layers[0:-4]: #Copy VGG-16 layers except last layers
        model1.add(layer)

    for layer in model1.layers: #Freezing RGB branch
        layer.trainable=False

    #Creating Branch 2
    for layer in reference_model2.layers:
        layer._name = layer.name + "_branch2"

    #Setting first layer of branch 2 so it has a 1-channel input.
    model2=Sequential()
    model2.add(Conv2D(input_shape=(224,224,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

    for layer in reference_model2.layers[2:-4]: #Copy VGG-16 layers except first and last layers
        model2.add(layer)

    #Concat models
    combinput=concatenate([model1.output,model2.output])

    #Final layers
    x=Flatten()(combinput)
    x=Dense(4096, activation="relu")(x)
    x=Dense(4096, activation="relu")(x)
    x=Dense(1, activation="linear")(x) #last linear layer for regression
    model=Model(inputs=[model1.input, model2.input], outputs=x)

    return model

## GPU-related function ##

def reset_keras2():
    ''' Unloads and loads GPU to reset memory cache '''

    import tensorflow as tf

    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    config.gpu_options.allow_growth=True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
