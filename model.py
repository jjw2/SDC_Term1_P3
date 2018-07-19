

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import random
#get_ipython().run_line_magic('matplotlib', 'inline')

images = []
measurements = []

# Select which datasets to use for training
# Thought about making this automatic, but wanted ability to select individual datasets
datasets = ['Train1', 'Train4', 'Train6', 'Train7', 'Train8', 'Train9', 'Train11', 'Train12']#, 'Train2', 'Train4', 'Train5']


samples = []
steer_correct = [0, 0.25, -0.25]

steer_angle_thresh = 0.04
steer_keep_pct = .33 # percentage of steering angles withn +/- steering threshold to keep

# Loop thorugh selected datasets

# Note: makes sense to do preprocessing here so batch size is properly attributed in the generator below

for dataset in datasets:
    tot_counter = 0;
    dataset_counter = 0;
    with open('data/' + dataset + '_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        
        for line in reader:
            tot_counter = tot_counter + 1
            # Filter out steering angles
            this_steer = float(line[3])
            
            if (((this_steer <= -steer_angle_thresh) | (this_steer >= steer_angle_thresh))  | 
                ((this_steer >= -steer_angle_thresh) & (this_steer <= steer_angle_thresh) & (random() < steer_keep_pct))):
                
                # create a new line for center, left, right
                for i in range(3): # center, left, right
                    
                    sourcepath = line[i]
                    
                    if("/" in sourcepath):
                        filename = sourcepath.split('/')[-1]
                    else:
                        filename = sourcepath.split('\\')[-1]

                    newpath = 'data/' + dataset + '/' + filename

                    newline = [newpath, this_steer+steer_correct[i]]            
                    samples.append(newline)
                    dataset_counter = dataset_counter + 1 
            
    
    print("Total number of images in", dataset, ": " , tot_counter*3)
    print("Number of images in", dataset, "after fitering : ", dataset_counter)

print("Final number of images: ", len(samples))

print("NOTE: This does not include any image flipping, which may occur later!")



angles = []
for sample in samples:
    angles.append(sample[1])

#plt.hist(angles, bins=41, range=(-1.0, 1.0))
    


import sklearn
from sklearn.model_selection import train_test_split

# Split training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)            


def generator(samples, batch_size = 64):
    # Center, Left, Right
    steer_correct = [0, 0.20, -0.20]
    num_samples = len(samples)
    
    while 1:
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                    
                path = batch_sample[0]
                angle = batch_sample[1]

                image = cv2.imread(path)
                #plt.imshow(image)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # Suggested by nvidia
                images.append(image)
                images.append(cv2.flip(image,1)) #Flip image

                angles.append(float(angle))
                angles.append(float(-angle))

            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        
train_generator = generator(train_samples, batch_size=512)
validation_generator = generator(validation_samples, batch_size=512)


## Set Up NNs ##
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout

dropout = [0.0, 0.3, 0.3, 0.5]

def nvidia1():

	model = Sequential()
	model.add(Cropping2D(cropping=((74,20), (30,30)), input_shape=(160, 320, 3))) # 66, 260
	model.add(Lambda(lambda x:(x/255.0)-0.5))
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))#31x98
	model.add(Dropout(dropout[1]))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))#14X47
	model.add(Dropout(dropout[1]))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))#5x22
	model.add(Dropout(dropout[1]))
	model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))#3x20
	model.add(Dropout(dropout[1]))
	model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))#1x18
	model.add(Flatten())
	model.add(Dense(1000))
	model.add(Dropout(dropout[3]))
	model.add(Dense(100))
	model.add(Dropout(dropout[2]))
	model.add(Dense(50))
	model.add(Dense(1))

	model.compile(loss = 'mse', optimizer = 'adam')
	model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*2,
                        validation_data=validation_generator, nb_val_samples=len(validation_samples)*2, nb_epoch=3)

	model.save('model.h5')

nvidia1()

