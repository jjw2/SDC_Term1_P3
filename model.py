import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

"""
Select datasets for training
Multiple separate datasets were captured in order to allow for mixing/matching
of data during training. This allowed for a better understanding of what
"worked" and what didn't

See the markdown file for a discussion of data collection.
"""
datasets = ['Train1', 'Train4', 'Train6', 'Train7', 'Train8', 'Train9', 'Train11', 'Train12']

"""
Steering correction for left and right camera images.
Initialized with 0.2, but arrived at the value below through trial and error.
"""
# Camera image: [Center, Left, Right]
steer_correct = [0, 0.25, -0.25]

"""
Parameters for filtering out data of the car going straight.
Track 1 in particular contains significant portions that are straight or have
minimal turning radius. Use of all such data biases the model towards driving
straight.
"""
steer_angle_thresh = 0.04
steer_keep_pct = 0.33 # pct of angles withn +/- steer_angle_thresh to keep


"""
The code below loops through each dataset and creates a list of images, which
includes individual entries for the left, right, and center camera images. This
list contains only the raw data; additional images are created in the generator
below, where each image is also flipped horizontally.

Another approach could have been to perform the image preprocessing outside
of the generator/model and actually save addition images. In this case, epochs didn't
take too long to train (<2 mins each using an Amazon AWS GPU instance), so I
chose to do the preprocessing in the pipeline. In the case where training time
is signifcantly larger, I might pre-process beforehand.
"""
samples = []
for dataset in datasets:
    tot_counter = 0; # Total number of entries
    dataset_counter = 0; # Number of entries per trainig set

    with open('data/' + dataset + '_log.csv') as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            tot_counter = tot_counter + 1

            # Filter out steering angles
            this_steer = float(line[3])
            if ((abs(this_steer) >= steer_angle_thresh) |
                ((abs(this_steer) < steer_angle_thresh) & (random() < steer_keep_pct))):

                # Center, left, and right images were in each Line; same steer angle
                for i in range(3): # center, left, right

                    sourcepath = line[i]

                    # Here, had too account for data collected on both my Mac
                    # and PC, where the direction of slashes is different.
                    if("/" in sourcepath):
                        filename = sourcepath.split('/')[-1]
                    else:
                        filename = sourcepath.split('\\')[-1]

                    # All data was stored in 'data' folder in project directory
                    # Not included in git repository due to size
                    newpath = 'data/' + dataset + '/' + filename
                    newline = [newpath, this_steer+steer_correct[i]]
                    samples.append(newline)
                    dataset_counter = dataset_counter + 1

    # Print some stats about dataset.
    print("Total number of images in", dataset, ": " , tot_counter*3)
    print("Number of images in", dataset, "after fitering : ", dataset_counter)

# Print some stats about full set of images.
print("Final number of images: ", len(samples))
print("NOTE: This does not include any image flipping, which may occur later!")



"""
Below is a generator for use in the Keras model. Each image that is loaded is
also flipped to remove any left/right bias that may have been inherent in the
data set.

NOTE: batch size must be an even number!
"""
# Split training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size = 64):

    num_samples = len(samples)
    """
    Because each image is being flipped horizontally below, dividing the batch
    size by 2 here, within this function, to compensate. Otherwise, batch size
    returned by the generator would be double the value that's passed to this
    function.
    """
    batch_size = int(batch_size/2);

    while 1:
        for offset in range(0, num_samples, batch_size):

            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                path = batch_sample[0]
                angle = batch_sample[1]

                image = cv2.imread(path)

                # Images are converted to YUV, as suggested in NVIDIA paper.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

                # Image and steering angle are flipped.
                images.append(image)
                images.append(cv2.flip(image,1)) #Flip image

                angles.append(float(angle))
                angles.append(float(-angle))

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)



# Training and validation generators
train_generator = generator(train_samples, batch_size=512)
validation_generator = generator(validation_samples, batch_size=512)


"""
NN set up below to match NVIDIA NN referenced in Udacity course, with some
differences. Most notably, the width of the input images, and thus, the size
of the output of the final convolution is different. This is discussed in
greater detail in the markdown.

Dropout was added throughout, with various magnitudes of dropout being applied
at different levels.
"""
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

dropout = { 'none': 0.0, 'some': 0.3, 'half': 0.5 }

def nvidia():

    model = Sequential()
    model.add(Cropping2D(cropping=((74,20), (30,30)), input_shape=(160, 320, 3))) # 66, 260
    model.add(Lambda(lambda x:(x/255.0)-0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(dropout['some']))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(dropout['some']))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(dropout['some']))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))
    model.add(Dropout(dropout['some'])) # consider less dropout here?
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))
    model.add(Dropout(dropout['none'])) # No dropout here; too few params
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Dropout(dropout['half']))
    model.add(Dense(100))
    model.add(Dropout(dropout['some']))
    model.add(Dense(50))
    model.add(Dropout(dropout['none'])) # No dropout here; too few params
    model.add(Dense(1))

    model.compile(loss = 'mse', optimizer = 'adam')
    model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*2,
                        validation_data=validation_generator, nb_val_samples=len(validation_samples)*2, nb_epoch=3)
    model.summary()
    model.save('model.h5')

nvidia()
