import csv
import cv2
import numpy as np


images = []
measurements = []
datasets = ['Train1', 'Train2']

for dataset in datasets:

	lines = []
	with open('data/' + dataset + '_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader: 
			lines.append(line)


	steer_correct = [0.2, 0, -0.2]
	for line in lines:
		for i in range(3):
			sourcepath = line[i]
			filename = sourcepath.split('/')[-1]
			currentpath = 'data/' + dataset + '/' + filename
			image = cv2.imread(currentpath)
			images.append(image)
			images.append(cv2.flip(image,1)) #Flip image
	
			measurement = float(line[3]) + steer_correct[i]
			measurements.append(measurement)
			measurements.append(-measurement)



X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)

## Set Up NN ##


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x:(x/255.0)-0.5))
model.add(Convolution2D(10, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(20, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, batch_size=32)

model.save('model.h5')

 
