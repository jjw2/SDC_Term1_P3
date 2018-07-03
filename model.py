import csv
import cv2
import numpy as np

lines = []
with open('data/Train1_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader: 
		lines.append(line)

images = []
measurements = []
for line in lines:
	sourcepath = line[0]
	filename = sourcepath.split('/')[-1]
	currentpath = 'data/Train1/' + filename
	image = cv2.imread(currentpath)
	images.append(image)
	
	measurement = float(line[3])
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)

## Set Up NN ##


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')

 
