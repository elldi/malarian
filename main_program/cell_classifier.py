import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model

img_width, img_height = 80,80
num_epoch = 5
activation_type = 'linear'
save_loc = "./basic_cnn_"+activation_type+"_"+str(num_epoch)+"_epochs.h5"


def create_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation(activation_type))

	return model

def train(model, train_generator, validation_generator):
	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

	model.fit_generator(
	        train_generator,
	        steps_per_epoch=70,
	        epochs=num_epoch,
	        validation_data=validation_generator,
        	validation_steps=70)

def load_pretrained_model(path):
	model = create_model()
	
	model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

	model.load_weights(path)
	return model

def start():
	training_data = "../train_2"
	validation_data = "../train_2"

	datagen = ImageDataGenerator(
        rescale=1./255)        # normalize pixel values to [0,1]
        # shear_range=0.2,       # randomly applies shearing transformation
        # zoom_range=0.2,        # randomly applies shearing transformation
        # horizontal_flip=True)  # randomly flip the images

	train_generator = datagen.flow_from_directory(
	        training_data,
	        target_size=(img_width, img_height),
	        batch_size=16,
	        class_mode='binary')

	validation_generator = datagen.flow_from_directory(
	        validation_data,
	        target_size=(img_width, img_height),
	        batch_size=32,
	        class_mode='binary')


	model = create_model()
	train(model, train_generator, validation_generator)

	model.save_weights(save_loc)
	print(model.evaluate_generator(validation_generator, 70))


def test_model():
	model = load_pretrained_model(save_loc)

	class1 = 0 
	class2 = 0
	counter = 0

	files = os.listdir("../train_2/uninfected/")
	for data in files:
		if(".jpg" in data):
			img = image.load_img("../train_2/uninfected/"+data, target_size= (img_width,img_height))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis = 0)
			x = preprocess_input(x)



			prediction = model.predict(x, verbose=0)
			print(str(counter) + ":" + str(prediction))
			if(prediction[0][0] < 0.5):
				class1+= 1
			else:
				class2+=1
			counter+=1

	print("Class 1: "+str(class1) + "/" + str(counter))
	print("Class 2: "+str(class2) + "/" + str(counter))

def predict_me(model, img):
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis = 0)
	x = preprocess_input(x)

	prediction = model.predict_classes(x, verbose=0)

	return prediction



# start()
# test_model()



