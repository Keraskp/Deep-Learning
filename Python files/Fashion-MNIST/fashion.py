import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.simplefilter(action='ignore',category=FutureWarning)
from sklearn.model_selection import train_test_split
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


classes=10
epochs=20


if __name__ == '__main__':

	train_df=pd.read_csv('./fashion-mnist_train.csv',sep=',')
	test_df=pd.read_csv('./fashion-mnist_test.csv',sep=',')

	train_data=np.array(train_df,dtype='float32')
	test_data=np.array(test_df,dtype='float32')

	x_train=train_data[:,1:]/255
	y_train=train_data[:,0]

	x_test=test_data[:,1:]/255
	y_test=test_data[:,0]

	x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)

	class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
	plt.figure(figsize=(10, 10))
	for i in range(36):
	    plt.subplot(6, 6, i + 1)
	    plt.xticks([])
	    plt.yticks([])
	    plt.grid(False)
	    plt.imshow(x_train[i].reshape((28,28)))
	    label_index = int(y_train[i])
	    plt.title(class_names[label_index])
	plt.show()

	image_rows = 28
	image_cols = 28
	batch_size = 4096
	image_shape = (image_rows,image_cols,1) 

	x_train = x_train.reshape(x_train.shape[0],*image_shape)
	x_test = x_test.reshape(x_test.shape[0],*image_shape)
	x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)

	cnn_model = tf.keras.Sequential([
	    tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),
	    tf.keras.layers.MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
	    tf.keras.layers.Dropout(0.2),
	    tf.keras.layers.Flatten(), # flatten out the layers
	    tf.keras.layers.Dense(32,activation='relu'),
	    tf.keras.layers.Dense(10,activation = 'softmax')
	    
	])

	cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam',metrics =['accuracy'])

	history = cnn_model.fit(
	    x_train,
	    y_train,
	    batch_size=4096,
	    epochs=75,
	    verbose=1,
	    validation_data=(x_validate,y_validate),
	)

	plt.figure(figsize=(10, 10))

	plt.subplot(2, 2, 1)
	plt.plot(history.history['loss'], label='Loss')
	plt.plot(history.history['val_loss'], label='Validation Loss')
	plt.legend()
	plt.title('Training - Loss Function')

	plt.subplot(2, 2, 2)
	plt.plot(history.history['accuracy'], label='Accuracy')
	plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
	plt.legend()
	plt.title('Train - Accuracy')

	score = cnn_model.evaluate(x_test,y_test,verbose=0)
	print('Test Loss : {:.4f}'.format(score[0]))
	print('Test Accuracy : {:.4f}'.format(score[1]))