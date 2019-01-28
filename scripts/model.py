import settings
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

from tensorflow.python.keras.callbacks import TensorBoard

import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


class DigitClassifier:

	def train_model(self, X_train, y_train):
		start_time = time.time()

		# model = keras.models.Sequential([
		# 	tf.keras.layers.Flatten(input_shape=(28, 28)),
		# 	tf.keras.layers.Dense(512, activation=tf.nn.relu),
		# 	tf.keras.layers.Dropout(0.2),
		# 	tf.keras.layers.Dense(10, activation=tf.nn.softmax)
		# ])

		MODEL_SETTINGS = {
			'input_shape': (28, 28, 1),

			'conv1_fmaps': 32,
			'conv1_kernel_size': 3,
			'conv1_strides': 1,
			'conv1_pad': 'SAME',

			'pool2_kernel_size': 2,

			'conv3_fmaps': 64,
			'conv3_kernel_size': 3,
			'conv3_strides': 2,
			'conv3_pad': 'SAME',

			'pool4_kernel_size': 2,

			'num_classes': 10
		}


		model = Sequential([
			Conv2D(input_shape=	MODEL_SETTINGS['input_shape'],
				   filters=		MODEL_SETTINGS['conv1_fmaps'],
				   kernel_size=	MODEL_SETTINGS['conv1_kernel_size'],
				   strides=		MODEL_SETTINGS['conv1_strides'],
				   padding=		MODEL_SETTINGS['conv1_pad'],
				   activation=	'relu',
				   name=		'conv_1'),
			MaxPooling2D(pool_size=MODEL_SETTINGS['pool2_kernel_size'], name='pool_2'),
			Conv2D(filters=		MODEL_SETTINGS['conv3_fmaps'],
				   kernel_size=	MODEL_SETTINGS['conv3_kernel_size'],
				   strides=		MODEL_SETTINGS['conv3_strides'],
				   padding=		MODEL_SETTINGS['conv3_pad'],
				   activation=	'relu',
				   name=		'conv_3'),
			MaxPooling2D(pool_size=MODEL_SETTINGS['pool4_kernel_size'], name='pool_4'),
			Flatten(),
			Dense(128, activation='relu', name='dense_1'),
			Dropout(0.5),
			Dense(MODEL_SETTINGS['num_classes'], activation='softmax', name='dense_2')
		])

		model.compile(optimizer='adam',
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])

		tensorboard = TensorBoard(log_dir=settings.lOGS_DIR)

		history = model.fit(X_train, y_train, epochs=15, callbacks=[tensorboard])
		end_time = time.time()
		print('Total train time = ', round(end_time - start_time), 's')

		self._visualize_model_training(history)
		return model

	def evaluate_model(self, model, X_test, y_test):
		y_test_predict = model.predict(X_test)
		predicted_label = []
		for y_test_i in y_test_predict:
			label = np.argmax(y_test_i)
			predicted_label.append(label)

		self._calculate_model_metrics(y_test, predicted_label)
		# test_loss, test_acc = model.evaluate(X_test, y_test)
		# print('Test accuracy = ', test_acc)

	@staticmethod
	def _calculate_model_metrics(y, y_pred):
		print('Calculating metrics...')
		labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		precision, recall, fscore, support = precision_recall_fscore_support(
			y, y_pred,
			labels=labels)

		precision = np.reshape(precision, (10, 1))
		recall = np.reshape(recall, (10, 1))
		fscore = np.reshape(fscore, (10, 1))
		data = np.concatenate((precision, recall, fscore), axis=1)
		df = pd.DataFrame(data)
		df.columns = ['Precision', 'Recall', 'Fscore']
		print(df)

		print('\n Average values')
		print('Precision = ', df['Precision'].mean())
		print('Recall = ', df['Recall'].mean())
		print('F1 score = ', df['Fscore'].mean())

	@staticmethod
	def _visualize_model_training(history):
		print(history.history.keys())
		plt.plot(history.history['acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('Number of epoch')
		plt.legend(['train'], loc='upper left')
		plt.show()
