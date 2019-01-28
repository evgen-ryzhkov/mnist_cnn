import tensorflow as tf
mnist = tf.keras.datasets.mnist


class MNISTData:

	def get_train_and_test_data(self):
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		X_train = self._prepare_x_data(X_train)
		X_test = self._prepare_x_data(X_test)
		return X_train, y_train, X_test, y_test

	@staticmethod
	def _prepare_x_data(X):
		# regularization
		X_prepared = X / 255

		# preparing input data for CNN (need 4 demensions)
		# last is number of channels. For greyscale img - 1
		n, img_width, img_height = X.shape
		X_prepared = X_prepared.reshape(n, img_width, img_height, 1)

		return X_prepared

