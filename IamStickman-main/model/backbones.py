from blocks import residual_block, inverted_residual_block, conv_block, up_sample

import tensorflow as tf
import sys

def create_vgg(x):
	x = tf.keras.layers.Conv2D(16,
		kernel_size=7,
		strides=4,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(32,
		kernel_size=3,
		strides=2,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(32,
		kernel_size=3,
		strides=1,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	return x

def unet(x):
# Conv Down
	# Première couche
	x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation("relu")(x)
	skip1 = tf.keras.layers.MaxPool2D((2, 2))(x)
	

	# Seconde Couche
	x = tf.keras.layers.Conv2D(128, 3, padding="same")(skip1)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation("relu")(x)
	skip2 = tf.keras.layers.MaxPool2D((2, 2))(x)

	# Troixième Couche
	x = tf.keras.layers.Conv2D(256 , 3, padding="same")(skip2)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation("relu")(x)
	skip3 = tf.keras.layers.MaxPool2D((2, 2))(x)

	# Quatrièeme Couche
	x = tf.keras.layers.Conv2D(512, 3, padding="same")(skip3)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation("relu")(x)
	skip4 = tf.keras.layers.MaxPool2D((2, 2))(x)

	# Bottle neck
	x = tf.keras.layers.Conv2D(512, kernel_size = 3, padding="same", strides=1, activation="relu")(skip4)
	x = tf.keras.layers.Conv2D(512, kernel_size = 3, padding="same", strides=1, activation="relu")(x)

# Conv Up
	# Première couch
	x = tf.keras.layers.UpSampling2D((2, 2))(x)
	x = tf.keras.layers.Concatenate(axis= 0)([x, skip4])
	x = tf.keras.layers.Conv2D(512, kernel_size = 3, padding="same", strides=1, activation="relu")(x)
	x = tf.keras.layers.Conv2D(512, kernel_size = 3, padding="same", strides=1, activation="relu")(x)

	# Deuxième couche
	x = tf.keras.layers.UpSampling2D((2, 2))(x)
	x = tf.keras.layers.Concatenate(axis= 0)([x, skip3])
	x = tf.keras.layers.Conv2D(256, kernel_size = 3, padding="same", strides=1, activation="relu")(x)
	x = tf.keras.layers.Conv2D(256, kernel_size = 3, padding="same", strides=1, activation="relu")(x)

	# Troisième couche
	x = tf.keras.layers.UpSampling2D((2, 2))(x)
	x = tf.keras.layers.Concatenate(axis= 0)([x, skip2])
	x = tf.keras.layers.Conv2D(128, kernel_size = 3, padding="same", strides=1, activation="relu")(x)
	x = tf.keras.layers.Conv2D(128, kernel_size = 3, padding="same", strides=1, activation="relu")(x)

	# Quatrième couche
	x = tf.keras.layers.UpSampling2D((2, 2))(x)
	x = tf.keras.layers.Concatenate(axis= 0)([x, skip1])
	x = tf.keras.layers.Conv2D(64, kernel_size = 3, padding="same", strides=1, activation="relu")(x)
	x = tf.keras.layers.Conv2D(64, kernel_size = 3, padding="same", strides=1, activation="relu")(x)

	# Sortie du reseau 
	x = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)

	return x


def creat_identity(x):
    return x

possible_backbones = {
	'VGG':create_vgg,
    'IDENTITY': creat_identity,
}
