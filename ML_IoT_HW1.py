#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


## OVERALL PROCESS: image → layer1 → layer2 → layer3 → output 

#Model1 = BASIC NEURAL NETWORK

def build_model1():
# Model1 is the baseline classifier (he simplest neural network that can reasonably attempt CIFAR-10 image classification)
# flatten 32×32×3 image → 3072-element vector (ONLY MODEL1)

# image → flatten → dense → dense → dense → output

# - Flatten (input processing layer)
# - Dense(128) + LeakyReLU (hidden layer 1)
# - Dense(128) + LeakyReLU (hidden layer 2)
# - Dense(128) + LeakyReLU (hidden layer 3)
# - Dense(output layer with 10 neurons)

# A logit is the numerical output of a neuron

# Flattening converts the image from a 2D spatial grid with pixel coordinates into a 1D list of values, so the model keeps the pixel intensities but loses all information about which pixels are neighbors and how they are arranged in space.
# This is exactly the inefficiency in layer 1
  N_units_1 = 128

  model = Sequential([
    layers.Flatten(input_shape = (32,32,3)), #input processing layer

    layers.Dense(N_units_1),
    layers.LeakyReLU(),

    layers.Dense(N_units_1),
    layers.LeakyReLU(),

    layers.Dense(N_units_1),
    layers.LeakyReLU(),

    layers.Dense(10)
  ])

  model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
  )
  return model

def build_model2():
  model = Sequential([
    keras.Input(shape=(32,32,3)),

    #2 convolution blocks (with strides)
    #
    layers.Conv2D(32, 3, strides=2, padding = "same", activation = "relu"),
    layers.BatchNormalization(),

    layers.Conv2D(64, 3, strides=2, padding = "same", activation = "relu"),
    layers.BatchNormalization(),
    # stop striding so later layers can refine features without destroying spatial detail

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    #FLATTEN feature maps (8x8x128) into a vector
    layers.Flatten(),

    #output layer of 10 neurons
    layers.Dense(10)

  ])

  model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
  )
  return model

def build_model3():
  #conv → conv → conv blocks → flatten → output
  model = Sequential([
  keras.Input(shape=(32,32,3)),

    #2 convolution blocks (with strides)
    #
    layers.Conv2D(32, 3, strides=2, padding = "same", activation = "relu"),
    layers.BatchNormalization(),

    layers.SeparableConv2D(64, 3, strides=2, padding = "same", activation = "relu"),
    layers.BatchNormalization(),
    # stop striding so later layers can refine features without destroying spatial detail

    layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    #FLATTEN feature maps (8x8x128) into a vector
    layers.Flatten(),

    #output layer of 10 neurons
    layers.Dense(10)

  ])

  model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
  )

  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():

  model = Sequential([
    keras.Input(shape=(32,32,3)),

    layers.SeparableConv2D(32, 3, strides=2, padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.SeparableConv2D(64, 3, strides=2, padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.SeparableConv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.SeparableConv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    # parameter saver
    layers.GlobalAveragePooling2D(),

    layers.Dense(10)
  ])

  model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
  )

  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import


if __name__ == '__main__':

    ########################################
    # Load CIFAR-10
    ########################################
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    # normalize to [0,1]
    train_images = train_images.astype("float32") / 255.0
    test_images  = test_images.astype("float32") / 255.0

    # validation split (last 5000 from training)
    val_images = train_images[-5000:]
    val_labels = train_labels[-5000:]

    train_images = train_images[:-5000]
    train_labels = train_labels[:-5000]

    print("Train:", train_images.shape)
    print("Val:", val_images.shape)
    print("Test:", test_images.shape)


    ########################################
    # Model 1 — Dense baseline
    ########################################
    model1 = build_model1()
    model1.summary()

    history1 = model1.fit(
        train_images, train_labels,
        epochs=30,
        batch_size=64,
        validation_data=(val_images, val_labels),
        verbose=2
    )

    test_loss1, test_acc1 = model1.evaluate(test_images, test_labels, verbose=0)
    print("Model1 test accuracy:", test_acc1)


    ########################################
    # Model 2 — CNN
    ########################################
    model2 = build_model2()
    model2.summary()

    history2 = model2.fit(
        train_images, train_labels,
        epochs=30,
        batch_size=64,
        validation_data=(val_images, val_labels),
        verbose=2
    )

    test_loss2, test_acc2 = model2.evaluate(test_images, test_labels, verbose=0)
    print("Model2 test accuracy:", test_acc2)


    ########################################
    # Model 3 — Separable CNN
    ########################################
    model3 = build_model3()
    model3.summary()

    history3 = model3.fit(
        train_images, train_labels,
        epochs=30,
        batch_size=64,
        validation_data=(val_images, val_labels),
        verbose=2
    )

    test_loss3, test_acc3 = model3.evaluate(test_images, test_labels, verbose=0)
    print("Model3 test accuracy:", test_acc3)


    ########################################
    # Model 50k — constrained model
    ########################################
    model50k = build_model50k()
    model50k.summary()

    history50k = model50k.fit(
        train_images, train_labels,
        epochs=30,
        batch_size=64,
        validation_data=(val_images, val_labels),
        verbose=2
    )

    test_loss50k, test_acc50k = model50k.evaluate(test_images, test_labels, verbose=0)
    print("Model50k test accuracy:", test_acc50k)

    # required by assignment
    model50k.save("best_model.h5")


    ########################################
    # Final summary printout for report
    ########################################
    print("\n=== FINAL TEST ACCURACIES ===")
    print("Model1:", test_acc1)
    print("Model2:", test_acc2)
    print("Model3:", test_acc3)
    print("Model50k:", test_acc50k)

  