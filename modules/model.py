import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

def build_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(train_dir, val_dir, epochs=30):
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(train_dir, target_size=(256, 256), class_mode='categorical')
    val_gen = datagen.flow_from_directory(val_dir, target_size=(256, 256), class_mode='categorical')

    model = build_model(num_classes=len(train_gen.class_indices))

    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen)
    return model, history, train_gen.class_indices


#import numpy as np
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model, test_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate model on test data
    loss, acc = model.evaluate(test_gen, steps=len(test_gen))

    # Predict classes
    predictions = model.predict(test_gen, steps=len(test_gen))
    predicted_classes = np.argmax(predictions, axis=1)

    # True class labels from the generator
    true_classes = test_gen.classes

    return acc, predicted_classes, true_classes


# def evaluate_model(model, test_dir, class_indices):
#     datagen = ImageDataGenerator(rescale=1./255)
#     test_gen = datagen.flow_from_directory(test_dir, target_size=(256, 256), class_mode='categorical', shuffle=False)

#     loss, acc = model.evaluate(test_gen)
#     #predictions = model.predict(test_gen)
#     predictions = np.argmax(model.predict(test_gen), axis=1)
#     true_classes = test_gen.classes

#     predicted_classes = predictions.argmax(axis=1)
#     return acc, predicted_classes, true_classes

def fine_tune_model(model, train_dir, val_dir, fine_tune_at=-30, epochs=20):
    model.layers[0].trainable = True

    for layer in model.layers[0].layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(train_dir, target_size=(256, 256), class_mode='categorical')
    val_gen = datagen.flow_from_directory(val_dir, target_size=(256, 256), class_mode='categorical')

    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen)
    return model, history
