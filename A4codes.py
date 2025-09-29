#COMP3105 A4
#Kamal Yassin (101265070)
#Hussam Al Nabtiti (101267733)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Visualization Function
def plotImg(x):
    img = x.reshape((84, 28))  
    plt.imshow(img, cmap='gray')  
    plt.show()
    return

# Learn Function
def learn(X, y):
    X_images = X.reshape(-1, 84, 28, 1) / 255.0
    y_digits = to_categorical(y, num_classes=10)  
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(84, 28, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.1
    )
    batch_size = 64
    epochs = 50
    train_generator = datagen.flow(
        X_images, y_digits,
        batch_size=batch_size,
        subset='training'
    )
    validation_generator = datagen.flow(
        X_images, y_digits,
        batch_size=batch_size,
        subset='validation'
    )
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-5)
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    return model

# Classify Function
def classify(Xtest, model):
    X_images = Xtest.reshape(-1, 84, 28, 1)/ 255.0
    predictions = model.predict(X_images)
    yhat = np.argmax(predictions, axis=1)
    return yhat

# Main Program
if __name__ == "__main__":
    csv = pd.read_csv('A4data/A4train.csv', header=None)
    data = csv.to_numpy()
    y = data[:, 0]
    X = data[:, 1:]
    print("Training the model...")
    model = learn(X, y)

    csv = pd.read_csv('A4data/A4val.csv', header=None)
    data = csv.to_numpy()
    ytest = data[:, 0]
    Xtest = data[:, 1:]

    print("Classifying the test data...")
    predictions = classify(Xtest, model)
    accuracy = 100 * np.mean(predictions == ytest)
    print(f"Accuracy: {accuracy:.2f}%")