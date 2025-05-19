# MNIST
https://colab.research.google.com/drive/19pK2mNYbORLLlkxbtYK0sU9IKsS41JYP

## MNIST Lösungen
https://colab.research.google.com/drive/1sCxyt_1Mpw-u2z7cfUhEnAmMWRCG86ox
https://colab.research.google.com/drive/1GsmzMcACHkLGYiq3Gz7ArHVd17J12pso
https://colab.research.google.com/drive/1JKvDMk61jN550vdh_E2hZHlCeKLwfOlC
https://colab.research.google.com/drive/1C7mJnpfGurUSflaR8rkrJX3Hoayk-JZC#scrollTo=HPL__XLpKCH5

## Links
https://www.tensorflow.org/api_docs/python/tf/keras/losses
https://www.tensorflow.org/api_docs/python/tf/keras/utils/load_img

## Gesamter Import
```python
# TensorFlow & Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Optional: für erweiterte Bildverarbeitung (eigenes Bild etc.)
import tensorflow_datasets as tfds

# Visualisierung & Numerik
import matplotlib.pyplot as plt
import numpy as np

# Dateiverwaltung und Logging
import os
import time

# Für erweiterte Modelloptimierung (Keras Tuner)
import keras_tuner as kt

# Für Confusion Matrix und Evaluation
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

%load_ext tensorboard

```

## First Steps
### Libraries laden
```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
```
### TF Version prüfen
```python
tf.__version__
```
###  Laden des MNIST-Datensatzes
```python
mnist = tf.keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
```
- X_train_full: Bilder der Trainingsdaten (60.000 Graustufenbilder, 28x28 Pixel)
- y_train_full: Labels (Zahlen 0–9)
- X_test, y_test: Testdaten (10.000 Bilder)

### Normalisieren und Split in Validation/Train
```python
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.
```
- Datensätze werden in Trainings- und Validierungsdaten aufgeteilt.
- Werte der Pixel (0–255) werden auf 0–1 skaliert.

### Mehrere Bilder anzeigen (Trainingsdaten-Vorschau)
```python
n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis("off")
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
```
- Zeigt 40 Ziffern aus dem Trainingsdatensatz als Vorschau.

### Ein einzelnes Bild anzeigen
```python
plt.imshow(X_train[5], cmap="binary")
plt.axis("off")
plt.show()
```
### Anzeige der Labels
```python
y_train
```
- Gibt alle Labels des Trainingssets zurück.

## Create und train the Model
Implement an ANN with 2 Hidden Layers, the first with 128 neurons and "relu" activation functions. The second layers instead contains 64 neurons (also relu activation function). Don't forget the output layer :)
Also, remember that due to the 2D structure of your input data you need to first flatten them. Use the following specific layer at the very beginning inside your Sequential Model:
tf.keras.layers.Flatten(input_shape=(28, 28))

### Modell erstellen
```python
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),        # Input: 28x28 -> 784
    keras.layers.Dense(128, activation="relu"),        # Erste versteckte Schicht
    keras.layers.Dense(64, activation="relu"),         # Zweite versteckte Schicht
    keras.layers.Dense(10, activation="softmax")       # Output-Schicht (10 Klassen)
])
```
### Modellübersicht anzeigen
```python
model.summary()
```

Define now which cost function to minimize in the .compile method. Take "sgd" as optimizer and track the accuracy as metric.
For classification:Pay attention to the last layer if sigmoid or softmax is explicitly indicated, then from_logits = False
https://www.tensorflow.org/api_docs/python/tf/keras/losses

### Modell kompilieren
```python
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer="sgd", 
              metrics=["accuracy"])
```
### Modell trainieren (15 Epochen)
```python
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_valid, y_valid))
```
### Modell evaluieren (Testdaten)
```python
model.evaluate(X_test, y_test)
```
Let's now predict the digits for the first 3 images in the test set. Feel free to check more cases. First we predict the score associated which every category and then we find out which one is the one with highest score

### Vorhersage berechnen
```python
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
```
### Klassen mit höchster Wahrscheinlichkeit extrahieren
```python
y_pred = np.argmax(y_proba, axis=1)
y_pred

#if you are working with binary classification, use instead the following line:
#y_pred = (y_proba > 0.5).astype("int32")
```
### Bilder und Vorhersagen anzeigen
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis("off")
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
```
## Experimental test
The final test consist in a "physical" test. 
1. Draw on a piece of paper 1 single digit between 0 and 9. 
2. Take a picture with your webcam of it 
3. Test if the neural network can correctly identify also your handwriting

https://www.tensorflow.org/api_docs/python/tf/keras/utils/load_img

### Bild laden und vorbereiten (auf 28×28 Pixel, Graustufen, normalisiert)
```python
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np

# Pfad zu deinem Bild – z. B. von einem Upload oder Google Drive
img_path = "dein_bild.jpg"  # Pfad anpassen!

# Bild laden (Größe auf 28x28 ändern, farbmodus = 'grayscale')
img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")

# In ein Array umwandeln
img_array = image.img_to_array(img)

# Optional: Farben invertieren (falls dein Bild schwarzer Hintergrund / weiße Ziffer hat)
# img_array = 255 - img_array

# Normalisieren auf [0, 1]
img_array = img_array / 255.0

# Tensor-Form: (1, 28, 28, 1) → Modell erwartet Batch-Dimension
img_array = tf.expand_dims(img_array, 0)
```
### Vorhersage treffen
```python
predictions = model(img_array, training=False)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(np.argmax(score), 100 * np.max(score))
)

# Optional: Wahrscheinlichkeiten ausgeben
print(score)
```

## Plot the evolution of accuracy and loss
The historical values of accuracy and loss during training and validation are stored during training in the variable "history". 
You can access them through history.history. Use this information to plot and compare the evolution of accuracy and loss for training and validation
```python
plt.figure(figsize=(10, 3))

# Genauigkeit: Training vs. Validierung
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

# Verlust: Training vs. Validierung
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
```
### Code mit EarlyStopping
EarlyStopping-Callback ist ein nützliches Werkzeug, um das Training automatisch zu beenden, wenn sich die Validierungsleistung nicht weiter verbessert – ideal zur Vermeidung von Overfitting und zur Trainingszeit-Optimierung.
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping_cb = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping_cb]
)
```
### ModelCheckpoint
```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_cb = ModelCheckpoint(
    "best_model.h5",              # Dateiname, unter dem gespeichert wird
    save_best_only=True,          # nur speichern, wenn val_loss sich verbessert
    monitor='val_loss',           # kann auch 'val_accuracy' sein
    mode='min',                   # bei val_loss will man das Minimum
    verbose=1                     # Fortschritt anzeigen
)

```
### Training starten mit beiden Callbacks
```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping_cb, checkpoint_cb]
)
```
## Exercises_WeeK_7 gem. Solutions
### Setup
```bash
pip install tensorflow tensorflow-datasets keras keras-tuner
```
### Import
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow_datasets as tfds
import numpy as np
import os
import time
```
### MNIST-Netzwerk aufbauen
####  Classification of hand-written digits
```python
# Step 1
tf.__version__ # tf version prüfen
# Step 2
mnist = tf.keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
#X_train_full.shape

# Step 3
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.# we want to have the input between 0 and 1
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# Step 4
plt.imshow(X_train[5], cmap="binary")
plt.axis('off')
plt.show()

# Step 4
y_train
```
#### Create and train the model
```python
# Step 1
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),#my figures have size 28 pixels x 28 pixels # Flatten is needed for img
  tf.keras.layers.Dense(128, activation="relu"), # Dense produces a fully conencted layer. #w/o Flatten, the input is given here
  tf.keras.layers.Dense(64, activation="relu"),
  tf.keras.layers.Dense(10, activation="softmax")#we use softmax because we are dealing with a multi-class classification
])

# Step 2
model.summary()

# Step 3
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Step 4
history = model.fit(X_train, y_train, epochs=15,
                    validation_data=(X_valid, y_valid))# 1 epoch is a full pass over the whole training set

# Step 5
model.evaluate(X_test, y_test)

# Step 6
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

# Step 7
y_pred=np.argmax(y_proba,axis=1)

#if you are working with binary classification, use instead the following line:
#y_pred = (y_proba > 0.5).astype("int32")

y_pred

# Step 8
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
```
#### Experimental test
```python
# Step 1
img = tf.keras.utils.load_img("prova2.jpg", target_size=(28, 28), color_mode="grayscale")
img_array = 1 - tf.keras.utils.img_to_array(img) / 255.0
background = img_array < 0.5
img_array[background] = 0.0
img_array = tf.expand_dims(img_array, 0)

# Step 2
img_array.shape

# Step 3
plt.imshow(img_array[0,...], cmap="binary", interpolation="nearest")
plt.colorbar()

# Step 4
predictions = model(img_array, training = False)
score = tf.nn.softmax(predictions)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(np.argmax(predictions,axis=1), 100 * np.max(score))
)
print(score)
```
#### Plot
```python
plt.figure(figsize=(10,3))
# Plot training & validation accuracy values
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
```
### EMNIST Netzwerk
#### Import TensorFlow and dependencies
```python
# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.config.list_physical_devices())
```
#### Load and prepare the EMNIST Letters dataset
```python
# Load EMNIST Letters dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',  # Using letters variant
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Convert to numpy arrays
def prepare_dataset(dataset):
    images = []
    labels = []
    for image, label in dataset:
        # Labels in EMNIST are 1-indexed (1-26), subtract 1 to make 0-indexed
        images.append(image.numpy())
        labels.append(label.numpy() - 1)  # Convert to 0-25 range

    return np.array(images), np.array(labels)

# Get training and test data
train_images, train_labels = prepare_dataset(ds_train)
test_images, test_labels = prepare_dataset(ds_test)

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create a letter map for easier visualization (A-Z)
letters = [chr(ord('A') + i) for i in range(26)]

print('Training set shape:', train_images.shape)
print('Test set shape:', test_images.shape)
```
#### Explore the data
```python
# Step 1
# Display information about the dataset
print(f"Number of training examples: {len(train_images)}")
print(f"Number of test examples: {len(test_images)}")
print(f"Image shape: {train_images[0].shape}")
print(f"Number of classes: {len(np.unique(train_labels))}")

# Step 2
# Display information about the dataset
print(f"Number of training examples: {len(train_images)}")
print(f"Number of test examples: {len(test_images)}")
print(f"Image shape: {train_images[0].shape}")
print(f"Number of classes: {len(np.unique(train_labels))}")
```
#### Create and train the model
```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # EMNIST images are 28x28 with 1 channel
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dense(64, activation="relu"),
  tf.keras.layers.Dense(26, activation="softmax")  # 26 classes for letters A-Z
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_split=0.1)
```
#### Evaluate the model
```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```
#### Make predictions
```python
# Select a few test images
X_new = test_images[:3]

# Make predictions
y_proba = model.predict(X_new)
y_pred = np.argmax(y_proba, axis=1)

print("Predicted classes:", y_pred)
print("Predicted letters:", [letters[pred] for pred in y_pred])
print("Actual classes:", test_labels[:3])
print("Actual letters:", [letters[label] for label in test_labels[:3]])

plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
```
#### Plot the evolution of accuracy and loss
```python
plt.figure(figsize=(10,3))
# Plot training & validation accuracy values
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.show()
```
### Improve EMNIST
#### Import
```python
# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.config.list_physical_devices())

#setup folder where you will save logs for tensorflow
root_logdir = os.path.join(os.curdir,"my_logs_ML2")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

#create keras Tensorboard callback
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
```
#### Load and prepare the EMNIST Letters dataset
```python
# Load EMNIST Letters dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',  # Using letters variant
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Convert to numpy arrays
def prepare_dataset(dataset):
    images = []
    labels = []
    for image, label in dataset:
        # Labels in EMNIST are 1-indexed (1-26), subtract 1 to make 0-indexed
        images.append(image.numpy())
        labels.append(label.numpy() - 1)  # Convert to 0-25 range

    return np.array(images), np.array(labels)

# Get training and test data
train_images, train_labels = prepare_dataset(ds_train)
test_images, test_labels = prepare_dataset(ds_test)

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create a letter map for easier visualization (A-Z)
letters = [chr(ord('A') + i) for i in range(26)]

print('Training set shape:', train_images.shape)
print('Test set shape:', test_images.shape)
```
#### Explore the data
```python
# Display information about the dataset
print(f"Number of training examples: {len(train_images)}")
print(f"Number of test examples: {len(test_images)}")
print(f"Image shape: {train_images[0].shape}")
print(f"Number of classes: {len(np.unique(train_labels))}")

# Display information about the dataset
print(f"Number of training examples: {len(train_images)}")
print(f"Number of test examples: {len(test_images)}")
print(f"Image shape: {train_images[0].shape}")
print(f"Number of classes: {len(np.unique(train_labels))}")
```
#### Model
```python
# Step 1
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # EMNIST images are 28x28 with 1 channel
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer = tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer = tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(26, activation = 'softmax')  # 26 classes for letters A-Z
])

# Step 2
model.summary()

# Step 3
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Step 4
history = model.fit(train_images, train_labels, epochs=30,
                    validation_split=0.1,
                    callbacks=[tensorboard_cb])

# Step 5
%load_ext tensorboard
%tensorboard --logdir my_logs_ML2

# Step 6
#using the tf.summary API
test_logdir = get_run_logdir()
img = np.reshape(train_images[0:20], (-1, 28, 28, 1))
writer = tf.summary.create_file_writer(test_logdir)
with writer.as_default():

    tf.summary.image('my_images', img, step=0)

# Step 7
model.save("best_model_emnist.keras")
```
#### Evaluate the model
```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```
#### Make predictions
```python
# Select a few test images
X_new = test_images[:3]

# Make predictions
y_proba = model.predict(X_new)
y_pred = np.argmax(y_proba, axis=1)

print("Predicted classes:", y_pred)
print("Predicted letters:", [letters[pred] for pred in y_pred])
print("Actual classes:", test_labels[:3])
print("Actual letters:", [letters[label] for label in test_labels[:3]])

plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
```
#### Additional visualizations
```python
# Select the first 15 test images
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*num_cols, 2*num_rows))

# Get predictions for all test images
predictions = model.predict(test_images)

for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(letters[predicted_label],
                              letters[true_label]),
                              color=color)
plt.tight_layout()
plt.show()
```

#### Plot the evolution of accuracy and loss
```python
plt.figure(figsize=(10,3))
# Plot training & validation accuracy values
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.show()
```
### deep neural network
```python
pip install -q -U keras-tuner

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner as kt

print(tf.__version__)
print(tf.config.list_physical_devices())
```
#### Setup Tensorboard logging
```python
# Setup folder for tensorboard logs
root_logdir = os.path.join(os.curdir, "my_logs_ML2")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

# Create tensorboard callback with more metrics
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=run_logdir,
    histogram_freq=1,  # Log histogram of weights
    write_graph=True,  # Log model graph
    write_images=True,  # Log model weights as images
    update_freq='epoch',  # Update at the end of each epoch
    profile_batch='500,520'  # Profile performance for batches 500-520
)
```
#### Load and preprocess EMNIST Letters dataset
```python
# Load EMNIST Letters dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',  # Using letters variant
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Convert to numpy arrays
def prepare_dataset(dataset):
    images = []
    labels = []
    for image, label in dataset:
        # Labels in EMNIST are 1-indexed (1-26), subtract 1 to make 0-indexed
        images.append(image.numpy())
        labels.append(label.numpy() - 1)  # Convert to 0-25 range

    return np.array(images), np.array(labels)

# Get training and test data
train_images, train_labels = prepare_dataset(ds_train)
test_images, test_labels = prepare_dataset(ds_test)

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create a letter map for easier visualization (A-Z)
letters = [chr(ord('A') + i) for i in range(26)]

print('Training set shape:', train_images.shape)
print('Test set shape:', test_images.shape)
```
#### Explore the data
```python
# Step 1
# Display information about the dataset
print(f"Number of training examples: {len(train_images)}")
print(f"Number of test examples: {len(test_images)}")
print(f"Image shape: {train_images[0].shape}")
print(f"Number of classes: {len(np.unique(train_labels))}")

# Step 2
# Display first 9 images from the training set
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f'Label: {letters[train_labels[i]]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
```
#### Define the model architecture with KerasTune
```python
def build_model(hp):
    model = tf.keras.Sequential([
        # Flatten the input images
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),

        # First dense layer with tunable units (similar to original MNIST example)
        tf.keras.layers.Dense(
            units=hp.Int('dense_1_units', min_value=128, max_value=512, step=64),
            activation='relu',
            kernel_initializer=tf.keras.initializers.he_normal
        ),

        # Second dense layer with tunable units
        tf.keras.layers.Dense(
            units=hp.Int('dense_2_units', min_value=64, max_value=256, step=32),
            activation='relu',
            kernel_initializer=tf.keras.initializers.he_normal
        ),

        # Optional third dense layer
        tf.keras.layers.Dense(
            units=hp.Int('dense_3_units', min_value=32, max_value=128, step=32),
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(
                hp.Float('l2_reg', min_value=0.001, max_value=0.1, sampling='log')
            )
        ),

        # Dropout layer with tunable rate
        tf.keras.layers.Dropout(
            rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        ),

        # Output layer - 26 classes (A-Z) with softmax activation
        tf.keras.layers.Dense(26, activation='softmax')
    ])

    # Compile with tunable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-3, max_value=1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```
#### Setup KerasTune
```python
# Initialize the tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    directory='keras_tuner_dir',
    project_name='emnist_letters'
)

# Display search space summary
tuner.search_space_summary()
```
#### Create a learning rate scheduler for better convergence
```python
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * 1/10. #tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
```
#### Perform hyperparameter search
```python
# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Add model checkpoint callback
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_checkpoint.keras',
    save_best_only=True,
    monitor='val_accuracy'
)

# Perform the search
tuner.search(
    train_images, train_labels,
    validation_split=0.1,
    epochs=10,
    callbacks=[tensorboard_cb, early_stopping, checkpoint_cb]
)
```
#### Get the best model
```python
# Step 1
# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

# Build the best model
best_model = build_model(best_hps)
best_model.summary()

# Step 2
# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

# Build the best model
best_model = build_model(best_hps)
best_model.summary()
```
#### Evaluate the model
```python
# Evaluate on test set
test_loss, test_acc = best_model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.4f}')

# Save the best model
best_model.save('best_emnist_letters_model.keras')
```
#### Add weights and gradients to Tensorboard
```python
# Create a writer for the custom TensorBoard data
custom_writer = tf.summary.create_file_writer(os.path.join(run_logdir, 'custom_metrics'))

# Log sample images with predictions
with custom_writer.as_default():
    # Get sample images
    sample_images = test_images[:10]
    sample_labels = test_labels[:10]

    # Make predictions
    predictions = best_model.predict(sample_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Convert images to right format for TensorBoard (add batch dimension)
    images_for_tb = np.reshape(sample_images, (-1, 28, 28, 1))

    # Add titles with true and predicted labels
    titles = [f"True: {letters[true]}, Pred: {letters[pred]}"
              for true, pred in zip(sample_labels, predicted_labels)]

    # Log images with predictions as titles
    tf.summary.image('Test Predictions', images_for_tb, max_outputs=10, step=0)
```
#### Visualize results
```python
# Plot training history
plt.figure(figsize=(10, 3))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.show()
```
#### Make predictions and visualize results
```python
# Get some test samples
sample_indices = np.random.choice(len(test_images), 25, replace=False)
sample_images = test_images[sample_indices]
sample_labels = test_labels[sample_indices]

# Make predictions
predictions = best_model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# Visualize predictions
plt.figure(figsize=(15, 15))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(sample_images[i].squeeze(), cmap='gray')

    # Get confidence for the prediction
    confidence = predictions[i][predicted_labels[i]] * 100

    color = 'green' if predicted_labels[i] == sample_labels[i] else 'red'
    plt.title(f'True: {letters[sample_labels[i]]}\nPred: {letters[predicted_labels[i]]}\nConf: {confidence:.2f}%',
              color=color, fontsize=9)
    plt.axis('off')
plt.tight_layout()
plt.show()
```
#### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Make predictions on all test data
predictions = best_model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix - normalized version
plt.figure(figsize=(15, 12))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
            xticklabels=letters,
            yticklabels=letters)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(test_labels, predicted_labels,
                           target_names=letters))
```
#### Identify commonly confused letter pairs
```python
# Find the most commonly confused letter pairs
num_classes = len(letters)
confusion_pairs = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j:  # Skip the diagonal
            # How often class i is predicted as class j
            confusion_pairs.append({
                'true': letters[i],
                'predicted': letters[j],
                'count': cm[i, j],
                'rate': cm_normalized[i, j]
            })

# Sort by count (descending)
confusion_pairs = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)

# Display the top 10 most confused pairs
print("Top 10 most confused letter pairs:")
for i, pair in enumerate(confusion_pairs[:10]):
    print(f"{i+1}. True: {pair['true']}, Predicted: {pair['predicted']}, Count: {pair['count']}, Rate: {pair['rate']:.2f}")
```
# Week 9

## Lösungen Ex 2
https://colab.research.google.com/drive/1D7BNmvutywwZu3FsDlA9yL8KjOPSQrTo#scrollTo=n3dfOkkHqhMA
