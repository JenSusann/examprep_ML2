# Week 8

## Lösung
https://colab.research.google.com/drive/1mHsobuqipdgnZOAvQXQpKZ2RaNfthPsv#scrollTo=xvwvpA64CaW_

## Training a deep (convolutional) neural network on CIFAR10
```python
# TensorFlow and tf.keras
import tensorflow as tf
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import time
import PIL.Image as Image
from IPython.display import Image, display
import matplotlib.cm as cm

print(tf.__version__)
print(tf.config.list_physical_devices())
```
## Ex. 1: Code Convolutions
```python
img =  tf.keras.utils.load_img('yourImage.jpeg', target_size=(128,128))
data =  tf.keras.utils.img_to_array(img)/255.0
samples = tf.expand_dims(data, 0)


plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

img =  tf.keras.utils.load_img('yourImage.jpeg', target_size=(128,128))
data =  tf.keras.utils.img_to_array(img)/255.0
samples = tf.expand_dims(data, 0)

channels = 3
```
### To-Do-1
Create a 7x7 vertical filter and then a horizontal filter
ATTENTION: Convolutions in keras are 4D, meaning [num_pixels, num_pixels, num_channels, num_filters]

#define the structure of the filter
filter = #ADD YOUR CODE

#assign the right numerical values to the array (filters weights)
filter[#ADD YOUR CODE
    ] = #ADD YOUR CODE

```python
# Wir definieren einen vertikalen Kantenfilter (z. B. Sobel-ähnlich oder einfacher Kantenfilter)
# Ein einfacher vertikaler Kantenfilter (3x3)
kernel = np.array([[ -1, 0, 1],
                   [ -2, 0, 2],
                   [ -1, 0, 1]])

# Erweitern auf 3 Kanäle (RGB): gleicher Filter wird für alle Kanäle verwendet
# Die Form muss (filter_height, filter_width, in_channels, out_channels) sein
filter = np.zeros((7, 7, channels, 1), dtype=np.float32)
#assign the right numerical values to the array (filters weights)
filter[:, 3, :, :] = 1  # vertical line
#filter[3, :, :, :] = 1  # horizontal line
```

```python
outputs = tf.nn.conv2d(samples, filter, strides=1, padding="SAME")

plt.imshow(outputs[0, :, :, 0], cmap="gray") # plot image's 1nd feature map
plt.axis("off")
plt.show()
```
## Ex. 2: CNN on CIFAR 10
```python
#setup folder where you will save logs for tensorflow
root_logdir = os.path.join(os.curdir,"my_logs_ML2_CIFAR")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

# Import the CIFAR 10 library, split into train, validation and test images.
cifar = tf.keras.datasets.cifar10

(train_full_images, train_full_labels), (test_images, test_labels) = cifar.load_data()

valid_images, train_images = train_full_images[:5000], train_full_images[5000:]
valid_labels, train_labels = train_full_labels[:5000], train_full_labels[5000:]
test_images = test_images

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Check out some properties of the imported dataset
print('training ds shape:', train_images.shape)
print('validation ds shape:', valid_images.shape)
print('test ds shape:', test_images.shape)

print("labels: ", train_labels)

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

# Normalization of data between 0 and 1
train_images = train_images / 255.0

valid_images = valid_images / 255.0

test_images = test_images / 255.0
run_logdir = get_run_logdir()

```
### To-Do-2.1
create keras Tensorboard callback and early stopping_callback
```python
# TensorBoard & EarlyStopping Callback
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=run_logdir)
earlystopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# TensorBoard & EarlyStopping Callback V2
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
earlystopping_cb  = tf.keras.callbacks.EarlyStopping(patience = 3, restore_best_weights = True)

# CNN-Modell
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 Klassen
])

# CNN-Modell V2
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer = tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
```

```python
model.summary()
```
### To-Do-2.2
```python
# Kompilieren
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Kompilieren V2
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.99),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Trainieren
history = model.fit(
    train_images, train_labels,
    epochs=50,
    validation_data=(valid_images, valid_labels),
    callbacks=[tensorboard_cb, earlystopping_cb]
)

# Trainieren V2 
model.fit(train_images, train_labels, epochs=20, validation_data=(valid_images, valid_labels), callbacks=[tensorboard_cb, earlystopping_cb])
```
```python
%load_ext tensorboard
%tensorboard --logdir my_logs_ML2_CIFAR
```

### Evaluate accuracy
```python
# Step 1
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Step 2
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[int(true_label)]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[int(true_label)].set_color('blue')

  # Step 3
  predictions = model.predict(test_images)

  # Step 4
  i = 0
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.show()
```
### Use the trained model
```python
img = tf.keras.utils.load_img("./test_image.jpg", target_size=(32,32)#download your own image
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Add the image to a batch where it's the only member.


predictions = model.predict(img_array)
score = predictions[0]

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
```
# Week 9 – Deep Learning & Image Processing (Full Implementation)

## Overview
This notebook contains the **fully working implementation** of the Week 9 exercises from the Deep Learning course. It includes:

- Data Augmentation with TensorFlow
- Smile vs No Smile Classification with MobileNetV2
- Vision Transformer (ViT) and DINOv2 Transfer Learning on the Genki4k Dataset using HuggingFace
- Concepts of Attention and Transformer models

---

## Installation
```bash
pip install tensorflow matplotlib numpy transformers datasets torch scikit-learn Pillow
```

---

## 🧪 Exercise 1: Data Augmentation with tf.keras
```python
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomCrop
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load sample image
img_path = 'sample_image.jpg'  # Replace with your image path
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Define augmentation pipeline
augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.1),
    RandomContrast(0.2),
    RandomCrop(200, 200)
])

# Visualize results
plt.figure(figsize=(10, 6))
for i in range(6):
    aug_img = augmentation(img_array)
    ax = plt.subplot(2, 3, i + 1)
    plt.imshow(tf.cast(aug_img[0], tf.uint8))
    plt.axis("off")
plt.suptitle("Exercise 1: Data Augmentation")
plt.tight_layout()
plt.show()
```

---

## Exercise 2: Smile Detection with MobileNetV2
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Data preparation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory('genki4k',
                                         target_size=(224, 224),
                                         subset='training',
                                         class_mode='binary')
val_data = datagen.flow_from_directory('genki4k',
                                       target_size=(224, 224),
                                       subset='validation',
                                       class_mode='binary')

# Model building
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data,
          validation_data=val_data,
          epochs=10,
          callbacks=[EarlyStopping(patience=3)])
```

---

## Exercise 3: Attention Mechanisms (Conceptual)
Watch:
- [Attention in Transformers – 3Blue1Brown](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [How Transformers Work](https://www.youtube.com/watch?v=wjZofJX0v4M)

Try: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

---

## Exercise 4: Transfer Learning with ViT (HuggingFace)
```python
from datasets import load_dataset, DatasetDict
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import torchvision.transforms as T
from PIL import Image
import os
import torch

# Load Genki4k images manually
def load_images_from_folder(folder, label):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            images.append({'image': Image.open(path).convert("RGB"), 'label': label})
    return images

data = load_images_from_folder('genki4k/smile', 1) + load_images_from_folder('genki4k/no_smile', 0)

dataset = DatasetDict({"train": data})

# Feature extractor
extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def transform(example):
    return extractor(images=example['image'], return_tensors="pt")

# Use trainer with custom collate function
def collate_fn(batch):
    pixel_values = torch.stack([extractor(img['image'], return_tensors="pt")['pixel_values'][0] for img in batch])
    labels = torch.tensor([img['label'] for img in batch])
    return {"pixel_values": pixel_values, "labels": labels}

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)

training_args = TrainingArguments(
    output_dir="./vit-genki4k",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    tokenizer=extractor,
    data_collator=collate_fn
)

# trainer.train()  # Uncomment to run training
```

---

##  Exercise 5: DINOv2 on Genki4k
```python
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base", num_labels=2)
extractor = AutoFeatureExtractor.from_pretrained("facebook/dinov2-base")

# Reuse training_args, dataset, collate_fn from Exercise 4

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    tokenizer=extractor,
    data_collator=collate_fn
)

# trainer.train()  # Uncomment to run training
```

---


