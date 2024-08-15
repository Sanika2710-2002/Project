#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import ops
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import scipy.io
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.image import extract_patches


# In[19]:


# Define paths
path_annot = 'D:\INTERNSHIP\DAiSEE\DAiSEE\labels'  # Path to the annotations directory
path_videos = 'D:\INTERNSHIP\DAiSEE\DAiSEE\DataSet\Validation\826382\826382028\extracted_frames'  # Path to the video files directory

# Function to load annotation files
def load_annotation(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    return pd.read_csv(file_path)

# Load annotation files with error handling
try:
    train_labels = load_annotation(os.path.join(path_annot, 'TrainLabels.csv'))
    val_labels = load_annotation(os.path.join(path_annot, 'ValidationLabels.csv'))
    test_labels = load_annotation(os.path.join(path_annot, 'TestLabels.csv'))
except FileNotFoundError as e:
    print(e)
    exit(1)

# Clean column names (remove trailing spaces)
train_labels.columns = train_labels.columns.str.strip()

# Debug: Print columns and first few rows to inspect the DataFrame
print("Train Labels Columns: ", train_labels.columns)
print("First few rows of Train Labels:\n", train_labels.head())

# Function to extract a frame from a video
def extract_frame(video_path, frame_number=0):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file {video_path} cannot be opened.")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Frame {frame_number} not found in video {video_path}.")
    
    frame = cv2.resize(frame, (224, 224))  # Resize to a fixed size
    frame = frame / 255.0  # Normalize to [0, 1]
    return frame

# Example: Load and preprocess training images
train_images = []
train_labels_list = []

# Process each row in the DataFrame
for idx, row in train_labels.iterrows():
    video_path = os.path.join(path_videos, row['ClipID'])
    try:
        frame = extract_frame(video_path)
        train_images.append(frame)
        train_labels_list.append(row[['Engagement', 'Frustration', 'Confusion', 'Boredom']].values)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        continue

# Convert to numpy arrays
import numpy as np
train_images = np.array(train_images)
train_labels_list = np.array(train_labels_list)

# Display the shape of the data
print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels_list.shape}")


# In[3]:


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# In[4]:


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# In[6]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Ensure the data has the right shape
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Define the Patches layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        return patches

# Example usage
patch_size = 32
patches_layer = Patches(patch_size)
patches = patches_layer(np.expand_dims(x_train[0], axis=0))
print(f"Patches shape: {patches.shape}")


# In[20]:


# Assuming x_train is defined as in the previous example
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Ensure the data has the right shape
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Parameters
patch_size = 32  # Size of the patches to be extracted from the input images

# Display the original image
plt.figure(figsize=(4, 4))
plt.imshow(x_train[0].astype("uint8"), cmap="gray")
plt.axis("off")
plt.title("Original Image")
plt.show()

# Extract patches
patches = extract_patches(images=np.expand_dims(x_train[0], axis=0),
                          sizes=[1, patch_size, patch_size, 1],
                          strides=[1, patch_size, patch_size, 1],
                          rates=[1, 1, 1, 1],
                          padding='VALID')

# Print information about patches
print(f"Image size: {x_train[0].shape[0]} X {x_train[0].shape[1]}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"{patches.shape[1]} patches per image \n{patches.shape[-1]} elements per patch")

# Visualize extracted patches
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = patch.numpy().astype("uint8")
    plt.imshow(patch_img.squeeze(), cmap="gray")
    plt.axis("off")
    ax.set_title(f"Patch {i + 1}")

plt.suptitle("Extracted Patches")
plt.tight_layout()
plt.show()


# In[9]:


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded


# In[10]:


def create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
):
    inputs = keras.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)

    bounding_box = layers.Dense(4)(
        features
    )  # Final four neurons that output bounding box

    # return Keras model.
    return keras.Model(inputs=inputs, outputs=bounding_box)


# In[13]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.losses import MeanSquaredError

# Function to create your Vision Transformer model
def create_vit_object_detector(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers, mlp_head_units):
    model = tf.keras.Sequential([
        # Example layers (replace with your ViT model)
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Function to run the experiment
def run_experiment(model, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train):
    optimizer = tf.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Compile model
    model.compile(optimizer=optimizer, loss=MeanSquaredError())

    checkpoint_filepath = "vit_object_detector.weights.h5"
    checkpoint_callback = callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    # Ensure x_train and y_train are NumPy arrays or TensorFlow tensors
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Run training with validation split
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback,
            callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )

    return history

# Example usage and configuration
image_size = 224
patch_size = 32
input_shape = (image_size, image_size, 3)
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 100
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4
mlp_head_units = [2048, 1024, 512, 64, 32]

# Dummy data placeholders (replace with actual data loading)
x_train = np.random.rand(100, image_size, image_size, 3)
y_train = np.random.rand(100, 1)

# Create and run the experiment
vit_object_detector = create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
)

history = run_experiment(
    vit_object_detector,
    learning_rate,
    weight_decay,
    batch_size,
    num_epochs,
    x_train,
    y_train
)


# In[14]:


def run_experiment(model, learning_rate, weight_decay, batch_size, num_epochs):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Compile model.
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    checkpoint_filepath = "vit_object_detector.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback,
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )

    return history


input_shape = (image_size, image_size, 3)  # input image shape
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 100
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4
mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers


history = []
num_patches = (image_size // patch_size) ** 2

vit_object_detector = create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
)

# Train model
history = run_experiment(
    vit_object_detector, learning_rate, weight_decay, batch_size, num_epochs
)


def plot_history(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_history("loss")


# In[18]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Example: Define image_size and load x_test, y_test appropriately
image_size = 224
x_test = 'D:\INTERNSHIP\DAiSEE\DAiSEE\DataSet' # Load your test images
y_test = 'D:\INTERNSHIP\DAiSEE\DAiSEE\DataSet\Test'  # Load corresponding ground truth bounding boxes

# Function to resize image and predict bounding boxes
def predict_and_plot(input_image, model, true_box):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    im = input_image.copy()

    # Display the image
    ax1.imshow(im.astype("uint8"))
    ax2.imshow(im.astype("uint8"))

    # Resize and preprocess input_image
    input_image = cv2.resize(input_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    input_image = input_image.astype(np.float32) / 255.0  # Normalize if needed

    # Predict bounding box
    preds = model.predict(input_image)[0]

    (h, w) = im.shape[0:2]

    top_left_x, top_left_y = int(preds[0] * w), int(preds[1] * h)
    bottom_right_x, bottom_right_y = int(preds[2] * w), int(preds[3] * h)

    box_predicted = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    # Create the predicted bounding box
    rect = patches.Rectangle(
        (top_left_x, top_left_y),
        bottom_right_x - top_left_x,
        bottom_right_y - top_left_y,
        facecolor="none",
        edgecolor="red",
        linewidth=1,
    )
    # Add the predicted bounding box to the image
    ax1.add_patch(rect)
    ax1.set_xlabel("Predicted: " + str(box_predicted))

    # Plot ground truth box
    top_left_x, top_left_y = int(true_box[0] * w), int(true_box[1] * h)
    bottom_right_x, bottom_right_y = int(true_box[2] * w), int(true_box[3] * h)
    box_truth = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    # Create the ground truth bounding box
    rect = patches.Rectangle(
        (top_left_x, top_left_y),
        bottom_right_x - top_left_x,
        bottom_right_y - top_left_y,
        facecolor="none",
        edgecolor="blue",
        linewidth=1,
    )
    # Add the ground truth bounding box to the image
    ax2.add_patch(rect)
    ax2.set_xlabel("Ground Truth: " + str(box_truth))

    plt.show()

# Example usage: Loop over test images
for i in range(min(len(x_test), 10)):  # Process up to 10 images
    input_image = x_test[i]
    true_box = y_test[i]  # Assuming y_test contains ground truth bounding boxes

    # Predict and plot
    predict_and_plot(input_image, vit_object_detector, true_box)


# In[ ]:





# In[ ]:





# In[ ]:




