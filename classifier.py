import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Dataset paths
base_dir = "data/cats_and_dogs"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# Parameters
img_height = 180
img_width = 180
batch_size = 32

# Load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Use train_ds to get class names
class_names = train_ds.class_names
print(f"\n✅ Class Names: {class_names}\n")

# Load validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names))  # Output layer
])

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
epochs = 5
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Evaluate accuracy
val_loss, val_acc = model.evaluate(val_ds)
print(f"\n✅ Accuracy on validation set: {val_acc * 100:.2f}%")

# Predict and display results for 5 validation images
plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
    predictions = model.predict(images)
    for i in range(5):
        ax = plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        pred_index = np.argmax(predictions[i])
        true_index = labels[i].numpy()
        plt.title(f"Pred: {class_names[pred_index]}\nTrue: {class_names[true_index]}")
        plt.axis("off")
        print(f"Image {i+1} → Predicted: {class_names[pred_index]} | True: {class_names[true_index]}")

plt.tight_layout()
plt.show()
