from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
from tensorflow.keras.datasets import cifar10

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
print("Data normalized successfully")
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Model summary
model.summary()
# 1. Compile the model
model.compile(
    optimizer='adam',                # Adam optimizer works well for image classification
    loss='sparse_categorical_crossentropy',  # Use sparse categorical for integer labels
    metrics=['accuracy']             # Track accuracy during training
)

# 2. Train the model
history = model.fit(
    x_train, y_train, 
    epochs=10,               # You can increase later for better accuracy
    batch_size=64,           # Mini-batch size
    validation_split=0.2     # 20% data used for validation
)

# 3. Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test Accuracy:", test_acc)

# 4. (Optional) Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Get predictions
preds = model.predict(x_test)

# Show 5 random test images with predictions
for i in np.random.randint(0, len(x_test), 5):
    plt.imshow(x_test[i])
    plt.title(f"Predicted: {np.argmax(preds[i])}, True: {y_test[i][0]}")
    plt.axis('off')
    plt.show()


