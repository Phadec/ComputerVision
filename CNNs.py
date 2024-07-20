import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

main_directory = "D:/datasets/dataset"

def load_and_preprocess_images(main_directory, max_files_per_class=2000):
    images = []
    labels = []
    
    for label in range(10):
        subdir_path = os.path.join(main_directory, str(label)).replace("\\","/")
        if os.path.isdir(subdir_path):
            
            files = [file for file in os.listdir(subdir_path) if file.endswith('.jpg') or file.endswith('.png')]
            
            files = files[:max_files_per_class]
            
            for file in files:
                file_path = os.path.join(subdir_path, file)
                image = Image.open(file_path)
                image = image.resize((28, 28))
                image_array = np.array(image)
                image_array = image_array / 255.0
                images.append(image_array)
                labels.append(int(label))
    
    images = np.array(images)
    labels = np.array(labels)
    
    labels = to_categorical(labels, num_classes=10)
    
    return images, labels


images, labels = load_and_preprocess_images(main_directory)


X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.4, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 4)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))


test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)


y_pred = model.predict(X_val).argmax(axis=1)
y_true = y_val.argmax(axis=1)  
print(classification_report(y_true, y_pred))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Check if the validation accuracy is better than 0.8 and save the model if true
# if val_accuracy > 0.8:
#     model.save('cnn_model.h5')
#     print("Model saved as cnn_model.h5")
# else:
#     print("Model not saved. Accuracy less than 0.8")
