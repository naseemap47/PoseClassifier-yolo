from keras import layers, Sequential
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import keras
from utils import save_model_ext
import pandas as pd
import os
import json
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--data", type=str, required=True,
                help="path to csv Data")
args = vars(ap.parse_args())

# Load .csv Data
df = pd.read_csv(args["data"])
class_list = df['Pose_Class'].unique()
class_list = sorted(class_list)
labels_string = json.dumps(class_list)
class_number = len(class_list)

# Create training and validation splits
x = df.copy()
y = x.pop('Pose_Class')
y, _ = y.factorize()
x = x.astype('float64')
y = keras.utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=0)

print('[INFO] Loaded csv Dataset')

# Keras Model Arc
model = Sequential([
    layers.Dense(512, activation='relu', input_shape=[x_train.shape[1]]),
    layers.Dense(256, activation='relu'),
    layers.Dense(class_number, activation="softmax")
])

# Model Summary
print('Model Summary: ', model.summary())

# model compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Save Model Dir
os.makedirs('runs', exist_ok=True)
c = 0
while True:
    path_save = os.path.join('runs', f'train{c}')
    if not os.path.exists(path_save):
        os.makedirs(path_save, exist_ok=True)
        break
    else:
        c += 1
        continue

# Add a checkpoint callback to store the checkpoint that has the highest
# validation accuracy.
ckpt_path = os.path.join(path_save, "weight-{epoch:02d}-{val_accuracy:.2f}.h5")
checkpoint = keras.callbacks.ModelCheckpoint(ckpt_path,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=20)

# Start training
print('[INFO] Model Training Started ...')
history = model.fit(x_train, y_train,
                    epochs=200,
                    batch_size=16,
                    validation_data=(x_test, y_test),
                    callbacks=[checkpoint, earlystopping])

print('[INFO] Model Training Completed')
save_model_ext(model, os.path.join(path_save, 'model.h5'), meta_data=labels_string)
print(f'[INFO] Model Successfully Saved \033[1m model.h5 \033[0;0m in {path_save}')

# Plot History
metric_loss = history.history['loss']
metric_val_loss = history.history['val_loss']
metric_accuracy = history.history['accuracy']
metric_val_accuracy = history.history['val_accuracy']

# Construct a range object which will be used as x-axis (horizontal plane) of the graph.
epochs = range(len(metric_loss))

# Plot the Graph.
plt.plot(epochs, metric_loss, 'blue', label=metric_loss)
plt.plot(epochs, metric_val_loss, 'red', label=metric_val_loss)
plt.plot(epochs, metric_accuracy, 'magenta', label=metric_accuracy)
plt.plot(epochs, metric_val_accuracy, 'green', label=metric_val_accuracy)

# Add title to the plot.
plt.title(str('Model Metrics'))

# Add legend to the plot.
plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])

# If the plot already exist, remove
plt.savefig(os.path.join(path_save, 'metrics.png'), bbox_inches='tight')
print(f'[INFO] Successfully Saved \033[1m metrics.png \033[0;0m in {path_save}')
