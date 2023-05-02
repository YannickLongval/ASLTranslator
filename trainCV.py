"""
Trains a computer vision model to detect ASL letters
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

"""Plot the preformance of the model on training/testing data

Args:
    history: the performance data of the model
    metric (string): the metric to be plotted
"""
def plot_graphs(history, metric:str) -> None:
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

# importing the train/test data for the model into pandas dataframes
train_pd:pd.DataFrame = pd.read_csv('data/sign_mnist_train.csv',delimiter=',',encoding='latin-1')
test_pd:pd.DataFrame = pd.read_csv('data/sign_mnist_test.csv',delimiter=',',encoding='latin-1')

# separate the target from the data. This is to create the tensorflow.data.Dataset
var_target_train, var_target_test = train_pd.pop('label'), test_pd.pop('label')

# creating the tf.data.Datasets
train:tf.data.Dataset = tf.data.Dataset.from_tensor_slices((train_pd.values, var_target_train.values))
test:tf.data.Dataset = tf.data.Dataset.from_tensor_slices((test_pd.values, var_target_test.values))


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

train = train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
test = test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
train = train.cache()

# setting buffer and batch sizes based on amount of data
BUFFER_SIZE:int = 1000
BATCH_SIZE:int = 128

# shuffle, and batch data
train = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test = test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(784)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(26)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# training model with training data, and validating with test data
history = model.fit(train, epochs=6, validation_data=test)

# save the model to be used in the future
model.save('models/ASL_CV_model')

# plot accuracy/loss of classifier
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.show()