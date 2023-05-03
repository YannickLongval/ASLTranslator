"""
Trains a computer vision model to detect ASL letters
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn

# importing the train/test data for the model into pandas dataframes
train_df:pd.DataFrame = pd.read_csv('data/sign_mnist_train.csv',delimiter=',',encoding='latin-1')
test_df:pd.DataFrame = pd.read_csv('data/sign_mnist_test.csv',delimiter=',',encoding='latin-1')

# Rename label into Label
train_df.rename(columns={'label':'Label'},inplace = True)
test_df.rename(columns={'label':'Label'},inplace = True)

# Shuffle
train_df = train_df.sample(frac = 1.0).reset_index(drop = True)
test_df = test_df.sample(frac = 1.0).reset_index(drop = True)

# # separate the target from the data. This is to create the tensorflow.data.Dataset
# var_target_train, var_target_test = train_pd.pop('label'), test_pd.pop('label')

# # creating the tf.data.Datasets
# train:tf.data.Dataset = tf.data.Dataset.from_tensor_slices((train_pd.values, var_target_train.values))
# test:tf.data.Dataset = tf.data.Dataset.from_tensor_slices((test_pd.values, var_target_test.values))


# def normalize_img(image, label):
#   """Normalizes images: `uint8` -> `float32`."""
#   return tf.cast(image, tf.float32) / 255., label

# train = train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# test = test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# train = train.cache()

# # setting buffer and batch sizes based on amount of data
# BUFFER_SIZE:int = 1000
# BATCH_SIZE:int = 128

# # shuffle, and batch data
# train = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test = test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(train_df.shape)


train_df_original = train_df.copy()

# Split into training, test and validation sets
val_index = int(train_df.shape[0]*0.1)

train_df = train_df_original.iloc[val_index:]
val_df = train_df_original.iloc[:val_index]

y_train = train_df['Label']
y_val = val_df['Label']
y_test = test_df['Label']

# Reshape the traing and test set to use them with a generator
X_train = train_df.drop('Label',axis = 1).values.reshape(train_df.shape[0], 28, 28, 1)
X_val = val_df.drop('Label',axis = 1).values.reshape(val_df.shape[0], 28, 28, 1)
X_test = test_df.drop('Label',axis = 1).values.reshape(test_df.shape[0], 28, 28, 1)

# Data augmentation
generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                            rotation_range=10,
                                                            zoom_range=0.10,
                                                            width_shift_range=0.1,
                                                            height_shift_range=0.1,
                                                            shear_range=0.1,
                                                            horizontal_flip=False,
                                                            fill_mode="nearest")

X_train_flow = generator.flow(X_train, y_train, batch_size=32)
X_val_flow = generator.flow(X_val, y_val, batch_size=32)

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters=32,  kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
                    tf.keras.layers.MaxPool2D(2,2, padding='same'),
                    
                    tf.keras.layers.Conv2D(filters=128,  kernel_size=(3,3), activation="relu"),
                    tf.keras.layers.MaxPool2D(2,2, padding='same'),
                    
                    tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation="relu"),
                    tf.keras.layers.MaxPool2D(2,2, padding='same'),
                    
                    tf.keras.layers.Flatten(),
                    
                    tf.keras.layers.Dense(units=1024, activation="relu"),                 
                    tf.keras.layers.Dense(units=256, activation="relu"),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(units=25, activation="softmax")
])

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# training model with training data, and validating with test data
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

history = model.fit(X_train_flow, 
                    validation_data=X_val_flow, 
                    epochs=5,
                    callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=5,
                            restore_best_weights=True), 
                        
                            learning_rate_reduction
                    ])

# save the model to be used in the future
model.save('models/ASL_CV_model')

fig, axes = plt.subplots(2, 1, figsize=(15, 10))
ax = axes.flat

pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot(ax=ax[0])
ax[0].set_title("Accuracy", fontsize = 15)
ax[0].set_ylim(0,1.1)

pd.DataFrame(history.history)[['loss','val_loss']].plot(ax=ax[1])
ax[1].set_title("Loss", fontsize = 15)
plt.show()

# Predict the label of the test_images
pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)

# Get the accuracy score
acc = sklearn.metrics.accuracy_score(y_test,pred)

# Display the results
print(f'## {acc*100:.2f}% accuracy on the test set')