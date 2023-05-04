"""
Trains a computer vision model to detect ASL letters
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# importing the train/test data for the model into pandas dataframes
train_df:pd.DataFrame = pd.read_csv('data/sign_mnist_train.csv',delimiter=',',encoding='latin-1')
test_df:pd.DataFrame = pd.read_csv('data/sign_mnist_test.csv',delimiter=',',encoding='latin-1')

# Rename label into Label
train_df.rename(columns={'label':'Label'},inplace = True)
test_df.rename(columns={'label':'Label'},inplace = True)

# Shuffle
train_df = train_df.sample(frac = 1.0).reset_index(drop = True)
test_df = test_df.sample(frac = 1.0).reset_index(drop = True)

train_df_original = train_df.copy()

# Split into training, test and validation sets, such that validation is 10% of total data
val_index = int(train_df.shape[0]*0.1)

train_df = train_df_original.iloc[val_index:]
val_df = train_df_original.iloc[:val_index]

y_train = train_df.pop('Label')
y_val = val_df.pop('Label')
y_test = test_df.pop('Label')

# Reshape the training and test set to use them with a generator
X_train = train_df.values.reshape(train_df.shape[0], 28, 28, 1)
X_val = val_df.values.reshape(val_df.shape[0], 28, 28, 1)
X_test = test_df.values.reshape(test_df.shape[0], 28, 28, 1)

# Data augmentation, create the generator to loop through the data, in batches
generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                            rotation_range=15,
                                                            zoom_range=0.10,
                                                            width_shift_range=0.05,
                                                            height_shift_range=0.05,
                                                            shear_range=0.1,
                                                            horizontal_flip=False,
                                                            vertical_flip=True,
                                                            fill_mode="nearest")

# generate batches for training/validation data
X_train_flow = generator.flow(X_train, y_train, batch_size=32)
X_val_flow = generator.flow(X_val, y_val, batch_size=32)

# creating CNN model to predict ASL character from 28x28 pixel image
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