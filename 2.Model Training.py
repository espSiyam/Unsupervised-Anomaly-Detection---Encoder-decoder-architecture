# Importing the libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose
from tensorflow.keras.preprocessing.image import img_to_array,load_img


# Loading the training data
training_data=np.load('./model_files/training.npy')
training_data = merged
frames=training_data.shape[2]
frames=frames-frames%10

training_data=training_data[:,:,:frames]
training_data=training_data.reshape(-1,227,227,10)
training_data=np.expand_dims(training_data,axis=4)
target_data=training_data.copy()

# Defining Model checkpoint, early stopping and reduce learning rate
callback_save = ModelCheckpoint("./model_files/saved_model.h5", 
                                monitor="mean_squared_error", 
                                save_best_only=True,
                                verbose=1)

callback_early_stopping = EarlyStopping(monitor='mean_squared_error', 
                                        patience=3,
                                        verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1,
                              patience=2, min_lr=0.00000001, verbose=1)

# Defining the model
stae_model=Sequential()
stae_model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='tanh'))
stae_model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
stae_model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True))
stae_model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))
stae_model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5))
stae_model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
stae_model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh'))
stae_model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy', 'mean_squared_error'])

# Training the model
epochs = 25
batch_size = 1

stae_model.fit(training_data,
               target_data, 
               batch_size=batch_size, 
               epochs=epochs, 
               callbacks = [callback_save,callback_early_stopping])