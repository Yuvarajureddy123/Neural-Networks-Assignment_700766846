#!/usr/bin/env python
# coding: utf-8

# In[18]:


from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist, fashion_mnist
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


# Modified autoencoder with an additional hidden layer
encoding_dim = 32  # Size of encoded representation

# Define the input placeholder (for 28x28 images flattened to 784)
input_img = Input(shape=(784,))

# Encoder
encoded = Dense(128, activation='relu')(input_img)  # First hidden layer
encoded = Dense(64, activation='relu')(encoded)     # Additional hidden layer
encoded = Dense(32, activation='relu')(encoded)     # Bottleneck layer

# Decoder
decoded = Dense(64, activation='relu')(encoded)     # Expanding from bottleneck
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)  # Reconstructing the original input

# Model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Load the Fashion MNIST dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Train autoencoder
history = autoencoder.fit(x_train, x_train,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test))


# In[20]:


# plot loss and accuracy using history object

plt.plot(autoencoder.history.history['loss'])
plt.plot(autoencoder.history.history['val_loss'])
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.title("Loss on train and test data")
plt.legend(['Train','Test'],loc='upper right')
plt.show()


# In[21]:


# plot the test image
def validate_image(data):

    random_sample = np.random.choice(data.shape[0],1,replace = False)
    print(f"Tesing on image {random_sample[0]} \n")

    # plot a sample of test data
    plt.imshow(data[random_sample].reshape(28,28))
    plt.show()

    # plot the sample using reconstructed test data

    pred = autoencoder.predict(data[random_sample].reshape(-1,784))
    plt.imshow(pred.reshape(28,28))
    plt.show()


# In[22]:


validate_image(x_test)


# In[22]:





# VIDEO URL : https://drive.google.com/file/d/1FDSByoHTzr6KV0OURiS_CTjWzX0wxPOk/view?usp=sharing
