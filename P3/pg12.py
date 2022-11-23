# %%
import keras
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# %%
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))

# %%
indexes = np.random.randint(0, X_train.shape[0], size=25)
images = X_train[indexes]
labels = y_train[indexes]

plt.figure(figsize=(5,5))
for i in range(len(indexes)):
    plt.subplot(5, 5, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
plt.show()

# %%
X_train.shape

# %%
X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')
X_train.shape

# %%
X_train = X_train / 255
X_test = X_test / 255

# %%
# Building the model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# %%
model.add(Dense(32, input_dim = 28 * 28, activation= 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'sigmoid'))
model.layers

# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
model.fit(X_train, y_train, batch_size=32, epochs=5)

# %%
scores = model.evaluate(X_test,y_test)
print('Accuracy : ',scores[1]*100)

# %%
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# %%
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred,axis=1)
print(y_pred)
print(y_pred_classes)

# %%
random_idx = np.random.choice(len(X_test))
x_sample = X_test[random_idx]
y_true = np.argmax(y_test,axis=1)
y_sample_true = y_true[random_idx]
y_sample_pred_class = y_pred_classes[random_idx]

plt.title('Predicted : {}, True : {}'.format(y_sample_pred_class,y_sample_true))
plt.imshow(x_sample.reshape(28, 28), cmap='gray')

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')

# %%
model.save('save.h5')
newmodel = keras.models.load_model('save.h5')
newmodel.predict(X_test)

random_idx = np.random.choice(len(X_test))
x_sample = X_test[random_idx]
y_true = np.argmax(y_test,axis=1)
y_sample_true = y_true[random_idx]
y_sample_pred_class = y_pred_classes[random_idx]

plt.title('Predicted : {}, True : {}'.format(y_sample_pred_class,y_sample_true))
plt.imshow(x_sample.reshape(28, 28), cmap='gray')

# %%



