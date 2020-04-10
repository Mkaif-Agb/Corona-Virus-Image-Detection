import keras
from matplotlib import pyplot as plt
from keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D, Convolution2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.regularizers import l2
import numpy as np
reg = 0.001

train_path = 'corona/train'
test_path = 'corona/test'
valid_path = 'corona/val'

model = Sequential()
model.add(Convolution2D(16, 3, 3, input_shape=(128, 128, 3), activation='relu', kernel_regularizer=l2(reg)))
model.add(Convolution2D(32, 3, 3, activation='relu', kernel_regularizer=l2(reg)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(reg)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer='adam', metrics=['acc'], loss='categorical_crossentropy')

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical')

filepath = 'Convolutional_corona_virus.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')
call_back_list = [checkpoint]

history = model.fit_generator(
    train_generator,
    samples_per_epoch=244,
    epochs=5,
    validation_data=validation_generator,
    callbacks=call_back_list,
    validation_steps=20)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

model.save(filepath)


def pred(img_path):
    prediction_image = image.load_img(img_path, target_size=(128, 128))
    prediction_image = image.img_to_array(prediction_image)
    prediction_image = np.expand_dims(prediction_image, axis=0)
    result = np.argmax(model.predict(prediction_image))
    # print(result)
    if result == 0:
        print("Corona")
    else:
        print("Healthy")


pred('corona/val/normal/IM-0149-0001.jpeg')


import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.models import load_model
model = load_model('Convolutional_corona_virus.h5')

dict = {"[0]": "corona ",
                      "[1]": "normal"}

dict_n = {"n0": "corona ",
                        "n1": "normal"}


def draw_test(name, pred, im):
    monkey = dict[str(pred)]
    BLACK = [0, 0, 0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100, cv2.BORDER_CONSTANT, value=BLACK)
    monkey = str(' Prediction - {}'.format(monkey))
    cv2.putText(expanded_image, monkey, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow(name, expanded_image)


def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0, len(folders))
    path_class = folders[random_directory]
    print("Class - " + str(path_class))
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0, len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path + "/" + image_name), path_class


for i in range(0, 10):
    input_im, path_class = getRandomImage("./corona/val/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    input_im = cv2.resize(input_im, (128, 128), interpolation=cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1, 128, 128, 3)
    original = str('Original Class - {}' .format(path_class))
    # Get Prediction
    res = np.argmax(model.predict(input_im, 1, verbose=0), axis=1)

    # Show image with predicted class
    cv2.putText(input_original, original, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    draw_test("Prediction", res, input_original)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# from vis.visualization import visualize_saliency, visualize_activation
# import os
# import matplotlib.pyplot as plt
# from vis.utils import utils
# from keras import activations
# import matplotlib.image as impg
#
# model.summary()
# layer_idx = utils.find_layer_idx(model, 'dense_2')
# model.layers[layer_idx].activation = activations.linear
# model = utils.apply_modifications(model)
# img = impg.imread('corona/val/corona/SARS-10.1148rg.242035193-g04mr34g0-Fig8b-day5.jpeg')
# grad = visualize_activation(model, layer_idx, seed_input=img, filter_indices=None,
#                           backprop_modifier=None, grad_modifier='absolute')
# plt.imshow(grad, alpha=.6)