from keras import models
from keras import layers
from keras import regularizers
from keras import initializers
from keras import activations
from keras import optimizers
from keras import losses
from keras import metrics
from keras import callbacks
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class_map = ""

def train_save():
  model = models.Sequential()

  model.add(layers.Conv2D(
    32, (5, 5),
    input_shape = (64, 64, 3),
    activation = 'relu'
  ))
  model.add(layers.MaxPooling2D(
    pool_size = (2, 2)
  ))

  model.add(layers.Conv2D(
    16, (5, 5),
    input_shape = (30, 30, 3),
    activation = 'relu'
  ))
  model.add(layers.MaxPooling2D(
    pool_size = (2, 2)
  ))

  model.add(layers.Conv2D(
    4, (5, 5),
    input_shape = (13, 13, 3),
    activation = 'relu'
  ))
  model.add(layers.MaxPooling2D(
    pool_size = (2, 2)
  ))

  model.add(layers.Flatten())

  model.add(layers.Dense(256,
    kernel_regularizer = regularizers.L2(1e-4),
    kernel_initializer=initializers.GlorotNormal(),
    bias_initializer = initializers.Zeros()
  ))
  model.add(layers.Dropout(0.2))
  model.add(layers.Activation(activations.relu))


  model.add(layers.Dense(64,
    kernel_regularizer = regularizers.L2(1e-4),
    kernel_initializer=initializers.GlorotNormal(),
    bias_initializer = initializers.Zeros()
  ))
  model.add(layers.Activation(activations.relu))


  model.add(layers.Dense(64,
    kernel_regularizer = regularizers.L2(1e-4),
    kernel_initializer=initializers.GlorotNormal(),
    bias_initializer = initializers.Zeros()
  ))
  model.add(layers.Activation(activations.relu))

  model.add(layers.Dense(6,
    kernel_initializer=initializers.GlorotNormal(),
    bias_initializer = initializers.Zeros()
  ))
  model.add(layers.Activation(activations.softmax))

  model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.BinaryCrossentropy(),
    metrics=[metrics.CategoricalAccuracy(), metrics.Precision()]
  )

  dataGen = image.ImageDataGenerator(
    rescale = 1.0/255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = False,
    validation_split = 0.2
  )

  X_train = dataGen.flow_from_directory(
    './seg_train/seg_train/',
    target_size = (64,64),
    batch_size  = 32,
    class_mode = 'categorical',
    subset ='training'
  )

  X_tests = dataGen.flow_from_directory(
    './seg_test/seg_test',
    target_size = (64,64),
    batch_size  = 32,
    class_mode = 'categorical',
    subset ='validation'
  )

  model.fit(X_train, 
    steps_per_epoch=300,
    epochs=30,
    validation_data=X_tests,
    validation_steps=30,
    callbacks=[
      callbacks.EarlyStopping(patience=3, restore_best_weights=True),
  #    callbacks.ModelCheckpoint(filepath = 'model.{epoch:02d}.h5')
    ]
  )

  model.save('model.keras')

  global class_map 
  class_map = dict([v,k] for k,v in X_train.class_indices.items())
  print(class_map)  

def test():
  print("> Insira o tipo da imagem: ", end="")
  tipo = input()
  print("> Insira o nome da imagem: ", end="")
  image = input()
  test_image_path = f"./seg_train/seg_train/{tipo}/{image}.jpg"

  predictions(test_image_path, actual_label =tipo)


def predictions(test_image_path, actual_label):
  global class_map
  try:
    #load and preprocessing image
    test_img = image.load_img(test_image_path, target_size = (64,64))
    test_img_arr = image.img_to_array(test_img)/ 255.0
    test_img_input = test_img_arr.reshape((1, test_img_arr.shape[0], test_img_arr.shape[1] , test_img_arr.shape[2]))

    # prediction
    model = tf.keras.models.load_model("model.keras")
    predicted_label = np.argmax(model.predict(test_img_input))
    predicted_img = class_map[predicted_label]


    plt.figure(figsize = (4,4))
    plt.imshow(test_img_arr)
    plt.title("predicted label: {}, actual label : {}".format (predicted_img, actual_label))
    plt.grid()
    plt.axis('off')
    plt.show()
  except ValueError as ve:
    print(f"Value error: {ve}")  

train_save()

while(True):
  try:
    print("[1] - Train and Save")
    print("[2] - Test")
    print("[3] - Exit")
    print("> ", end="")
    option = (int)(input())

    if(option == 1):
      train_save()
    elif(option == 2):
      test()
    elif(option == 3):
      break
  except:
    print("Invalid Input")
