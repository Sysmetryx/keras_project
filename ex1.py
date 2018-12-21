from keras.datasets import mnist
from keras.models import Sequential
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_yaml
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

def loader():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_train, y_train, X_test, y_test

def saveModel(model, savename):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        print("Yaml Model ", savename, ".yaml saved to disk")
    # serialize weights to HDF5
    model.save_weights(savename + ".h5")
    print("Weights ", savename, ".h5 saved to disk")


def ex1():
    # Loading dataset
    X_train, y_train, X_test, y_test = ex0.loader()
    # Creating empty model
    model = Sequential()
    # Adding fully connected layer
    model.add(Dense(10, input_dim=784, name='fc1'))
    # Adding an output layer with softmax activation
    model.add(Activation('softmax'))
    # Display network info
    model.summary()
    # Preparing network for train with learning rate
    learning_rate = 0.1
    sgd = SGD(learning_rate)
    # selecting training method
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    # formating data
    batch_size = 100
    nb_epoch = 20
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1)
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def ex2():
    model = Sequential()
    model.add(Dense(100, input_dim=784, name='fc1'))
    model.add(Activation('sigmoid'))
    model.add(Dense(10, name='Out'))
    model.add(Activation('softmax'))
    model.summary()
    # Preparing network for train with learning rate
    learning_rate = 1.0
    sgd = SGD(learning_rate)
    # selecting training method
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    # formating data
    X_train, y_train, X_test, y_test = ex0.loader()
    batch_size = 100
    nb_epoch = 100
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1)
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    saveModel(model, "ex2model")


def ex3():
    model = Sequential()
    x_train, y_train, x_test, y_test = loader()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    conv = Conv2D(16, kernel_size=(5, 5), activation='relu',
             input_shape=input_shape,
           padding='valid')
    model.add(conv)
    pool = MaxPooling2D(pool_size=(2, 2))
    model.add(pool)
    conv2 = Conv2D(32, kernel_size=(5, 5), activation='relu',
             input_shape=input_shape,
           padding='valid')
    model.add(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))
    model.add(pool2)
    model.add(Flatten())
    model.add(Dense(100, input_dim=784, name='fc1'))
    model.add(Activation('sigmoid'))
    model.add(Dense(10, name='Out'))
    model.add(Activation('softmax'))
    model.summary()
    batch_size = 100
    nb_epoch = 100
    # Preparing network for train with learning rate
    learning_rate = .5
    sgd = SGD(learning_rate)
    # selecting training method
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1)
    scores = model.evaluate(x_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    saveModel(model, "ex3model")


def main():
    # ex1()
    # ex2()
    ex3()

if __name__ == '__main__':
    main()
