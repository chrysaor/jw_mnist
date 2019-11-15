import numpy
import matplotlib.pyplot

if __name__ == 'main':
    train_data_file = open('mnist_data/mnist_train.csv', 'r')
    data_list = train_data_file.readlines()
    train_data_file.close()

    all_values = data_list[0].split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation=None)
