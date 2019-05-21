# -*- coding: utf-8 -*-

from mnist_mvc_s4 import load_mnist_data
from mnist_mvc_s4 import MnistCNN, MnistDNN
from mnist_mvc_s4 import LeNetModeler
from mnist_mvc_s4 import Trainer
from mnist_mvc_s4 import evaluate_test
from keras.models import load_model
from keras.optimizers import Adam, RMSprop


def main():
    # load dataset
    dataset = load_mnist_data()

    # mnist cnn design
    mnist_cnn = MnistCNN(dataset)
    X_train, Y_train, X_test, Y_test = mnist_cnn.preprocess()

    # # make model
    # lenet = LeNetModeler()
    # lenet_model = lenet.create_fixed_model()

    # load model
    model_path = r'C:\Programing\GPU_dl_with_keras\src\cul_02\mnist_lenet_log\model_file.hdf5'
    lenet_model = load_model(model_path)


    # train model
    trainer = Trainer(lenet_model, dir_nm='mnist_lenet_log', loss="categorical_crossentropy", optimizer=RMSprop())
    trainer.train(X_train, Y_train, batch_size=128, epochs=5, validation_split=0.2)

    # evaluate model
    evaluate_test(lenet_model, X_test, Y_test)


# def re_main():
#     # load dataset
#     # TODO: Make Class LoadData(object, MNIST)
#     dataset = MNISTDatasetDNN()
#
#     # split data
#     # TODO: Make Method preprocess(self, 'DNN/CNN')
#     X_train, Y_train, X_test, Y_test = dataset.preprocess()
#
#     # load model
#     model_path = r'C:\Programing\GPU_dl_with_keras\src\cul_02\mnist_log\model_file.hdf5'
#     model = load_model(model_path)
#
#     # train model
#     trainer = Trainer(model, loss="categorical_crossentropy", optimizer=RMSprop())
#     trainer.train(X_train, Y_train, batch_size=128, epochs=2, validation_split=0.2)
#
#     # evaluate model
#     evaluate_test(model, X_test, Y_test)


if __name__ == '__main__':
    main()    # step1 or 3
    # re_main()   # step2






