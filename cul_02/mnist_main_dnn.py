# -*- coding: utf-8 -*-

import os
from mnist_mvc_s4 import load_mnist_data
from mnist_mvc_s4 import MnistCNN, MnistDNN
from mnist_mvc_s4 import LeNetModeler, DNNModeler
from mnist_mvc_s4 import Trainer
from mnist_mvc_s4 import evaluate_test
from keras.models import load_model
from keras.optimizers import Adam, RMSprop


def main():
    # load dataset
    dataset = load_mnist_data()

    # mnist cnn design
    mnist_dnn = MnistDNN(dataset)
    X_train, Y_train, X_test, Y_test = mnist_dnn.preprocess()

    # check model
    model_path = r'C:\Programing\GPU_dl_with_keras\src\cul_02\mnist_dnn_log\model_file.hdf5'
    if os.path.exists(model_path):
        # load model
        dnn_model = load_model(model_path)
        epoch = 10
    else:
        # make model
        dnn = DNNModeler()
        dnn_model = dnn.create_multilayered_model([128, 128, 128, 128, 128])
        epoch = 100

    # model_path = r'C:\Programing\GPU_dl_with_keras\src\cul_02\mnist_dnn_log\model_file.hdf5'
    # # make model
    # dnn = DNNModeler()
    # dnn_model = dnn.create_fixed_model()
    # epoch = 100

    dnn_model.summary()

    # train model
    trainer = Trainer(dnn_model, dir_nm='mnist_dnn_log', loss="categorical_crossentropy", optimizer=RMSprop())
    trainer.train(X_train, Y_train, batch_size=128, epochs=epoch, validation_split=0.2)

    # evaluate model
    evaluate_test(dnn_model, X_test, Y_test)


if __name__ == '__main__':
    main()






