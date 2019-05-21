import os

from pathlib import Path
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from model.dataloader import load_cifar10
from model.preprocessor import PreprocessAdaptingCifer10toInception3
from model.networkmodeler import PublicModeler
from model.networktrainer import Trainer
from model.networkevaluator import evaluate_test


def transfer_learning():
    """
    # 1. Load data, Cifar10 and Preprocess for adapting Inception v3 network.
    # 2. Load network model, InceptionV3 (299, 299, 3).
    # 3. Redesign last layers of loaded model for adapting present task.
    # 4. Train model "except loaded network".<Transfer-learning>
    """

    # 1. Load data, Cifar10 and Preprocess for adapting Inception v3 network.
    dataset = load_cifar10()
    preprocessor = PreprocessAdaptingCifer10toInception3(dataset)
    X_train, Y_train, X_test, Y_test = preprocessor.get_batch()

    # 2. Load network model, InceptionV3 (299, 299, 3).
    # 3. Redesign last layers of loaded model for adapting present task.
    model = PublicModeler.take_inception3_model()

    # 4. Train model "except loaded network".<Transfer-learning>
    trainer = Trainer(model, logdir_nm="model_inception3_log", loss="categorical_crossentropy", optimizer=RMSprop())
    trainer.train_with_agu(
        X_train, Y_train, batch_size=26, epochs=8, validation_split=0.2
    )


def fine_turning():
    """
        # 1. Load plane dataset.
        # 2. Load transfer_learning model.
        # 3. Redesign upper layers of model for Fine-Turning.

        # 5. Redesign model so that upper layers are trained
        # 6. Train model "in the back part of the network".<Fine-Turning>
        # 7. Evaluate network model.
    """

    # 1. Load plane dataset.
    import numpy as np
    from keras.utils import to_categorical

    (X_train, Y_train), (X_test, Y_test) = load_cifar10()

    X_train = X_train.reshape(-1, 32, 32, 3).astype(np.uint8)
    Y_train = to_categorical(Y_train).astype(np.uint8)

    X_test = X_test .reshape(-1, 32, 32, 3).astype(np.uint8)
    Y_test = to_categorical(Y_test).astype(np.uint8)

    import cv2
    image_shape = (190, 190, 3)
    def upscale(x):
        data_size = x.shape[0]
        data_upscaled = np.zeros(
            (data_size, image_shape[0], image_shape[1], image_shape[2]),
            np.uint8
        )
        for i, img in enumerate(x):
            large_img = cv2.resize(img, dsize=(image_shape[0], image_shape[1]), )
            data_upscaled[i] = large_img
        return data_upscaled

    X_train = upscale(X_train)
    X_test = upscale(X_test)

    print('raw up')

    X_test = (X_test / 255).astype(np.float32)
    Y_test = Y_test.astype(np.float32)


    # 2. Load transfer_learning model.
    model_path = str(Path(os.path.dirname(__file__)).parent.
                     joinpath('model_inception3_log').joinpath('model_file.hdf5'))
    model = load_model(model_path)

    # 3. Redesign upper layers' settings of model for Fine-Turning.
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # 4. Compile model as Fine-Turning.
    trainer = Trainer(
        model, logdir_nm="model_FT_log", loss="categorical_crossentropy", optimizer=SGD(lr=0.001, momentum=0.9))

    # 5. Train model as Fine-Turning.
    trainer.train_generator(
        X_train, Y_train, batch_size=128, epochs=20, validation_split=0.2
    )

    # 7. Evaluate network model.
    model_path = str(Path(os.path.dirname(__file__)).parent.
                     joinpath('model_FT_log').joinpath('model_file.hdf5'))
    model = load_model(model_path)
    evaluate_test(model, X_test, Y_test)


if __name__ == '__main__':
    # transfer_learning()
    fine_turning()
