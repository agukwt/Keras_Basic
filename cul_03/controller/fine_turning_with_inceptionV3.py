import os

from pathlib import Path
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from model.dataloader import load_cifar10
from model.preprocessor import PreprocessAdaptingCifer10toInception3
from model.networkmodeler import PublicModeler
from model.networktrainer import Trainer
from model.networkevaluator import evaluate_test


def transfer_learnig():
    """
    """

    # 1. Load data, Cifar10 and Preprocess for adapting Inception v3 network.
    dataset = load_cifar10()
    preprocessor = PreprocessAdaptingCifer10toInception3(dataset)
    X_train, Y_train, X_test, Y_test = preprocessor.get_batch()

    # 2. Load network model, InceptionV3 (299, 299, 3).
    # 3. Redesign last layers of loaded model for adapting present task.
    model = PublicModeler.take_inception3_model()

    # 4. Define model "except loaded network".<Transfer-learning>
    trainer = Trainer(model, logdir_nm="model_inception3_log", loss="categorical_crossentropy", optimizer=RMSprop())

    # 5.
    trainer.train_with_agu(
        X_train, Y_train, batch_size=26, epochs=8, validation_split=0.2
    )


def fine_turnig():
    """
    """
    dataset = load_cifar10()
    preprocessor = PreprocessAdaptingCifer10toInception3(
        dataset,
        train_data_size=20000, test_data_size=1000)
    X_train, Y_train, X_test, Y_test = preprocessor.get_batch()

    # 5. Redesign model so that upper layers are trained
    model_path = str(Path(os.path.dirname(__file__)).parent.
                     joinpath('model_inception3_log').joinpath('model_file.hdf5'))
    model = load_model(model_path)

    # 6. Train model "in the back part of the network".<Fine-Turning>
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    trainer = Trainer(model, logdir_nm="model_FT_log", loss="categorical_crossentropy",
                      optimizer=SGD(lr=0.001, momentum=0.9))
    trainer.train_with_agu(
        X_train, Y_train, batch_size=128, epochs=20, validation_split=0.2
    )

    # 7. Evaluate network model.
    model_path = str(Path(os.path.dirname(__file__)).parent.
                     joinpath('model_FT_log').joinpath('model_file.hdf5'))
    model = load_model(model_path)
    evaluate_test(model, X_test, Y_test)


if __name__ == '__main__':
    # transfer_learnig()
    fine_turnig()
