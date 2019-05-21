import os

from pathlib import Path
from keras.models import load_model
from keras.optimizers import Adam, RMSprop

from model.dataloader import load_mnist
from model.preprocessor import CleanPreprocess
from model.networkmodeler import LeNetModeler
from model.networktrainer import Trainer
from model.networkevaluator import evaluate_test


def main():
    # load data
    dataset = load_mnist()

    # Preprocess dataset
    mnist_cnn = CleanPreprocess(dataset)
    X_train, Y_train, X_test, Y_test = mnist_cnn.preprocess_mnist_cnn()

    # check model
    model_path = Path(os.path.dirname(__file__)).parent.joinpath('model_log').joinpath('model_file.hdf5')
    reuse_model_flag = model_path.exists()
    # reuse_model_flag = None

    if reuse_model_flag:
        # load model
        model = load_model(model_path)
        epoch = 10
    else:
        # design model
        lenet = LeNetModeler()
        model = lenet.create_fixed_model()
        epoch = 100

    # train model
    trainer = Trainer(model, logdir_nm="model_log", timestamp=False, loss="categorical_crossentropy", optimizer=RMSprop())
    trainer.train(X_train, Y_train, batch_size=128, epochs=epoch, validation_split=0.2)

    # evaluate model accuracy
    evaluate_test(model, X_test, Y_test)


if __name__ == '__main__':
    main()


