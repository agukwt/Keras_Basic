import os

from pathlib import Path
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from model.dataloader import load_cifar10
from model.preprocessor import CleanPreprocess
from model.networkmodeler import LeNetModeler
from model.networktrainer import Trainer
from model.networkevaluator import evaluate_test


def main():
    # load data
    dataset = load_cifar10()

    # Preprocess dataset
    cp = CleanPreprocess(dataset)
    X_train, Y_train, X_test, Y_test = cp.preprocess_cifer10()

    # model
    lenet = LeNetModeler()
    model = lenet.create_simple_model()

    # train model
    trainer = Trainer(model, logdir_nm="model_cifer10_log",
                      timestamp=False, loss="categorical_crossentropy", optimizer=RMSprop())
    trainer.train(X_train, Y_train, batch_size=128, epochs=100, validation_split=0.2)

    # evaluate model accuracy
    evaluate_test(model, X_test, Y_test)


if __name__ == '__main__':
    main()
