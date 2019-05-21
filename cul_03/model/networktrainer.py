import os

import numpy as np
from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


# Mixup可能なDataGenerator（Data Augmentation）
class MixUpDataGenerator(ImageDataGenerator):
    def __init__(self, mix_up_alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mix_up_alpha = mix_up_alpha

    def mix_up(self, X1, y1, X2, y2):
        assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
        batch_size = X1.shape[0]
        l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)
        X_l = l.reshape(batch_size, 1, 1, 1)
        y_l = l.reshape(batch_size, 1)
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y

    def flow_from_directory(self, *args, **kwargs):
        batches = super().flow_from_directory(*args, **kwargs)
        while True:
            batch_x, batch_y = next(batches)
            # Mix-up
            if self.mix_up_alpha > 0:
                while True:
                    batch_x_2, batch_y_2 = next(batches)
                    m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
                    if m1 < m2:
                        batch_x_2 = batch_x_2[:m1]
                        batch_y_2 = batch_y_2[:m1]
                        break
                    elif m1 == m2:
                        break
                batch_x, batch_y = self.mix_up(batch_x, batch_y, batch_x_2, batch_y_2)
            yield (batch_x, batch_y)


# 画像データをメモリ内に確保(uint8)し、バッチ時にfloat32にキャストするGenerator
class MyDataGenerator(object):
    def flow(self, X, y=None, batch_size=32, shuffle=True):
        n_sample = X.shape[0]
        n_batch = n_sample // batch_size

        while True:
            indices = np.arange(n_sample)
            if shuffle:
                np.random.shuffle(indices)

            for i in range(n_batch):
                current_indices = indices[i * batch_size:(i + 1) * batch_size]
                X_batch = (X[current_indices] / 255.0).astype(np.float32)
                if y is None:
                    yield X_batch
                else:
                    y_batch = (y[current_indices]).astype(np.float32)
                    yield X_batch, y_batch


class Trainer(object):
    def __init__(self, model, logdir_nm="model_log", timestamp=False, loss="categorical_crossentropy", optimizer=Adam()):
        self._train_model = model
        self.loss = loss
        self.optimizer = optimizer

        self._train_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])
        self.verbose = 1

        if timestamp:
            from time import gmtime, strftime
            tictoc = strftime("%Y%m%d_%H%M_%z", gmtime())
            timestamp_str = '_' + str(tictoc)
            logdir_nm += timestamp_str

        self.log_dir = Path(os.path.dirname(__file__)).parent.joinpath(logdir_nm)
        self._train_model_file_name = "model_file.hdf5"

        # print(self.log_dir)

    def clean_dir(self):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)  # remove previous execution
        os.mkdir(self.log_dir)

    def train(self, x_train, y_train, batch_size=128, epochs=20, validation_split=0.2):
        self.clean_dir()

        model_path = os.path.join(self.log_dir, self._train_model_file_name)
        self._train_model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[
                  TensorBoard(log_dir=self.log_dir),
                  ModelCheckpoint(model_path, save_best_only=True),
                  EarlyStopping(monitor='val_loss', patience=10, verbose=1)],
            verbose=self.verbose)

    def train_with_agu(self, x_train, y_train, batch_size=128, epochs=20, validation_split=0.2):
        self.clean_dir()

        data_generate = ImageDataGenerator(
            featurewise_center=False,               # 1. set input mean to 0 over the dataset
            samplewise_center=False,                # 2. set each sample mean to 0
            featurewise_std_normalization=False,    # 3. divide inputs by std
            samplewise_std_normalization=False,     # 4. divide each input by its std
            zca_whitening=False,                    # 5. apply ZCA whitening
            rotation_range=0,                       # 6. randomly rotate images in the range (0~180)
            width_shift_range=0.1,                  # 7. randomly shift images horizontally
            height_shift_range=0.1,                 # 8. randomly shift images vertically
            horizontal_flip=True,                   # 9. randomly flip images
            vertical_flip=False)                    # 10. randomly flip images

        # compute quantities for normalization (mean, std etc)
        data_generate.fit(x_train)

        # split for validation data
        indices = np.arange(x_train.shape[0])
        np.random.seed(seed=1024)
        np.random.shuffle(indices)

        validation_size = int(x_train.shape[0] * validation_split)

        x_train, x_valid = \
            x_train[indices[:-validation_size], :], x_train[indices[-validation_size:], :]

        y_train, y_valid = \
            y_train[indices[:-validation_size], :], y_train[indices[-validation_size:], :]

        model_path = os.path.join(self.log_dir, self._train_model_file_name)
        self._train_model.fit_generator(
            data_generate.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=epochs,
            validation_data=(x_valid, y_valid),
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(model_path, save_best_only=True),
                EarlyStopping(monitor='val_loss', patience=10, verbose=1)],
            verbose=self.verbose,
            workers=4
        )

    def train_generator(self, x_train, y_train, batch_size=128, epochs=20, validation_split=0.2):
        self.clean_dir()

        model_path = os.path.join(self.log_dir, self._train_model_file_name)

        # split for validation data
        indices = np.arange(x_train.shape[0])
        np.random.seed(seed=1024)
        np.random.shuffle(indices)

        validation_size = int(x_train.shape[0] * validation_split)

        x_train, x_valid = \
            x_train[indices[:-validation_size], :], x_train[indices[-validation_size:], :]

        y_train, y_valid = \
            y_train[indices[:-validation_size], :], y_train[indices[-validation_size:], :]

        data_generate = MyDataGenerator()

        x_valid = (x_valid / 255.0).astype(np.float32)
        y_valid = y_valid.astype(np.float32)

        self._train_model.fit_generator(
            data_generate.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=epochs,
            validation_data=(x_valid, y_valid),
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(model_path, save_best_only=True),
                EarlyStopping(monitor='val_loss', patience=10, verbose=1)],
            verbose=self.verbose
        )





