import abc

from keras import Sequential
from keras.initializers import TruncatedNormal
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.layers.normalization import BatchNormalization


class NetworkModeler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_fixed_model(self):
        pass


class LeNetModeler(NetworkModeler):
    def __init__(self):
        self.input_shape = None
        self.num_classes = None

        self.ACTIVATION = 'relu'
        self.DROPOUT = 0.2

    def create_fixed_model(self):
        # 10 outputs
        # final stage is softmax
        self.input_shape = (28, 28, 1)
        self.num_classes = 10

        model = Sequential()

        # extract image features by convolution and max pooling layers
        model.add(Conv2D(20,                            # next 特徴量マップの深さ(Chanel)数
                         kernel_size=5,                 # 特徴量マップの1辺のサイズ
                         padding="same",                # サイズ差の埋め方：same=>0埋め
                         input_shape=self.input_shape,  # 入力次元情報
                         activation=self.ACTIVATION))   # 活性化方法
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(50, kernel_size=5, padding="same", activation=self.ACTIVATION))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # classify the class by fully-connected layers
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))
        model.add(Dense(self.num_classes))
        model.add(Activation("softmax"))

        model.summary()

        return model

    def create_simple_model(self):
        self.input_shape = (32, 32, 3)
        self.num_classes = 10
        self.DROPOUT = 0.5

        model = Sequential()

        model.add(Conv2D(32,                            # next 特徴量マップの深さ(Chanel)数
                         kernel_size=3,                 # 特徴量マップの1辺のサイズ
                         padding="same",                # サイズ差の埋め方：same=>0埋め
                         input_shape=self.input_shape,  # 入力次元情報
                         activation=self.ACTIVATION))   # 活性化方法
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.DROPOUT))

        model.add(Conv2D(64, kernel_size=3, padding="same", activation=self.ACTIVATION))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(self.DROPOUT))
        model.add(Dense(self.num_classes))
        model.add(Activation("softmax"))

        model.summary()

        return model


class CNNModeler(object):
    def __init__(self):
        self.input_shape = None
        self.num_classes = None

        self.ACTIVATION = 'relu'
        self.DROPOUT = 0.2

    def create_deep_model(self):
        self.input_shape = (32, 32, 3)
        self.num_classes = 10
        self.DROPOUT = 0.25

        model = Sequential()

        model.add(Conv2D(32,                            # next 特徴量マップの深さ(Chanel)数
                         kernel_size=3,                 # 特徴量マップの1辺のサイズ
                         padding="same",                # サイズ差の埋め方：same=>0埋め
                         input_shape=self.input_shape,  # 入力次元情報
                         activation=self.ACTIVATION))   # 活性化方法
        model.add(Conv2D(32, kernel_size=3, activation=self.ACTIVATION))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.DROPOUT))

        model.add(Conv2D(64, kernel_size=3, padding="same", activation=self.ACTIVATION))
        model.add(Conv2D(64, kernel_size=3, activation=self.ACTIVATION))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.DROPOUT))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation("softmax"))

        model.summary()

        return model


class DNNModeler(NetworkModeler):
    def __init__(self):
        self.N_HIDDEN = 128  # 隠れ層のニューロン数
        self.INPUT_SHAPE = 784  # input shape
        self.ACTIVATION = 'relu'
        self.NB_CLASSES = 10  # number of outputs = number of digits
        self.DROPOUT = 0.2

    def create_multilayered_model(self, Hidden_Layers_Neuron_List):
        if Hidden_Layers_Neuron_List:
            model = Sequential()

            layers_neuron_list = Hidden_Layers_Neuron_List
            list_input_dim = ([self.INPUT_SHAPE] + layers_neuron_list)[:-1]
            # list_input_dim    : ([n_in] + n_hiddens)[:-1]
            #                     [784, 200, 200, 200] -> [784, 200, 200]

            for i, input_dim in enumerate(list_input_dim):
                model.add(Dense(layers_neuron_list[i],
                                input_dim=input_dim,
                                kernel_initializer=TruncatedNormal(stddev=0.01)))
                model.add(Activation(self.ACTIVATION))
                model.add(Dropout(self.DROPOUT))

            model.add(Dense(self.NB_CLASSES, kernel_initializer=TruncatedNormal(stddev=0.01)))
            model.add(Activation('softmax'))

            model.summary()

            return model

        else:
            return ValueError

    def create_fixed_model(self):
        return self.create_multilayered_model([128, 128, 128])


class PublicModeler(object):
    def __init__(self):
        pass

    @staticmethod
    def take_inception3_model():
        base_model = InceptionV3(weights="imagenet", include_top=False)
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        prediction = Dense(10, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=prediction)

        model.summary()

        return model
