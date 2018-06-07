import os
import sys
# import tarfile
# import urllib.request as urllib
# import fire
import keras
import math
import numpy as np
import pickle
from keras import backend as K, regularizers
from keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.engine.training import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, \
    Input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img
import numpy as np
import scipy
from scipy import misc
import os
import argparse
from scipy.ndimage.filters import gaussian_filter
from keras.models import load_model

sys.path.append('/cs/labs/daphna/gadic/curriculum_learning/')
from data_generator import DataGenerator
from utils import unpickle

sys.path.append('/cs/labs/daphna/gadic/curriculum_learning/cifar100/')
sys.path.append('/cs/labs/daphna/gadic/curriculum_learning/stl10/')
# from cifar100_model import create_model
from stl10_model import create_model
# from model import Model
from collections import defaultdict
from keras.metrics import categorical_accuracy as accuracy
import numpy as np

from keras.layers import Dropout, Dense, LSTM

from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

subset_model_path = "/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/data/"
CURRICULUM_SCHEDULER_TYPE = "None"
LR_SCHEDULER_TYPE = "lr_sched1"

import tensorflow as tf

np.random.seed(45)
# import tensorflow as tf
tf.set_random_seed(45)
sess = tf.Session()
import numpy as np
from keras import backend as K

K.set_session(sess)


def _build_model_small(n_classes=5, activation='elu', dropout_1_rate=0.25, dropout_2_rate=0.25,
                       reg_factor=200e-4, bias_reg_factor=None, batch_norm=False):
    l2_reg = regularizers.l2(reg_factor)
    l2_bias_reg = None
    if bias_reg_factor:
        regularizers.l2(bias_reg_factor)

    # input image dimensions
    h, w, d = 32, 32, 3

    if K.image_data_format() == 'channels_first':
        input_shape = (3, h, w)
    else:
        input_shape = (h, w, 3)

    # input image dimensions
    x = input_1 = Input(shape=input_shape)

    x = Conv2D(filters=4, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    # if batch_norm:
    #     x = BatchNormalization()(x)
    # x = Activation(activation=activation)(x)
    # x = Conv2D(filters=4, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
    # if batch_norm:
    #     x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_1_rate)(x)

    x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    # if batch_norm:
    #     x = BatchNormalization()(x)
    # x = Activation(activation=activation)(x)
    # x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
    # if batch_norm:
    #     x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_1_rate)(x)

    x = Flatten()(x)
    x = Dense(units=8, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    x = Dropout(rate=dropout_2_rate)(x)
    x = Dense(units=n_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation='softmax')(x)

    return Model(inputs=[input_1], outputs=[x])


def get_model(input_shape, num_classes, dp, activation='elu', reg_factor=200e-4, bias_reg_factor=None):
    l2_reg = regularizers.l2(reg_factor)
    l2_bias_reg = None
    if bias_reg_factor:
        regularizers.l2(bias_reg_factor)
    # input image dimensions
    h, w, d = 32, 32, 3
    if K.image_data_format() == 'channels_first':
        input_shape = (3, h, w)
    else:
        input_shape = (h, w, 3)
    # input image dimensions
    x = input_1 = Input(shape=input_shape)
    x = Conv2D(filters=4, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dp)(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dp)(x)
    x = Flatten()(x)
    x = Dense(units=8, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
    x = Activation(activation=activation)(x)
    x = Dropout(rate=dp)(x)
    x = Dense(units=num_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
    x = Activation(activation='softmax')(x)
    model = Model(inputs=[input_1], outputs=[x])
    return model


def get_model2(input_shape, num_classes, dp):
    x = input_1 = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(rate=0.5)(x)
    x = Dropout(rate=dp)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dp)(x)
    x = Flatten()(x)
    # x = Dense(units=128)(x)
    # x = Dropout(rate=dp)(x)
    x = Dense(units=num_classes)(x)
    x = Activation(activation='softmax')(x)
    model = Model(inputs=[input_1], outputs=[x])
    return model


def _build_model_large(n_classes=5, activation='elu', dropout_1_rate=0.25, dropout_2_rate=0.5,
                       reg_factor=200e-4, bias_reg_factor=None, batch_norm=False):
    l2_reg = regularizers.l2(reg_factor)  # K.variable(K.cast_to_floatx(reg_factor))
    l2_bias_reg = None
    if bias_reg_factor:
        l2_bias_reg = regularizers.l2(bias_reg_factor)  # K.variable(K.cast_to_floatx(bias_reg_factor))

    # input image dimensions
    h, w, d = 32, 32, 3

    if K.image_data_format() == 'channels_first':
        input_shape = (3, h, w)
    else:
        input_shape = (h, w, 3)

    # input image dimensions
    x = input_1 = Input(shape=input_shape)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_1_rate)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_1_rate)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg,
               bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg,
               bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_1_rate)(x)

    x = Conv2D(filters=256, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg,
               bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Conv2D(filters=256, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg,
               bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_1_rate)(x)

    x = Flatten()(x)
    x = Dense(units=512, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    x = Dropout(rate=dropout_2_rate)(x)
    x = Dense(units=n_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation='softmax')(x)

    model = Model(inputs=[input_1], outputs=[x])
    # model.l2_reg = l2_reg
    # model.l2_bias_reg = l2_bias_reg
    return model


def _build_model_large2(n_classes=5, activation='elu', dropout_1_rate=0.25, dropout_2_rate=0.25,
                        reg_factor=200e-4, bias_reg_factor=None, batch_norm=False):
    l2_reg = regularizers.l2(reg_factor)  # K.variable(K.cast_to_floatx(reg_factor))
    l2_bias_reg = None
    if bias_reg_factor:
        l2_bias_reg = regularizers.l2(bias_reg_factor)  # K.variable(K.cast_to_floatx(bias_reg_factor))

    # input image dimensions
    h, w, d = 32, 32, 3

    if K.image_data_format() == 'channels_first':
        input_shape = (3, h, w)
    else:
        input_shape = (h, w, 3)

    # input image dimensions
    x = input_1 = Input(shape=input_shape)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg,
               bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg,
               bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_1_rate)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg,
               bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg,
               bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_1_rate)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_1_rate)(x)

    x = Conv2D(filters=32, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Conv2D(filters=32, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(
        x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_1_rate)(x)

    x = Flatten()(x)
    x = Dense(units=32, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    x = Dropout(rate=dropout_2_rate)(x)
    x = Dense(units=n_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation='softmax')(x)

    model = Model(inputs=[input_1], outputs=[x])
    # model.l2_reg = l2_reg
    # model.l2_bias_reg = l2_bias_reg
    return model


def get_lr_scheduler(initial_lr):
    def lr_scheduler1(epoch):
        # return initial_lr
        if epoch < 20:
            return initial_lr
        elif epoch < 40:
            return initial_lr / 2
        elif epoch < 50:
            return initial_lr / 4
        elif epoch < 60:
            return initial_lr / 8
        elif epoch < 70:
            return initial_lr / 16
        elif epoch < 80:
            return initial_lr / 32
        elif epoch < 90:
            return initial_lr / 64
        else:
            return initial_lr / 128

    return lr_scheduler1


if __name__ == "__main__":
    print(keras.__version__)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--generate_weights", default="False", help="dataset to use")
    parser.add_argument("--dataset", default="cifar100/subset1", help="dataset to use")
    # parser.add_argument("--dataset", default="cifar100", help="dataset to use")
    parser.add_argument("--net_type", default="large", help="network size ..")
    parser.add_argument("--optimizer", default="sgd", help="")
    parser.add_argument("--comp_grads", default="False", help="")
    parser.add_argument("--learning_rate", "-lr", default=2e-3, type=float)
    parser.add_argument("--l2_reg", default=200e-4, type=float)
    parser.add_argument("--bias_l2_reg", default=None, type=float)
    parser.add_argument("--dropout1", default=0.25, type=float)
    parser.add_argument("--dropout2", default=0.5, type=float)
    parser.add_argument("--curriculum", "-cl", default="None")
    parser.add_argument("--curriculum_scheduler", default="naive")
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--data_aug", default="False", help="augmentation")
    parser.add_argument("--exp", default=20, type=int)
    parser.add_argument("--rept", default=5, type=int)
    parser.add_argument("--lr_sched", default="lr_sched1")

    destination = "/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/data/model/"
    # destination = "/cs/labs/daphna/gadic/curriculum_learning/cifar100/data/model/"
    # destination = "/cs/labs/daphna/gadic/curriculum_learning/stl10/data/model/"
    args = parser.parse_args()

    if args.dataset == "cifar10":
        import cifar10.cifar10 as dataset
    elif args.dataset == "cifar100":
        import cifar100.cifar100 as dataset
    elif args.dataset == "stl10":
        import stl10.stl10 as dataset
    elif "subset" in args.dataset:
        import cifar100.cifar100_subset as dataset

        dataset.dataset_name = args.dataset.split('/')[-1]
        dataset.data_path = os.path.join("/cs/labs/daphna/gadic/curriculum_learning/cifar100/",
                                         dataset.dataset_name + "/data/")
        # dataset.results_path = os.path.join("/cs/labs/daphna/gadic/curriculum_learning/cifar100/",dataset.dataset_name + "/results2/")
        dataset.destination = "/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/data/model/"
        dataset.results_path = os.path.join("/cs/labs/daphna/gadic/curriculum_learning/cifar100/",
                                            dataset.dataset_name + "/results/")



    h, w, d = 32, 32, 3
    input_shape = (h, w, d)
    num_classes = 5
    batch_size = 100
    epochs = 1
    dp = 0.25
    initial_lr = 1e-3
    # model2 = get_model2(input_shape=input_shape, num_classes=num_classes, dp=dp)
    # print(model2.summary())
    #
    # model2.compile(loss=keras.losses.categorical_crossentropy,
    #                optimizer=keras.optimizers.adam(),
    #                metrics=['accuracy'])

    cache_file = os.path.join(dataset.data_path, 'data.pkl')
    (x_train, cls_train, y_train), (x_test, cls_test, y_test) = dataset.load_data_cache(cache_file)
    x_train = (x_train - 128.) / 128
    x_test = (x_test - 128.) / 128

    # model2.fit(x_train, y_train,
    #            batch_size=batch_size,
    #            epochs=epochs,
    #            verbose=1,
    #            validation_data=(x_test, y_test))
    # score = model2.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test Model2 accuracy:', score[1])

    from keras.callbacks import LearningRateScheduler
    reduce_lr = LearningRateScheduler(get_lr_scheduler(initial_lr))
    num_samples = len(y_train)
    class_labels = np.argmax(y_train, axis=1)
    labels_count = np.zeros(num_samples)
    num_exps = 20
    num_repts = 5
    index = 0
    # trained_model_path = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/' \
    #                      + 'results/small_final/sgd/0.002/0.005/None/exp{0}' + \
    #                      '/model_trained{1}.h5'
    # trained_model_path = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/' \
    #                      + 'results/small_final/sgd/0.002/0.005/curriculum/exp{0}' + \
    #                      '/model_trained{1}.h5'
    trained_model_path = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/' \
                         + 'results_submission_icml/archive/large/sgd/0.002/0.002/None/exp{0}' + \
                         '/model_trained{1}.h5'
     # sorted_indices1 = unpickle(os.path.join(dataset.data_path, 'sorted_indices.pkl')).astype(np.int).reshape(-1, )
    # sorted_indices2 = unpickle(os.path.join(dataset.data_path, 'sorted_indices_mc.pkl')).astype(np.int).reshape(-1, )
    # # # print(sorted_indices1[:100])
    # # print(sorted_indices1.shape)
    # # print(sorted_indices2.shape)
    # # exit()
    # # print(sorted_indices2[:100])
    # # print(np.max(sorted_indices2))




    # import scipy.stats.mstats as st
    # #
    # print(st.spearmanr(sorted_indices1, sorted_indices2))
    # temp = np.arange(0, 2500)
    # np.random.shuffle(temp)
    # print(st.spearmanr(sorted_indices1, temp))
    # # exit()
    for e in range(0, num_exps, 5):
        for r in range(0, num_repts, 5):
            index += 1
            print("BEFORE #############", dataset.data_path)
            # net_type = 'small'
            net_type = 'large'
            old_model = load_model(trained_model_path.format(e, r))
            old_model.summary()
            score = old_model.evaluate(x_test, y_test, verbose=0)
            print('Test Model loaded accuracy:', score[1])
            print("MID #############")
            dp = 0.9
            model = _build_model_large(n_classes=num_classes, dropout_1_rate=dp, dropout_2_rate=dp)

            # model = get_model(input_shape=input_shape, num_classes=num_classes, dp=dp)
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.adam(),
                          metrics=['accuracy'])

            score = model.evaluate(x_test, y_test, verbose=0)
            print('Test Model1 accuracy:', score[1])
            model.set_weights(old_model.get_weights())
            score = model.evaluate(x_test, y_test, verbose=0)
            print('Test Model1 accuracy:', score[1])
            score = model.evaluate(x_train, y_train, verbose=0)
            print('Train Model1 loss:', score[0])
            print('Train Model1 accuracy:', score[1])
            print("AFTER #############", index)

            nb_MC_samples = 100
            MC_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
            learning_phase = True  # use dropout at test time
            # MC_samples = [MC_output([test_data[:100], learning_phase])[0] for _ in range(nb_MC_samples)]
            MC_samples = [MC_output([x_train, learning_phase])[0] for _ in range(nb_MC_samples)]
            MC_samples = np.array(MC_samples)  # [#samples x batch size x #classes]
            # print(MC_samples.shape)

            for j in range(num_samples):
                # classes
                # cls = np.argmax(y_train[j], axis=1)
                # print(y_train[j], cls)
                cls_count = np.sum(np.argmax(MC_samples, axis=2)[:, j] == class_labels[j])
                labels_count[j] += cls_count
                # print("cls:", cls)
                # print("cls_count", cls_count)
            # print(labels_count[:100])
            sorted_ = labels_count.argsort(axis=0)[::-1]

            # sorted_ = np.squeeze(sorted_)
            # print(np.squeeze(sorted_).shape)
            # print(sorted_.shape)
            # print(sorted_)
            # exit()
            # print(labels_count[sorted_[:100]])
            # print(labels_count[sorted_[:100]] / index)

    sorted_indices = np.zeros((np.int(num_samples / num_classes), num_classes))

    for c in range(num_classes):
        # print(c)
        # print(class_labels[sorted_])
        print(sorted_[np.where(class_labels[sorted_] == c)])
        sorted_indices[:, c] = sorted_[np.where(class_labels[sorted_] == c)]

    # print(sort_indices[:10])
    # print(sort_indices[:10].dtype)
    parent_path = '/cs/labs/daphna/gadic/curriculum_learning/'
    save_path = 'cifar100/subset1/'
    with open(os.path.join(dataset.data_path, 'sorted_indices_mc_large.pkl'), mode='wb') as file:
        pickle.dump(sorted_indices, file)
    sorted_indices1 = unpickle(os.path.join(dataset.data_path, 'sorted_indices.pkl')).astype(np.int).reshape(-1, )
    # sorted_indices2 = unpickle(os.path.join(dataset.data_path, 'sorted_indices_mc.pkl')).astype(np.int).reshape(-1, )
    sorted_indices2 = unpickle(os.path.join(dataset.data_path, 'sorted_indices_mc_large.pkl')).astype(np.int).reshape(-1, )

    import scipy.stats.mstats as st
    print(st.spearmanr(sorted_indices1, sorted_indices2))
    print(sorted_indices1[:100])
    print(sorted_indices2[:100])