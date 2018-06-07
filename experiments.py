import os
import sys
import tarfile
import urllib.request as urllib
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
from cifar100_model import create_model
# from stl10_model import create_model
# from model import Model
from collections import defaultdict

seed = np.random.seed(42)
subset_model_path = "/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/data/"
CURRICULUM_SCHEDULER_TYPE = "None"
LR_SCHEDULER_TYPE = "lr_sched1"

def generate_init_weights(dest_path, n_classes, net_type):
    # for type in ['large']:#,'medium','small']:
    destination = os.path.join(dest_path, net_type + '/')
    if not os.path.exists(destination):
        os.makedirs(destination)
    print(net_type)
    for j in range(20):
        # model = create_model(net_type=net_type)
        # model.save(os.path.join(destination, 'model_init_{0}.h5'.format(j)))
        try:
            model = create_model(net_type=net_type, n_classes=n_classes)
            model.save(os.path.join(destination, 'model_init_{0}.h5'.format(j)))
            print(j)
        except:
            pass

    return None


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

    def lr_scheduler2(epoch):
        return initial_lr
        # if epoch < 20:
        #     return initial_lr
        # elif epoch < 40:
        #     return initial_lr / 2
        # elif epoch < 50:
        #     return initial_lr / 4
        # elif epoch < 60:
        #     return initial_lr / 8
        # elif epoch < 70:
        #     return initial_lr / 16
        # elif epoch < 80:
        #     return initial_lr / 32
        # elif epoch < 90:
        #     return initial_lr / 64
        # else:
        #     return initial_lr / 128

    def lr_scheduler3(epoch):
        # return initial_lr
        if epoch < 30:
            return initial_lr
        elif epoch < 50:
            return initial_lr / 2
        elif epoch < 70:
            return initial_lr / 4
        elif epoch < 80:
            return initial_lr / 8
        elif epoch < 90:
            return initial_lr / 16
        else:
            return initial_lr / 32

    def lr_scheduler4(epoch):
        # return initial_lr
        if epoch < 30:
            return initial_lr
        elif epoch < 60:
            return initial_lr / 2
        elif epoch < 80:
            return initial_lr / 4
        elif epoch < 90:
            return initial_lr / 8
        else:
            return initial_lr / 16

    def lr_scheduler5(epoch):
        # return initial_lr
        if epoch < 20:
            return initial_lr
        elif epoch < 30:
            return initial_lr * 2
        elif epoch < 40:
            return initial_lr * 4
        elif epoch < 50:
            return initial_lr / 2
        elif epoch < 70:
            return initial_lr / 4
        elif epoch < 90:
            return initial_lr / 8
        else:
            return initial_lr / 16


    if LR_SCHEDULER_TYPE == "lr_sched1":
        return lr_scheduler1
    elif LR_SCHEDULER_TYPE == "lr_sched2":
        return lr_scheduler2
    elif LR_SCHEDULER_TYPE == "lr_sched3":
        return lr_scheduler3
    elif LR_SCHEDULER_TYPE == "lr_sched4":
        return lr_scheduler4
    elif LR_SCHEDULER_TYPE == "lr_sched5":
        return lr_scheduler5


def get_curriculum_schedule(len_x):

    def cl_scheduler1(data_iter):
        epoch = data_iter.epoch
        # print("dfgadfasdfadfa")
        if epoch < 2:
            data_limit = np.int(np.ceil(len_x * 0.1))  # 10000
        elif epoch < 4:
            data_limit = np.int(np.ceil(len_x * 0.2))  # 20000
        elif epoch < 6:
            data_limit = np.int(np.ceil(len_x * 0.3))  # 30000
        elif epoch < 8:
            data_limit = np.int(np.ceil(len_x * 0.4))  # 30000
        elif epoch < 10:
            data_limit = np.int(np.ceil(len_x * 0.5))  # 30000
        elif epoch < 12:
            data_limit = np.int(np.ceil(len_x * 0.6))  # 30000
        elif epoch < 14:
            data_limit = np.int(np.ceil(len_x * 0.7))  # 30000
        elif epoch < 16:
            data_limit = np.int(np.ceil(len_x * 0.8))  # 30000
        elif epoch < 18:
            data_limit = np.int(np.ceil(len_x * 0.9))  # 30000
        elif epoch < 20:
            data_limit = np.int(np.ceil(len_x * 0.95))  # 40000

        else:
            data_limit = np.int(np.ceil(len_x * 1))  # 50000
        return data_limit

    def cl_scheduler2(data_iter):
        epoch = data_iter.epoch
        if epoch < 3:
            data_limit = np.int(np.ceil(len_x * 0.1))  # 10000
        elif epoch < 6:
            data_limit = np.int(np.ceil(len_x * 0.2))  # 20000
        elif epoch < 9:
            data_limit = np.int(np.ceil(len_x * 0.3))  # 30000
        elif epoch < 12:
            data_limit = np.int(np.ceil(len_x * 0.4))  # 30000
        elif epoch < 15:
            data_limit = np.int(np.ceil(len_x * 0.5))  # 30000
        elif epoch < 18:
            data_limit = np.int(np.ceil(len_x * 0.6))  # 30000
        elif epoch < 21:
            data_limit = np.int(np.ceil(len_x * 0.7))  # 30000
        elif epoch < 24:
            data_limit = np.int(np.ceil(len_x * 0.8))  # 30000
        elif epoch < 27:
            data_limit = np.int(np.ceil(len_x * 0.9))  # 30000
        elif epoch < 30:
            data_limit = np.int(np.ceil(len_x * 0.95))  # 40000

        else:
            data_limit = np.int(np.ceil(len_x * 1))  # 50000
        # print(data_limit)
        return data_limit
    #
    def cl_scheduler3(data_iter):
        epoch = data_iter.epoch
        if epoch < 2:
            data_limit = np.int(np.ceil(len_x * 0.1))  # 10000
        elif epoch < 4:
            data_limit = np.int(np.ceil(len_x * 0.3))  # 20000
        elif epoch < 6:
            data_limit = np.int(np.ceil(len_x * 0.5))  # 30000
        elif epoch < 7:
            data_limit = np.int(np.ceil(len_x * 0.6))  # 30000
        elif epoch < 9:
            data_limit = np.int(np.ceil(len_x * 0.9))  # 30000
        elif epoch < 10:
            data_limit = np.int(np.ceil(len_x * 0.95))
        else:
            data_limit = np.int(np.ceil(len_x * 1))  # 50000
        return data_limit


    def adaptive(data_iter):
        # print(data_iter.image_data_generator.history[data_iter.dc_idx]['loss'])
        # print(data_iter.epoch)
        # print(type(data_iter.epoch))
        # data_limit = data_iter.data_limit
        # print('dsfdfsdf', data_iter.epoch, data_iter.dc_idx)
        if (data_iter.epoch < 2):
            data_iter.data_limit = np.int(np.ceil(len_x * 0.2))
        elif (data_iter.epoch > data_iter.dc_idx + 1):
            loss_prev = data_iter.image_data_generator.history[data_iter.dc_idx]['loss']
            loss_cur = data_iter.image_data_generator.history[(data_iter.epoch - 2)]['loss']
            # loss_cur = data_iter.image_data_generator.history[(data_iter.epoch-1)]['loss']
            # print("ratio",(loss_cur/loss_prev ) )
            if (((loss_cur/loss_prev ) < 0.99) or (data_iter.epoch - data_iter.dc_idx >= 3)):
                data_iter.data_limit += np.int(np.ceil(len_x * 0.05))
                data_iter.data_limit = np.min((len_x,data_iter.data_limit))
                # data_iter.data_limit = data_limit
                data_iter.dc_idx = data_iter.epoch
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',data_iter.data_limit)


    def adaptive2(data_iter):
        # print(data_iter.image_data_generator.history[data_iter.dc_idx]['loss'])
        # print(data_iter.epoch)
        # print(type(data_iter.epoch))
        # data_limit = data_iter.data_limit
        # print('dsfdfsdf', data_iter.epoch, data_iter.dc_idx)
        if (data_iter.epoch < 2):
            data_iter.data_limit = np.int(np.ceil(len_x * 0.3))
        elif (data_iter.epoch > data_iter.dc_idx + 1):
            loss_prev = data_iter.image_data_generator.history[data_iter.dc_idx]['loss']
            loss_cur = data_iter.image_data_generator.history[(data_iter.epoch - 2)]['loss']
            # loss_cur = data_iter.image_data_generator.history[(data_iter.epoch-1)]['loss']
            # print("ratio",(loss_cur/loss_prev ) )
            if (((loss_cur/loss_prev ) < 0.99) or (data_iter.epoch - data_iter.dc_idx >= 4)):
                data_iter.data_limit += np.int(np.ceil(len_x * 0.05))
                data_iter.data_limit = np.min((len_x,data_iter.data_limit))
                # data_iter.data_limit = data_limit
                data_iter.dc_idx = data_iter.epoch
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',data_iter.data_limit)


    def adaptive_small(data_iter):
        # print(data_iter.image_data_generator.history[data_iter.dc_idx]['loss'])
        # print(data_iter.epoch)
        # print(type(data_iter.epoch))
        # data_limit = data_iter.data_limit
        # print('dsfdfsdf', data_iter.epoch, data_iter.dc_idx)
        if (data_iter.epoch > 80):
            data_iter.data_limit = np.int(np.ceil(len_x))
        if (data_iter.epoch < 2):
            data_iter.data_limit = np.int(np.ceil(len_x * 0.04))
        elif (data_iter.epoch > data_iter.dc_idx + 1):
            loss_prev = data_iter.image_data_generator.history[data_iter.dc_idx]['loss']
            loss_cur = data_iter.image_data_generator.history[(data_iter.epoch - 2)]['loss']
            # loss_cur = data_iter.image_data_generator.history[(data_iter.epoch-1)]['loss']
            # print("ratio",(loss_cur/loss_prev ) )
            if (((loss_cur/loss_prev ) < 0.9) or (data_iter.epoch - data_iter.dc_idx >= 2)):
                data_iter.data_limit += np.int(np.ceil(len_x * 0.05))
                data_iter.data_limit = np.min((len_x,data_iter.data_limit))
                # data_iter.data_limit = data_limit
                data_iter.dc_idx = data_iter.epoch
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',data_iter.data_limit)
                # print(data_iter.data_limit)
        # if save_dir == 'easy':
        #     if not increase_data and data_limit < train_labels.shape[0] and step > 20 and (
        #                 step - increase_step) > 10:
        #         #     # increase_data = loss_out <= inc_crit #or (((step-increase_step) > 400) and (step > (steps/10)))
        #         #     # increase_data = loss_out <= (inc_crit * 0.99 ** (data_limit / 100)) or (
        #         #     # ((step - increase_step) > 1000) and (step > (steps / 20)))
        #         #     # increase_data = loss_out <= (inc_crit * 0.95 ** (data_limit / 100))  or (((step-increase_step) > 1000))# and (step > (steps/20)))
        #         #     # increase_data = loss_out <= (inc_crit * 0.97 ** (data_limit / 100)) or ((step - increase_step) > 500)
        #         # # if not increase_data and data_limit < train_labels.shape[0] and step > 20 and  (step-increase_step) > 5 :
        #         #     increase_data = loss_out <= inc_crit or (step-increase_step) > 100
        #         increase_data = np.mean(loss_arr[-10:]) <= (inc_crit * 0.97 ** (data_limit / 100)) or (
        #             (step - increase_step) > 50)
        # if len(history)
        # increase_data = np.mean(history[epoch]['loss'])
        #                 <= (inc_crit * 0.97 ** (data_limit / 100)) or (
        #     (step - increase_step) > 50)
        #
        # if epoch < 2:
        #     data_limit = np.int(np.ceil(len_x * 0.1))  # 10000
        # elif epoch < 4:
        #     data_limit = np.int(np.ceil(len_x * 0.3))  # 20000
        # elif epoch < 6:
        #     data_limit = np.int(np.ceil(len_x * 0.5))  # 30000
        # elif epoch < 7:
        #     data_limit = np.int(np.ceil(len_x * 0.6))  # 30000
        # elif epoch < 9:
        #     data_limit = np.int(np.ceil(len_x * 0.9))  # 30000
        # elif epoch < 10:
        #     data_limit = np.int(np.ceil(len_x * 0.95))
        # else:
        #     data_limit = np.int(np.ceil(len_x * 1))  # 50000
        # print(data_iter.data_limit)
        # return data_limit


    def cl_scheduler6(data_iter):
        # print(data_iter.image_data_generator.history[data_iter.dc_idx]['loss'])
        # print(data_iter.epoch)
        # print(type(data_iter.epoch))
        # data_limit = data_iter.data_limit
        # print('dsfdfsdf', data_iter.epoch, data_iter.dc_idx)
        if (data_iter.epoch < 2):
            data_iter.data_limit = np.int(np.ceil(len_x * 0.1))
        elif (data_iter.epoch > data_iter.dc_idx + 1):
            loss_prev = data_iter.image_data_generator.history[data_iter.dc_idx]['loss']
            loss_cur = data_iter.image_data_generator.history[(data_iter.epoch - 2)]['loss']
            # loss_cur = data_iter.image_data_generator.history[(data_iter.epoch-1)]['loss']
            # print("ratio",(loss_cur/loss_prev ) )
            if (((loss_cur/loss_prev ) < 0.999) or (data_iter.epoch - data_iter.dc_idx >= 2)):
                data_iter.data_limit += np.int(np.ceil(len_x * 0.1))
                data_iter.data_limit = np.min((len_x,data_iter.data_limit))
                # data_iter.data_limit = data_limit
                data_iter.dc_idx = data_iter.epoch
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',data_iter.data_limit)



    def cl_scheduler7(data_iter):
        # print(data_iter.image_data_generator.history[data_iter.dc_idx]['loss'])
        # print(data_iter.epoch)
        # print(type(data_iter.epoch))
        # data_limit = data_iter.data_limit
        # print('dsfdfsdf', data_iter.epoch, data_iter.dc_idx)
        if (data_iter.epoch < 2):
            data_iter.data_limit = np.int(np.ceil(len_x * 0.3))
        elif (data_iter.epoch > data_iter.dc_idx + 1):
            loss_prev = data_iter.image_data_generator.history[data_iter.dc_idx]['loss']
            loss_cur = data_iter.image_data_generator.history[(data_iter.epoch - 2)]['loss']
            # loss_cur = data_iter.image_data_generator.history[(data_iter.epoch-1)]['loss']
            # print("ratio",(loss_cur/loss_prev ) )
            if (((loss_cur/loss_prev ) < 0.999) or (data_iter.epoch - data_iter.dc_idx >= 1)):
                data_iter.data_limit += np.int(np.ceil(len_x * 0.2))
                data_iter.data_limit = np.min((len_x,data_iter.data_limit))
                # data_iter.data_limit = data_limit
                data_iter.dc_idx = data_iter.epoch
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',data_iter.data_limit)


    if CURRICULUM_SCHEDULER_TYPE == "naive":
        return cl_scheduler1
    if CURRICULUM_SCHEDULER_TYPE == "naive_small":
        return cl_scheduler2
    if CURRICULUM_SCHEDULER_TYPE == "naive_large":
        return cl_scheduler3
    if CURRICULUM_SCHEDULER_TYPE == "adaptive":
        return adaptive2
    if CURRICULUM_SCHEDULER_TYPE == "adaptive_small":
        return adaptive_small
    if CURRICULUM_SCHEDULER_TYPE == "adaptive_stl":
        return cl_scheduler6
    if CURRICULUM_SCHEDULER_TYPE == "adaptive_stl_adam":
        return cl_scheduler7




class TrainHistory(keras.callbacks.Callback):
    def __init__(self, datagen):
        super(TrainHistory, self).__init__()
        self.datagen = datagen



    def on_epoch_end(self, epoch, logs={}):
        logs["loss"], logs["acc"] = self.model.evaluate(self.datagen.x_train,
                                                        self.datagen.y_train,
                                                        batch_size=100, verbose=1)

        #print(self.model.evaluate(self.datagen.x_test,
        #                          self.datagen.y_test,
        #                          batch_size=100, verbose=1))
        self.datagen.history[epoch+1] = {'acc': logs['acc'], 'val_acc': logs['val_acc'], 'val_loss': logs['val_loss'], 'loss': logs['loss']}
        # print(self.datagen.history)
        # print(self.datagen.history)
        print(logs["acc"])


def create_gradients_fetcher(model, tensor):
    loss = K.categorical_crossentropy(model.targets, model.output)
    grads = K.gradients(loss, tensor)
    fetch_grads = K.function([model.input, model.targets[-1], K.learning_phase()], grads)
    grad_length = 0
    for t in tensor:
        grad_length += np.prod(K.int_shape(t))
        # print(grad_length)

    # print(model.summary())
    # exit()
    def gradients_fetcher(x, y, batch_size=500):
        # for te in tensor:
        #     print(*K.int_shape(te))

        n_batches = math.ceil(x.shape[0] / batch_size)
        grads = np.zeros([n_batches, grad_length], dtype='float32')

        for i in range(n_batches):
            print(i)
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size

            x_batch = x[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]

            grad = fetch_grads([x_batch, y_batch, 0.0])
            # for g in grad:
            #     print(g.shape)
            # print(grad)
            # exit()
            # print(np.concatenate([g.flatten() for g in grad]).shape)
            # grads[batch_start:batch_end] = np.concatenate([g.flatten() for g in grad])
            grads[i:i + 1] = np.concatenate([g.flatten() for g in grad])

        return grads

    return gradients_fetcher


# def calc_grads(model, x, y):
#     grads_history
#     listOfVariableTensors = model.trainable_weights
#     gradients_fetcher = create_gradients_fetcher(model, listOfVariableTensors)
#
#
#
#     # grads1 = gradients_fetcher(x, y, batch_size=2500)
#     for i in range(len(x)):
#         grads2 = gradients_fetcher(x[i], y[i], batch_size=1)
#
#         euc_dist = np.sum(np.subtract(grads1, grads2) ** 2)
#         cos_dist = scipy.spatial.distance.cosine(grads1, grads2)
#         print("samples  batch #{0}:    dist= {1}    cos-sim={2}".format(b, euc_dist, 1 - cos_dist))
#         # print("samples  {0}:    ".format(b), 1-scipy.spatial.distance.cosine(grads1, grads2))
#
#         grads_history[(i, b)].append((euc_dist, 1 - cos_dist))


def run_experiments(dataset, net_type="large", optimizer="sgd", initial_lr=2e-3,
                    batch_size=100, num_epochs=100, num_exps=20, num_repts=5, l2_reg=200e-4, bias_l2_reg=None,
                    curriculum="None", sorted_indices=None, data_augmentation=False, comp_grads=False):
    cache_file = os.path.join(dataset.data_path, 'data.pkl')
    (x_train, cls_train, y_train), (x_test, cls_test, y_test) = dataset.load_data_cache(cache_file)

    # x_train = x_train.astype('float32')
    # x_train /= 255
    x_train = (x_train - 128.) / 128
    # x_test = x_test.astype('float32')
    x_test = (x_test - 128.) / 128

    if data_augmentation == "True":
        datagen = DataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            # rescale=1. / 255,
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=True,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True  # randomly flip images
        )
    else:
        print("No Aug")
        datagen = DataGenerator()

    index_array = np.arange(len(x_train))
    datagen.subset_index_array = index_array
    datagen.sorted_indices = sorted_indices
    datagen.steps_per_epoch = len(x_train) / batch_size
    datagen.num_classes = dataset.num_classes
    datagen.curriculum = False
    datagen.x_test = x_test
    datagen.y_test = y_test

    # print(sorted_indices.shape)
    x_train = x_train[sorted_indices]
    # print(x_train.shape)
    # exit()
    y_train = y_train[sorted_indices]
    datagen.x_train = x_train
    datagen.y_train = y_train
    if curriculum != "None":  # curriculum
        # x_train = x_train[sorted_indices]
        # y_train = y_train[sorted_indices]
        datagen.curriculum = True
        datagen.curriculum_schedule = get_curriculum_schedule(len(x_train))


    if comp_grads:
        comp_grads = 'comp_grads/'
    else:
        comp_grads = ''

    # results_path = os.path.join(dataset.results_path,
    #                             net_type + "_acc8/" + comp_grads + optimizer + "/" + str(initial_lr) + "/" + str(
    #                                 l2_reg) + "/" + curriculum + "/")
    results_path = os.path.join(dataset.results_path,
                                net_type + "/" + comp_grads + optimizer + "2/" + str(initial_lr) + "/" + str(
                                    l2_reg) + "/" + curriculum + "/")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    print(results_path)

    reduce_lr = LearningRateScheduler(get_lr_scheduler(initial_lr))
    train_acc = TrainHistory(datagen)

    if "subset" in dataset.data_path:
        dataset.data_path = subset_model_path
    ######################################################################  testing
    with open(os.path.join(dataset.data_path, 'svm_results.pkl'), mode='rb') as file:
        prob_estimates, preds_svm, _, _, _, _ = pickle.load(file)
    ######################################################################  testing end

    for exp in range(0, num_exps):
        grads_history1 = defaultdict(list)
        grads_history2 = defaultdict(list)
        print("Experiment  ", exp)
        results_path_ = os.path.join(results_path, "exp{0}/".format(exp))
        if not os.path.exists(results_path_):
            os.makedirs(results_path_)
        for rpt in range(num_repts):
            print("Rept.  ", rpt)
            old_model = load_model(
                os.path.join(dataset.data_path, 'model/' + net_type + '/model_init_{0}.h5'.format(exp)))
            model = create_model(net_type=net_type, n_classes=dataset.num_classes, reg_factor=l2_reg,
                                 bias_reg_factor=bias_l2_reg)
            model.set_weights(old_model.get_weights())
            # K.set_value(model.l2_reg, K.cast_to_floatx(100e-3))
            # K.set_value(model.l2_bias_reg, K.cast_to_floatx(100e-3))
            if optimizer == "adam":
                opt = keras.optimizers.adam(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                                  amsgrad=False)
            elif optimizer == "sgd":
                opt = keras.optimizers.sgd(lr=initial_lr)

            model.compile(
                loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy']
            )
            model.summary()
            listOfVariableTensors = model.trainable_weights
            gradients_fetcher = create_gradients_fetcher(model, listOfVariableTensors)

            # ######################################################################  testing
            # history = model.fit(x=x_train, y=y_train,
            #                     batch_size=batch_size,
            #                     epochs=num_epochs,
            #                     verbose=2,
            #                     sample_weight=[],
            #                     validation_data=(x_test, y_test),
            #                     callbacks=[train_acc, reduce_lr],
            #                     steps_per_epoch=len(x_train) / batch_size)
            # ######################################################################  testing end

            loss,acc =  model.evaluate(x_train, y_train, batch_size=100, verbose=1)
            val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=100, verbose=1)
            train_acc.datagen.history = defaultdict(list)
            train_acc.datagen.history[0] = {'acc': acc, 'val_acc':val_acc, 'val_loss': val_loss, 'loss': loss}
            print({'acc': acc, 'val_acc': val_acc, 'val_loss': val_loss, 'loss': loss})


            if comp_grads:

                num_batches = np.int(len(x_train) / batch_size)
                for e in range(num_epochs + 1):
                    print("Epoch #", e)

                    # (euc_dist, cos_dist, sim, rad, angle) = grads_comp_update(model, x_train, y_train, batch_size)
                    #
                    # grads_history1[(rpt, e, i)].append((euc_dist, cos_dist, sim, rad, angle))
                    #
                    #
                    #

                    grads_full = gradients_fetcher(x_train, y_train, batch_size=1)
                    grads1 = gradients_fetcher(x_train, y_train, batch_size=len(x_train))#2500)

                    for i in range(len(x_train)):
                        grads2 = grads_full[i]
                        euc_dist = np.sum(np.subtract(grads1, grads2) ** 2)
                        cos_dist = scipy.spatial.distance.cosine(grads1, grads2)
                        rad = 1 - cos_dist
                        sim = 1 - cos_dist
                        angle = np.degrees(np.arccos(rad))
                        grads_history1[(rpt, e, i)].append((euc_dist, cos_dist, sim, rad, angle))
                        #print((euc_dist, cos_dist, sim, rad, angle))

                    for b in range(num_batches):
                        grads2 = gradients_fetcher(x_train[b * batch_size:(b + 1) * batch_size],
                                                   y_train[b * batch_size:(b + 1) * batch_size], batch_size=batch_size)
                        euc_dist = np.sum(np.subtract(grads1, grads2) ** 2)
                        cos_dist = scipy.spatial.distance.cosine(grads1, grads2)
                        rad = 1 - cos_dist
                        sim = 1 - cos_dist
                        angle = np.degrees(np.arccos(rad))
                        grads_history2[(rpt, e, b)].append((euc_dist, cos_dist, sim, rad, angle))
                        #print((euc_dist, cos_dist, sim, rad, angle))




                        # grads_history1[(rpt, e)] = gradients_fetcher(x_train, y_train, batch_size=1)
                        # grads1 = gradients_fetcher(x_train, y_train, batch_size=2500)
                        # grads_history2[(rpt, e,num_batches)].append(grads1)
                        # for b in range(num_batches):
                        #     grads2 = gradients_fetcher(x_train[b * batch_size:(b + 1) * batch_size],
                        #                                y_train[b * batch_size:(b + 1) * batch_size], batch_size=batch_size)
                        #     grads_history2[(rpt, e, num_batches)].append(grads2)



                        #   euc_dist = np.sum(np.subtract(grads1, grads2) ** 2)
                        #    cos_dist = scipy.spatial.distance.cosine(grads1, grads2)
                        #    print("samples  batch #{0}:    dist= {1}    cos-sim={2}".format(b, euc_dist, cos_dist))
                        # print("samples  {0}:    ".format(b), 1-scipy.spatial.distance.cosine(grads1, grads2))

                        # grads_history[(i,b)].append((euc_dist, 1-cos_dist))


                        # grads1 = gradients_fetcher(x_train, y_train, batch_size=2500)#np.sum(grads_history[e], axis=0)
                        # print(grads_history[e].shape)
                        # print(grads1.shape)
                        #
                        # for i in range(len(x_train)):
                        #     grads2 = grads_history[e][i]
                        #     euc_dist = np.sum(np.subtract(grads1, grads2) ** 2)
                        #     cos_dist = scipy.spatial.distance.cosine(grads1, grads2)
                        #
                        #     print("sample  #{0}:    dist= {1}    cos-sim={2} ".format(i, euc_dist, cos_dist))
                        #     angle = math.degrees(math.acos(cos_dist))
                        #     print("angle = {0}".format(angle))

                        #
                        #     batch_grad = grads_history[e]
                        #
                        #     grads2 = gradients_fetcher(x_train[b * batch_size:(b + 1) * batch_size],
                        #                                y_train[b * batch_size:(b + 1) * batch_size], batch_size=batch_size)
                        #
                        #     euc_dist = np.sum(np.subtract(grads1, grads2) ** 2)
                        #     cos_dist = scipy.spatial.distance.cosine(grads1, grads2)
                        #     print("samples  batch #{0}:    dist= {1}    cos-sim={2}".format(b, euc_dist, 1-cos_dist))
                        #     # print("samples  {0}:    ".format(b), 1-scipy.spatial.distance.cosine(grads1, grads2))
                        #
                        # grads1 = gradients_fetcher(x_train, y_train, batch_size=2500)
                        # print("DIFF    ", np.sum(np.abs(np.sum(grads_history[e], axis=0) - np.sum(grads1, axis=0))))


                        # grads1 = gradients_fetcher(x_train[:100], y_train[:100], batch_size=100)
                        # grads2 = gradients_fetcher(x_train[:100], y_train[:100], batch_size=10)
                        # grads3 = gradients_fetcher(x_train[:100], y_train[:100], batch_size=1)
                        # grads4 = gradients_fetcher(x_train[:100], y_train[:100], batch_size=1)
                        # print(np.sum(grads1,axis=0).shape)
                        # print(np.sum(grads2,axis=0).shape)
                        # print("DIFF    ", np.sum(np.abs(np.sum(grads1,axis=0) - np.sum(grads2,axis=0))))
                        # print("DIFF    ", np.sum(np.abs(np.sum(grads3) - np.sum(grads4))))
                        # print("DIFF    ", np.sum(np.abs(np.sum(grads1) - np.sum(grads3))))

                        # grads2 = gradients_fetcher(x_train[:10], y_train[:10], batch_size=1)
                        # grads4 = gradients_fetcher(x_train[:10], y_train[:10], batch_size=1)
                        #
                        #
                        # grads_history[e] = gradients_fetcher(x_train[:100], y_train[:100], batch_size=1)
                        # grads1 = gradients_fetcher(x_train[:100], y_train[:100], batch_size=10)
                        # print(np.sum(np.sqrt((grads1 - np.sum(grads_history[e], axis=0))**2)))


                        # grads1 = gradients_fetcher(x_train, y_train, batch_size=2500)
                        # for b in range(np.int(len(x_train) / batch_size)):
                        #
                        #   grads2 = gradients_fetcher(x_train[b * batch_size:(b + 1) * batch_size],
                        #                               y_train[b * batch_size:(b + 1) * batch_size], batch_size=batch_size)

                        #   euc_dist = np.sum(np.subtract(grads1, grads2) ** 2)
                        #    cos_dist = scipy.spatial.distance.cosine(grads1, grads2)
                        #    print("samples  batch #{0}:    dist= {1}    cos-sim={2}".format(b, euc_dist, cos_dist))
                        # print("samples  {0}:    ".format(b), 1-scipy.spatial.distance.cosine(grads1, grads2))

                        # grads_history[(i,b)].append((euc_dist, 1-cos_dist))

                    history = model.fit(x=x_train, y=y_train,
                                        batch_size=100,  # batch_size,
                                        epochs=1,
                                        verbose=2,
                                        sample_weight=[],
                                        validation_data=(x_test, y_test),
                                        callbacks=[train_acc])






                    # grads_history1[(rpt, num_epochs)] = gradients_fetcher(x_train, y_train, batch_size=1)
                    # grads1 = gradients_fetcher(x_train, y_train, batch_size=2500)
                    # grads_history2[(rpt, num_epochs, num_batches)].append(grads1)
                    # for b in range(num_batches):
                    #     grads2 = gradients_fetcher(x_train[b * batch_size:(b + 1) * batch_size],
                    #                                y_train[b * batch_size:(b + 1) * batch_size], batch_size=batch_size)
                    #     grads_history2[(rpt, num_epochs, num_batches)].append(grads2)
                    # with open(os.path.join(results_path_, "gradsHistoryDict1_{0}".format(rpt)), 'wb') as file_pi:
                    #     pickle.dump(grads_history1, file_pi, protocol=4)
                    # with open(os.path.join(results_path_, "gradsHistoryDict2_{0}".format(rpt)), 'wb') as file_pi:
                    #     pickle.dump(grads_history2, file_pi, protocol=4)
            else:
                history = model.fit_generator(generator=datagen.flow(x=x_train, y=y_train, batch_size=batch_size,
                                                                     shuffle=True, seed=seed),
                                              steps_per_epoch=len(x_train) / batch_size,
                                              epochs=num_epochs,
                                              verbose=2,
                                              validation_data=(x_test, y_test),
                                              callbacks=[train_acc, reduce_lr],
                                              workers=4)
                # print(x_train.shape)
                # grads1 = gradients_fetcher(x_train[:10], y_train[:10], batch_size=10)
                # grads3 = gradients_fetcher(x_train[:10], y_train[:10], batch_size=10)
                # grads2 = gradients_fetcher(x_train[:10], y_train[:10], batch_size=1)
                # grads4 = gradients_fetcher(x_train[:10], y_train[:10], batch_size=1)



                # # print(np.std(grads2))
                # print(np.sum(np.sqrt((grads1 - np.mean(grads2, axis=0))**2)))
                #
                # print(np.sum((grads1 - np.mean(grads2[:100], axis=0)) ** 2))
                #
                # print(np.sum((grads1 - np.mean(grads2[-100:], axis=0)) ** 2))
                # print(np.sum(np.abs(grads1 - np.mean(grads2, axis=0))))

                # exit()


            model.save(os.path.join(results_path_, "model_trained{0}.h5".format(rpt)))
            with open(os.path.join(results_path_, "trainHistoryDict{0}".format(rpt)), 'wb') as file_pi:
                pickle.dump(train_acc.datagen.history, file_pi)
        if comp_grads:
            with open(os.path.join(results_path_, "gradsHistoryDict1"), 'wb') as file_pi:
                pickle.dump(grads_history1, file_pi, protocol=4)
            with open(os.path.join(results_path_, "gradsHistoryDict2"), 'wb') as file_pi:
                pickle.dump(grads_history2, file_pi, protocol=4)

    return None

# res_path = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/large/comp_grads/sgd/0.05/0.001/None/exp{0}/'
# res_path2 = res_path.format(0)
# with open(os.path.join(res_path2, "gradsHistoryDict2"), 'rb') as file_pi:
#       grads_history_temp = pickle.load(file_pi)
#
#
#
#
#
# results = np.zeros((2, 5, 40, 25,5))
# for i in range(2):
#     results_path_ = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/large/comp_grads/sgd/0.05/0.001/None/exp{0}/'.format(str(i))
#     with open(os.path.join(results_path_, "gradsHistoryDict2"), 'rb') as file_pi:
#         grads_history = pickle.load(file_pi)
#     for rpt in range(5):
#         for epoch in range(40):
#             for b in range(25):
#                 results[i, rpt, epoch,b,0:5] = grads_history[(rpt, epoch,b)][0][0:5]
# np.save('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/res1_b.npy', results)
#
#
# #res = np.mean(results,axis=1)
# np.save('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/res1_b.npy',results)
# 0,10,
#
#
#
#
#
#
#
#
#
#
#
# results = np.zeros((1, 5, 40, 2500,5))
# for i in range(1):
#     results_path_ = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/large/comp_grads/sgd/0.05/0.001/None/exp{0}/'.format(str(i))
#     with open(os.path.join(results_path_, "gradsHistoryDict1"), 'rb') as file_pi:
#         grads_history = pickle.load(file_pi)
#     for rpt in range(5):
#         for epoch in range(40):
#             for b in range(2500):
#                 results[i, rpt, epoch,b,0:5] = grads_history[(rpt, epoch,b)][0][0:5]
# res = np.mean(results,axis=1)
# np.save('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/res1.npy',res)

#
# results = np.zeros((20, 25, 10))
# for i in range(10):
#     results_path_ = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset4/results/comp_grads/sgd/0.1/0.001/None/exp{0}/'.format(str(i))
#     with open(os.path.join(results_path_, "gradsHistoryDict"), 'rb') as file_pi:
#         grads_history = pickle.load(file_pi)
#     for iter in range(20):
#         for b in range(25):
#             results[iter,b,i] = grads_history[(iter,b)][0][1]
# res = np.mean(results,axis=2)
# np.save('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/comp_grads/res4.npy',res)


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
    parser.add_argument("--sorted_indices_file", default="sorted_indices.pkl")
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

    # print(destination)
    # print(args.net_type)
    # print(dataset.num_classes)
    # exit()
    # generate_init_weights(destination,n_classes=dataset.num_classes, net_type=args.net_type)
    # exit()
    if args.generate_weights == "True":
        generate_init_weights(destination, n_classes=dataset.num_classes, net_type=args.net_type)
    comp_grads = False
    if args.comp_grads == "True":
        comp_grads = True

    # try:
    #     a = json.loads("false")
    # except:
    #     pass

    sorted_indices = None
    sorted_indices = unpickle(os.path.join(dataset.data_path, args.sorted_indices_file)).astype(np.int).reshape(-1, )
    if args.curriculum != "None":
        CURRICULUM_SCHEDULER_TYPE = args.curriculum_scheduler
        sorted_indices = unpickle(os.path.join(dataset.data_path, args.sorted_indices_file)).astype(np.int)
        # sorted_indices = unpickle(os.path.join(dataset.data_path,'sorted_indices2.pkl')).astype(np.int)
        if args.curriculum == "anti-curriculum":
            sorted_indices = np.flip(sorted_indices.reshape(-1, ), axis=0)
        elif args.curriculum == "control-curriculum":
            for j in range(dataset.num_classes):
                np.random.shuffle(sorted_indices[:, j])
            sorted_indices = sorted_indices.reshape(-1, )
        elif "curriculum" in args.curriculum:
            sorted_indices = sorted_indices.reshape(-1, )
            print(args.curriculum)

    LR_SCHEDULER_TYPE = args.lr_sched
    # print(sorted_indices)
    run_experiments(dataset=dataset, net_type=args.net_type, optimizer=args.optimizer, initial_lr=args.learning_rate,
                    batch_size=args.batch_size, num_epochs=args.num_epochs, num_exps=args.exp, num_repts=args.rept,
                    l2_reg=args.l2_reg, bias_l2_reg=args.bias_l2_reg,
                    curriculum=args.curriculum, sorted_indices=sorted_indices, data_augmentation=args.data_aug,
                    comp_grads=comp_grads)

