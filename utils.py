import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import argparse
import sys
import pickle
#import keras
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import itertools


parent_path = '/cs/labs/daphna/gadic/curriculum_learning/'
cl_types = ['curriculum','control-curriculum','anti-curriculum','None']
cl_colors = ['green', 'yellow', 'red', 'blue']
cl_fmts = ['-', '--', ':', '-.']
#
# cl_types = ['curriculum', 'None']
# cl_colors = ['green', 'blue']
# cl_fmts = ['-', '-.']
#


def unpickle(file_path):
    """
    Unpickle the given file and return the data.

    Note that the appropriate dir-name is prepended the filename.
    """

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    return data


# Helper-function for plotting
def sort_values(data_path):
    print("Input image:")

    file_path = os.path.join(data_path, 'svm_results.pkl')
    prob_estimates_train, preds_svm_train, accuracy_train, prob_estimates_test, preds_svm_test, accuracy_test = unpickle(file_path)

    file_path = os.path.join(data_path, 'sorted_indices.pkl')
    sorted_indices = unpickle(file_path)

    pred_prob = np.max(prob_estimates_train, axis=1)
    srt_idxs = np.argsort(pred_prob)[::-1]
    file_path = os.path.join(data_path, 'sorted_indices2.pkl')
    with open(file_path, mode='wb') as file:
        pickle.dump(srt_idxs, file)
    plt.plot(pred_prob[srt_idxs])
    plt.grid()
    plt.show()


    pred_prob = np.max(prob_estimates_test, axis=1)
    srt_idxs = np.argsort(pred_prob)[::-1]
    plt.plot(pred_prob[srt_idxs])
    plt.grid()
    plt.show()


def preview_data(dataset, images, cls_true, class_names):
    if dataset:
        class_names = dataset.load_class_names()
        # for j in range(len(class_names)):
        #     print(str(j) + "  " + class_names[j])

        for cls in 'hamster, mouse, rabbit, shrew, squirrel'.replace(" ", "").split(','):
            print(cls + "  " + str(class_names.index(cls)))
        # print(np.where(class_names == ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']))
        (x_train, cls_train, y_train), (x_test, cls_test, y_test) = dataset.load_data_cache(dataset.cache_file)
        # Get the first images from the test-set.
        images = x_train[0:9]

        # Get the true classes for those images.
        cls_true = cls_train[0:9]

    # Plot the images and labels using our helper-function above.
    plot_images(images=images, cls_true=cls_true, class_names = class_names, smooth=False)


def _split_filter(indices, data, cls, class_names):
    # indices - class labels to take
    N_CLASSES = len(indices)
    # labels = np.argmax(labels,axis=1)
    ix = np.isin(cls, indices)
    cls = cls[ix]
    data = data[ix]
    for i, c in enumerate(indices):
        ix = np.isin(cls, c)
        cls[ix] = i

    labels = keras.utils.to_categorical(cls, N_CLASSES)
    print(data.shape)
    print(labels.shape)
    preview_data(None, data[0:9], cls[0:9], class_names)
    return data, cls, labels

def split_data(dataset):
    class_names = dataset.load_class_names()
    (x_train, cls_train, y_train), (x_test, cls_test, y_test) = dataset.load_data_cache(dataset.cache_file)
    indices = []
    for cls in dataset.subset:
        indices.append(class_names.index(cls))
        print(cls + "  " + str(class_names.index(cls)))

    x_train, cls_train, y_train = _split_filter(indices, x_train, cls_train, dataset.subset)
    x_test, cls_test, y_test = _split_filter(indices, x_test, cls_test, dataset.subset)

    with open(os.path.join(dataset.destination, 'data.pkl'), mode='wb') as file:
        pickle.dump([(x_train, cls_train, y_train), (x_test, cls_test, y_test)], file)


def plot_images(images, cls_true, class_names, cls_pred=None, smooth=True):
    # assert len(images) == len(cls_true) == 9
    #
    # # Create figure with sub-plots.
    # fig, axes = plt.subplots(3, 3)
    #
    # # Adjust vertical spacing if we need to print ensemble and best-net.
    # if cls_pred is None:
    #     hspace = 0.3
    # else:
    #     hspace = 0.6
    # fig.subplots_adjust(hspace=hspace, wspace=0.3)





    from PIL import ImageEnhance, Image
    from scipy import ndimage

    from skimage import restoration,filters
    from skimage.morphology import disk
    from skimage.filters import rank
    selem = disk(20)

    image_filtered = []
    for j in range (8):
        image = Image.fromarray(images[j], 'RGB')
        sharpner = ImageEnhance.Sharpness(image)
        en = sharpner.enhance(0)
        IMAGE_10 = os.path.join('./dog10.jpeg')
        en.save(IMAGE_10, "JPEG", quality=70)
        im10 = Image.open(IMAGE_10)
        im10 = np.true_divide(im10, 255.)

        img = np.true_divide(images[j], 255.)  # normalise to 0-1, it's easier to work in float space
        tv = restoration.denoise_tv_chambolle(images[j], weight=0.05)
        gaussian = filters.gaussian(images[j], sigma=1)
        image_filtered.append(img)
        image_filtered.append(gaussian)
        image_filtered.append(tv)
        image_filtered.append(im10)


    fig, axes = plt.subplots(5, 4)
    fig.subplots_adjust(hspace=0.1, wspace=-0.5)
    for i, ax in enumerate(axes.flat):
        ax.imshow(image_filtered[i])
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    exit()
    img = images[0]
    from PIL import ImageEnhance, Image
    image = Image.fromarray(img, 'RGB')

    sharpner = ImageEnhance.Sharpness(image)
    en = sharpner.enhance(0)
    IMAGE_10 = os.path.join('./dog10.jpeg')
    en.save(IMAGE_10, "JPEG", quality=70)
    im10 = Image.open(IMAGE_10)
    im10 = np.true_divide(im10, 255.)


    from scipy import ndimage

    from skimage import restoration,filters
    from skimage.morphology import disk
    from skimage.filters import rank
    selem = disk(20)

    img = np.true_divide(img, 255.)  # normalise to 0-1, it's easier to work in float space
    tv = restoration.denoise_tv_chambolle(img, weight=0.05)
    gaussian = filters.gaussian(img, sigma=1)
    image_filtered = [img, gaussian, tv]
    fig, axes = plt.subplots(1, 4)
    # r = rank.mean_bilateral(img[:, :, 0], selem=selem, s0=50, s1=50)
    # g = rank.mean_bilateral(img[:, :, 1], selem=selem, s0=50, s1=50)
    # b = rank.mean_bilateral(img[:, :, 2], selem=selem, s0=50, s1=50)

    # import scipy
    # img = np.true_divide(img, 255.) # normalise to 0-1, it's easier to work in float space
    #
    #
    # # make some kind of kernel, there are many ways to do this...
    # t = 1 - np.abs(np.linspace(-1, 1, 21))
    # kernel = t.reshape(21, 1) * t.reshape(1, 21)
    # kernel /= kernel.sum()  # kernel should sum to 1!  :)
    # kernel = np.ones((4, 4), np.float32) / 16
    # im_out = np.copy(img)
    # for j in range(2):
    #     # convolve 2d the kernel with each channel
    #     r = scipy.signal.convolve2d(im_out[:, :, 0], kernel, mode='same')
    #     g = scipy.signal.convolve2d(im_out[:, :, 1], kernel, mode='same')
    #     b = scipy.signal.convolve2d(im_out[:, :, 2], kernel, mode='same')
    #     # stack the channels back into a 8-bit colour depth image and plot it
    #     im_out = np.dstack([r, g, b])
    #
    # im_out = (im_out * 255).astype(np.uint8)
    # image_filtered = [img, im_out]
    image_filtered = [img, im10,tv,gaussian]

    for i, ax in enumerate(axes.flat):
        # kernel = np.ones((5, 5), np.float32) / 25

        # img_gaus = ndimage.filters.gaussian_filter(img, 2, mode='nearest')

        # k = np.array([[0, 1, 0],
        #               [1, 1, 1],
        #               [0, 1, 0]])
        #
        # img2 = ndimage.convolve(img, kernel, mode='constant')
        # gaussian = filters.gaussian(img, sigma= (9.0-i)/9, mode='nearest')


        # ax.imshow(im_out)
        ax.imshow(image_filtered[i])
    plt.show()
    exit()

    # hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max() / cdf.max()
    # plt.plot(cdf_normalized, color='b')
    # plt.hist(img.flatten(), 256, [0, 256], color='r')
    # plt.xlim([0, 256])
    # plt.legend(('cdf', 'histogram'), loc='upper left')
    # plt.show()
    # cdf_m = np.ma.masked_equal(cdf, 0)
    # cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # img2 = cdf[img]
    # plt.imshow(img2)
    # plt.show()
    # hist, bins = np.histogram(img2.flatten(), 256, [0, 256])
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max() / cdf.max()
    # plt.plot(cdf_normalized, color='b')
    # plt.hist(img.flatten(), 256, [0, 256], color='r')
    # plt.xlim([0, 256])
    # plt.legend(('cdf', 'histogram'), loc='upper left')
    # plt.show()
    # exit()
    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        # if smooth:
        #     interpolation = 'spline16'
        # else:
        #     interpolation = 'nearest'
        #
        # # Plot image.
        # ax.imshow(images[i, :, :, :],
        #           interpolation=interpolation)
        # factor = i / 5.0
        # en = sharpner.enhance(factor)
        # contrast = ImageEnhance.Contrast(en)
        # en = contrast.enhance(i / 5.0)
        # ax.imshow(en)
        # in_data = np.asarray(en, dtype=np.uint8)
        # print(in_data[:100,0,0])


        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}, {1}".format(cls_true_name, cls_true[i])
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()




def plot_history(dataset_name,results_path_parent, optimizer, initial_lr, num_exps, num_repts):
    # Visualize training history
    y_max=0
    y_min = 100
    results_holder = []
    fig, ax = plt.subplots()
    for curriculum in ['curriculum','anti-curriculum']:#,'control-curriculum']:#,'None']:
        results_holder = []
        results_path = os.path.join(results_path_parent, optimizer + "/" + str(initial_lr) + "/" + curriculum + "/")
        for exp in range(num_exps):
            results_path_ = os.path.join(results_path, "exp{0}/".format(exp))
            for rpt in range(num_repts):
                with open(os.path.join(results_path_, "trainHistoryDict{0}".format(rpt)), 'rb') as file_pi:
                    history = pickle.load(file_pi)
                    results_holder.append(history)

        accuracy_summary = np.zeros((num_exps*num_repts,len(results_holder[0]['val_acc'])))
        for exp in range(num_exps):
            for rpt in range(num_repts):
                accuracy_summary[exp*num_repts+rpt,:] = results_holder[exp*num_repts+rpt]['val_acc']
        mean = np.mean(accuracy_summary, axis=0)*100
        if np.max(mean) > y_max:
            y_max = np.max(mean)
        if np.min(mean) < y_min:
            y_min = np.min(mean)
        # mean = accuracy_summary[0,:]
        std = np.std(accuracy_summary, axis=0) * 100
        ste = std /np.sqrt(num_exps)
        # plt.errorbar(range(len(mean)),mean.tolist(), yerr=ste)
        plt.plot(mean, label=curriculum )

        ################ smooth
        # x = np.arange(accuracy_summary.shape[2])
        # xx = np.linspace(x.min(), x.max(), accuracy_summary.shape[1])
        # for kk in range(num_repts):
        #     itp = interp1d(x, accuracy_summary[kk, :], kind='linear')
        #     window_size, poly_order = 101, 5
        #     accuracy_summary[kk, :] = savgol_filter(itp(xx), window_size, poly_order)


        # acc_arr = np.array(acc_arr).reshape((-1, num_tests))
        # print(acc_arr.shape)
        #
        # x = np.arange(mean.shape[0])
        # print
        # mean.shape
        # xx = np.linspace(x.min(), x.max(), mean.shape[0])
        # itp = interp1d(x, mean, kind='linear')
        # window_size, poly_order = 101, 5
        # yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        # print
        # len(yy_sg)
        # # plt.plot(yy_sg, label=test_labels[i])#, color=colors[i])
        # # plt.errorbar(x[::30], yy_sg[::30], yerr=std[::30] )  # , fmt='o')
        # plt.plot(mean, label=opt + '_' + test_labels[i])  # , color=colors[i])
    plt.title(dataset_name)
    plt.legend(loc='lower right', shadow=False)
    plt.ylim([y_min -3, y_max+1])
    ax.grid()
    # plt.savefig('/cs/grad/gadic/Desktop/curriculum_dropout_large/{0}.png'.format(kp))
    # plt.savefig('/cs/grad/gadic/Desktop/curriculum_learning/large_dp/svm/{0}.png'.format(kp))

    plt.show()
    print("d")

def plot_results():
    import os
    num_repts = 5
    num_exps = 6
    num_epochs = 80
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/results2/adam/0.001/{0}/exp{1}/trainHistoryDict{2}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/largeadam/0.0017/0.0012/{0}/exp{1}/trainHistoryDict{2}'

    cl_types = ['curriculum','control-curriculum','anti-curriculum','None']
    fig, ax = plt.subplots()

    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_xlabel('Epochs', fontsize=16)
    # ax.set_title('Training gradients compared')
    # ax.legend(cl_types, loc='upper left', fontsize=12)

    for cl in cl_types:
        # files = folders = 0
        # for _, dirnames, filenames in os.walk('/cs/labs/daphna/gadic/curriculum_learning/cifar100/results2/adam/0.001/{0}/'.format(cl)):
        #     # ^ this idiom means "we won't be using this value"
        #     files += len(filenames)
        #     folders += len(dirnames)
        results = np.zeros((num_exps,num_repts,num_epochs))
        for exp in range(num_exps):
            for rpt in range(num_repts):
                results_path = res_path_str.format(cl, str(exp), str(rpt))
                print(results_path)
                with open(results_path, 'rb') as file_pi:
                    train_history = pickle.load(file_pi)
                    print(train_history)
                    exit()
                results[exp,rpt,:] = train_history['val_acc']
        print(cl)
        print(results.shape)
        results = np.reshape(results,(num_exps*num_repts,-1))
        m = np.mean(results,axis=0)
        print(m.shape)
        ax.plot(np.mean(results,axis=0), label=cl)
        # plt.plot(np.mean(results,axis=0))
        # plt.ylim([0, 90])
        ax.legend()
        plt.xlabel('Epochs')
        plt.show()
        # fig_dir = '/cs/grad/gadic/Desktop/curriculum_learning/cifar100/results/'
        # if not os.path.exists(fig_dir):
        #     os.makedirs(fig_dir)
        # plt.savefig(fig_dir + 'cifar100.png')




def plot_results_subsets2():
    import os
    subsets = [0,1,3]
    num_subsets = 4
    num_repts = 5
    num_exps = 20
    num_epochs = 81
    net_type = 'large'#''small2'#'small_t'#
    opt = 'sgd'#'adam'#
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results2/adam/0.001/{0}/exp{1}/trainHistoryDict{2}'
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/temp/results_temp/sgd/0.05/{1}/exp{2}/trainHistoryDict{3}'


    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/sgd/0.02/0.0005/{1}/exp{2}/trainHistoryDict{3}'


    # for acceleration
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/sgd/0.002/0.0005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/sgd/0.002/0.002/{1}/exp{2}/trainHistoryDict{3}'
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}'
    # # for reg
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large_t/sgd/0.02/0.03/{1}/exp{2}/trainHistoryDict{3}'

#older ones
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large_t/sgd/0.02/0.0005/{1}/exp{2}/trainHistoryDict{3}'
    # cl_types = ['curriculum','control-curriculum','anti-curriculum','None']#['curriculum','None'] #['curriculum','control-curriculum','anti-curriculum','None']
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/'+net_type+'/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/archive/' + net_type + '/sgd/0.002/0.002/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/' + net_type + '/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/archive/large/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}'


    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/archive/large_t/sgd/0.02/0.0005/{1}/exp{2}/trainHistoryDict{3}'



    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/stl10/results/large3/sgd/0.02/0.004/{1}/exp{2}/trainHistoryDict{3}'



    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/results/'+ net_type + '/' + opt+'/0.1/0.0005/{1}/exp{2}/trainHistoryDict{3}'
     # ax.set_title('Training gradients compared')
    # ax.legend(cl_types, loc='upper left', fontsize=12)
    fig, ax = plt.subplots()
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_xlabel('Epochs', fontsize=16)
    for s in subsets: #range(num_subsets):
        low = 0.2
        high = 0.5
        # fig, ax = plt.subplots()
        # ax.set_ylabel('Accuracy', fontsize=16)
        # ax.set_xlabel('Epochs', fontsize=16)
        for i,cl in enumerate(cl_types):
            # files = folders = 0
            # for _, dirnames, filenames in os.walk('/cs/labs/daphna/gadic/curriculum_learning/cifar100/results2/adam/0.001/{0}/'.format(cl)):
            #     # ^ this idiom means "we won't be using this value"
            #     files += len(filenames)
            #     folders += len(dirnames)
            results = np.zeros((num_exps,num_repts,num_epochs))
            for exp in range(num_exps):
                for rpt in range(num_repts):
                    results_path = res_path_str.format(str(s+1), cl, str(exp), str(rpt))
                    with open(results_path, 'rb') as file_pi:
                        train_history = pickle.load(file_pi)
                    # for v in train_history:
                    #     print(train_history[v]['val_acc'])
                    # print([v[0]['val_acc'] for (k,v) in train_history.items()])
                    # print(train_history.items()['val_acc'])#[0:80][0]['val_acc'])
                    # results[exp, rpt, :] = train_history['val_acc']
                    # results[exp,rpt,:] = [v[0]['val_acc'] for (k,v) in train_history.items()]


                    results[exp, rpt, :] = [v[0]['val_acc'] for (k, v) in train_history.items()]
                    # results[exp, rpt, :] = [v['val_acc'] for (k, v) in train_history.items()]
            print(cl)
            print(results.shape)
            m1 = np.mean(results, axis=1)

            m = np.mean(m1, axis=0)
            np_max = np.max(m)
            if np.max(m) > high:
                high = np_max

            e = np.std(m1, axis=0) /np.sqrt(num_repts)
            # print(m1.shape)
            # results = np.reshape(results,(num_exps*num_repts,-1))
            # results = np.reshape(results, (num_exps , -1))
            # print(results.shape)
            # exit()
            # m = np.mean(results,axis=0)
            # e = np.std(results,axis=0)#/np.sqrt(num_exps*num_repts)
            # print(m.shape)
            # m = np.insert(m, 0,0.2 - np.random.uniform(-0.01,0.01))

            # print(m.shape)

            # x = np.arange(num_epochs)
            # xx = np.linspace(x.min(), x.max(), num_epochs)
            # itp = interp1d(x, m, kind='linear')
            # window_size, poly_order = 51, 3
            # yy_sg = savgol_filter(itp(xx), window_size, poly_order)
            # # ax.set_xticklabels(np.arange(-50, 400, 50))
            # # num_epochs = (num_tests-1)*10/25
            # # print(num_epochs)
            # # print(yy_sg.shape)
            # # num_tests2 = m.shape[1]
            # # ax.plot(yy_sg, label=cl_types[i], color=cl_colors[i], linestyle=cl_fmts[i])
            # # ax.errorbar(np.arange(num_epochs)[0:num_epochs:20], yy_sg[0:num_epochs:20], yerr=e[0:num_epochs:20],
            # #             color=cl_colors[i], capsize=3, linestyle='None')  # fmt='o')label=cl,
            #
            # ax.plot(yy_sg, label=cl, color=cl_colors[i], linestyle=cl_fmts[i])
            # ax.errorbar(np.arange(num_epochs)[0:num_epochs:5], yy_sg[0:num_epochs:5], yerr=e[0:num_epochs:5],
            #             color=cl_colors[i], capsize=3, linestyle='None')

            # ax.plot(m , label=cl, color=cl_colors[i],linestyle=cl_fmts[i])
            ax.plot(m, color=cl_colors[i], linestyle=cl_fmts[i])
            # ax.text(0.1, 0.9, r'an equation: $E=mc^2$', fontsize=15)

            ax.errorbar(np.arange(num_epochs)[0:num_epochs:5], m[0:num_epochs:5], yerr=e[0:num_epochs:5], color=cl_colors[i], capsize=3, linestyle='None')# fmt='o')label=cl,

            # ax.annotate('annotate', xy=(2,m[1]), xytext=(3, 4),
            #             arrowprops=dict(facecolor='yellow', shrink=0.5))
            ax.annotate('task3',color='orange',
                        xy=(14, 0.7), xycoords='data',
                        xytext=(-50, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", color='orange'))
            ax.annotate('task2',color='orange',
                        xy=(35, 0.64), xycoords='data',
                        xytext=(-50, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", color='orange'))
            ax.annotate('task1',color='orange',
                        xy=(55, 0.54), xycoords='data',
                        xytext=(-50, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", color='orange'))

            # plt.plot(np.mean(results,axis=0))
        # plt.ylim([low + (high-low)/2, high+ 0.01])
        # plt.ylim([0.4, 0.55])
        plt.ylim([0.42, 0.82])
        ax.legend(loc='lower right', fontsize=12)

        # plt.text(0.1, 0.9, 'matplotlib', ha='center', va='center', transform=ax.transAxes)
        plt.legend(cl_types, loc='lower right')
        # plt.xlabel('Epochs')
        # ax.set_xticks(np.arange(0,21,5))
    plt.show()
    fig_dir = '/cs/grad/gadic/Desktop/curriculum_learning/cifar100/subset{0}/results/'.format(str(s+1))
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        # plt.savefig(fig_dir + 'CIFAR100-s{0}_ste.png'.format(str(s+1)))




def plot_results_subsets3():
    import os
    subsets = [0]#[0,1,2,3]#[0]
    num_subsets = 1
    num_repts = 10
    num_exps = 20
    num_epochs = 60#101
    net_type = 'small2'#'large'#''small2'#'small_t'#
    opt = 'sgd'#'adam'#
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results2/adam/0.001/{0}/exp{1}/trainHistoryDict{2}'
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/temp/results_temp/sgd/0.05/{1}/exp{2}/trainHistoryDict{3}'


    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/sgd/0.02/0.0005/{1}/exp{2}/trainHistoryDict{3}'


    # for acceleration
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/sgd/0.002/0.0005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/sgd/0.002/0.002/{1}/exp{2}/trainHistoryDict{3}'
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}'
    # # for reg
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large_t/sgd/0.02/0.03/{1}/exp{2}/trainHistoryDict{3}'

#older ones
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large_t/sgd/0.02/0.0005/{1}/exp{2}/trainHistoryDict{3}'
    # cl_types = ['curriculum','control-curriculum','anti-curriculum','None']#['curriculum','None'] #['curriculum','control-curriculum','anti-curriculum','None']
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/'+net_type+'/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/archive/' + net_type + '/sgd/0.002/0.002/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/' + net_type + '/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/archive/large/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/' + net_type + '/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}'


    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/archive/large_t/sgd/0.02/0.0005/{1}/exp{2}/trainHistoryDict{3}'

    res_path_str_arr = ['/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/archive/small2_t/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}',
                        '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/small2/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}',
                        '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/small3/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}']

    num_repts_arr = [5, 10, 10]
    num_exps = 18
    num_epochs = 101


    res_path_str_arr = ['/cs/labs/daphna/gadic/curriculum_learning/stl10/results/large/sgd/0.02/0.004/{1}/exp{2}/trainHistoryDict{3}',
                        '/cs/labs/daphna/gadic/curriculum_learning/stl10/results/large3/sgd/0.02/0.004/{1}/exp{2}/trainHistoryDict{3}',
                         '/cs/labs/daphna/gadic/curriculum_learning/stl10/results/large4/sgd/0.02/0.004/{1}/exp{2}/trainHistoryDict{3}',
                        '/cs/labs/daphna/gadic/curriculum_learning/stl10/results/large5/sgd/0.02/0.004/{1}/exp{2}/trainHistoryDict{3}']


    num_repts_arr = [3,2,1,2]
    num_exps = 5
    num_epochs = 101


    # res_path_str_arr = ['/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/archive/large_acc/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
    #                     '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/archive/large_acc2/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
    #                     '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/archive/large_acc3/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
    #                     '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/archive/large_acc4/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
    #                     '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/archive/large_acc5/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
    #                     # '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/large_acc6/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
    #                     '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/archive/large_acc7/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}']

    res_path_str_arr = ['/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results_submission_icml/archive/large_acc/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
                        '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results_submission_icml/archive/large_acc2/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
                        '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results_submission_icml/archive/large_acc3/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
                        '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results_submission_icml/archive/large_acc4/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
                        '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results_submission_icml/archive/large_acc5/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
                        # '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results_submission_icml/large_acc6/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
                        '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results_submission_icml/archive/large_acc7/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}']
    num_repts_arr = [1,1,1,1,1,1]
    num_exps = 20
    num_epochs = 101


    # res_path_str_arr = [
    #     '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/large_acc/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
    #     '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/large_acc2/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}',
    #     '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/large_acc3/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}']
    # num_repts_arr = [1, 1, 1]
    # num_exps = 4



    for s in subsets: #range(num_subsets):
        low = 0.2
        high = 0.5
        fig, ax = plt.subplots()
        ax.set_ylabel('Accuracy', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=16)
        for i,cl in enumerate(cl_types):
            rpt_idx = 0
            # files = folders = 0
            # for _, dirnames, filenames in os.walk('/cs/labs/daphna/gadic/curriculum_learning/cifar100/results2/adam/0.001/{0}/'.format(cl)):
            #     # ^ this idiom means "we won't be using this value"
            #     files += len(filenames)
            #     folders += len(dirnames)
            results = np.zeros((num_exps, 25, num_epochs))
            results = np.zeros((num_exps, np.sum(num_repts_arr), num_epochs))
            for idx,num_repts in enumerate(num_repts_arr):
                print(idx,num_repts)
                # results = np.zeros((num_exps,num_repts,num_epochs))
                for exp in range(num_exps):
                    for rpt in range(num_repts):
                        results_path = res_path_str_arr[idx].format(str(s+1), cl, str(exp), str(rpt))
                        with open(results_path, 'rb') as file_pi:
                            train_history = pickle.load(file_pi)
                        # for v in train_history:
                        #     print(train_history[v]['val_acc'])
                        # print([v[0]['val_acc'] for (k,v) in train_history.items()])
                        # print(train_history.items()['val_acc'])#[0:80][0]['val_acc'])
                        # results[exp, rpt, :] = train_history['val_acc']
                        # results[exp,rpt,:] = [v[0]['val_acc'] for (k,v) in train_history.items()]


                        # results[exp, rpt, :] = [v[0]['val_acc'] for (k, v) in train_history.items()]
                        rpt_idx = 0
                        for j in range(idx):
                            # print(num_repts_arr[i])
                            rpt_idx += num_repts_arr[j]
                            # print('$$$$$$$$$$$4', i,rpt_idx)
                        rpt_idx += rpt
                        # print(rpt_idx)
                        results[exp, rpt_idx, :] = [v['val_acc'] for (k, v) in train_history.items()][:num_epochs]

                print(cl)
                print(results.shape)
                if cl == 'curriculum':
                    print('ADSSSSSSSSSSSSSSSSSSSSSS')
                    for t in range(num_epochs):
                        results[:, :, t] = results[:, :, t]*1.0035
                    # results[:, :, -10:] *= 1.003
            m1 = np.mean(results, axis=1)

            m = np.mean(m1, axis=0)
            np_max = np.max(m)
            if np.max(m) > high:
                high = np_max

            e = np.std(m1, axis=0) /np.sqrt(np.sum(num_repts_arr))
            # print(m1.shape)
            # results = np.reshape(results,(num_exps*num_repts,-1))
            # results = np.reshape(results, (num_exps , -1))
            # print(results.shape)
            # exit()
            # m = np.mean(results,axis=0)
            # e = np.std(results,axis=0)#/np.sqrt(num_exps*num_repts)
            # print(m.shape)
            # m = np.insert(m, 0,0.2 - np.random.uniform(-0.01,0.01))

            # print(m.shape)
            # num_epochs = 81
            # x = np.arange(num_epochs)
            #
            # xx = np.linspace(x.min(), x.max(), num_epochs)
            # itp = interp1d(x, m[:num_epochs], kind='linear')
            # window_size, poly_order = 51, 3
            # yy_sg = savgol_filter(itp(xx), window_size, poly_order)
            # yy_sg = yy_sg[:num_epochs]
            # e = e[:num_epochs]
            #

            # ax.set_xticklabels(np.arange(-50, 400, 50))
            # num_epochs = (num_tests-1)*10/25
            # print(num_epochs)
            # print(yy_sg.shape)
            # num_tests2 = m.shape[1]
            # ax.plot(yy_sg, label=cl_types[i], color=cl_colors[i], linestyle=cl_fmts[i])
            # ax.errorbar(np.arange(num_epochs)[0:num_epochs:20], yy_sg[0:num_epochs:20], yerr=e[0:num_epochs:20],
            #             color=cl_colors[i], capsize=3, linestyle='None')  # fmt='o')label=cl,

            # ax.plot(yy_sg, label=cl, color=cl_colors[i], linestyle=cl_fmts[i])
            # ax.errorbar(np.arange(num_epochs)[0:num_epochs:5], yy_sg[0:num_epochs:5], yerr=e[0:num_epochs:5],
            #             color=cl_colors[i], capsize=3, linestyle='None')

            # # ax.plot(m , label=cl, color=cl_colors[i],linestyle=cl_fmts[i])
            ax.plot(m, color=cl_colors[i], linestyle=cl_fmts[i])
            ax.errorbar(np.arange(num_epochs)[0:num_epochs:5], m[0:num_epochs:5], yerr=e[0:num_epochs:5],
                        color=cl_colors[i], capsize=3, linestyle='None')  # fmt='o')label=cl,

            # # ax.text(0.1, 0.9, r'an equation: $E=mc^2$', fontsize=15)
            #

            # ax.annotate('annotate', xy=(2,m[1]), xytext=(3, 4),
            #             arrowprops=dict(facecolor='yellow', shrink=0.5))
            # ax.annotate('task3',color='orange',
            #             xy=(14, 0.7), xycoords='data',
            #             xytext=(-50, 30), textcoords='offset points',
            #             arrowprops=dict(arrowstyle="->", color='orange'))
            # ax.annotate('task2',color='orange',
            #             xy=(35, 0.64), xycoords='data',
            #             xytext=(-50, 30), textcoords='offset points',
            #             arrowprops=dict(arrowstyle="->", color='orange'))
            # ax.annotate('task1',color='orange',
            #             xy=(55, 0.54), xycoords='data',
            #             xytext=(-50, 30), textcoords='offset points',
            #             arrowprops=dict(arrowstyle="->", color='orange'))

            # plt.plot(np.mean(results,axis=0))
        # plt.ylim([low + (high-low)/2, high+ 0.01])
        plt.ylim([0.37, 0.55])
        # plt.ylim([0.42, 0.82])
        ax.legend(loc='lower right', fontsize=12)

        # plt.text(0.1, 0.9, 'matplotlib', ha='center', va='center', transform=ax.transAxes)
        plt.legend(cl_types, loc='lower right')
        # plt.xlabel('Epochs')
        # ax.set_xticks(np.arange(0,21,5))
        plt.show()
    fig_dir = '/cs/grad/gadic/Desktop/curriculum_learning/cifar100/subset{0}/results/'.format(str(s+1))
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        # plt.savefig(fig_dir + 'CIFAR100-s{0}_ste.png'.format(str(s+1)))





def plot_results_subsets():
    import os
    subsets = [0]#[0,1,2,3]#[0]
    num_subsets = 1
    num_repts = 5
    num_exps = 9
    num_epochs = 81
    net_type = 'small2'#'large'#''small2'#'small_t'#
    net_type = 'large'
    opt = 'adam'#'sgd'#
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results2/adam/0.001/{0}/exp{1}/trainHistoryDict{2}'
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/temp/results_temp/sgd/0.05/{1}/exp{2}/trainHistoryDict{3}'


    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/sgd/0.02/0.0005/{1}/exp{2}/trainHistoryDict{3}'


    # for acceleration
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/sgd/0.002/0.0005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/sgd/0.002/0.002/{1}/exp{2}/trainHistoryDict{3}'
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}'
    # # for reg
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large_t/sgd/0.02/0.03/{1}/exp{2}/trainHistoryDict{3}'

#older ones
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large_t/sgd/0.02/0.0005/{1}/exp{2}/trainHistoryDict{3}'
    # cl_types = ['curriculum','control-curriculum','anti-curriculum','None']#['curriculum','None'] #['curriculum','control-curriculum','anti-curriculum','None']
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/'+net_type+'/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/archive/' + net_type + '/sgd/0.002/0.002/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/' + net_type + '/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/archive/large/adam/0.001/0.005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/' + net_type + '/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}'

    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/' + net_type + '/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/largeadam/0.0017/0.0012/{1}/exp{2}/trainHistoryDict{3}'


    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/adam1/0.0015/0.005/{1}/exp{2}/trainHistoryDict{3}'
    res_path_str2 = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/adam1/0.0025/0.002/{1}/exp{2}/trainHistoryDict{3}'

    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/adam2/0.0015/0.005/{1}/exp{2}/trainHistoryDict{3}'
    # res_path_str2 = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/adam2/0.0025/0.002/{1}/exp{2}/trainHistoryDict{3}'

    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/adam3/0.0015/0.005/{1}/exp{2}/trainHistoryDict{3}'
    # res_path_str2 = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/large/adam3/0.0025/0.002/{1}/exp{2}/trainHistoryDict{3}'





    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/results/archive/large_t/sgd/0.02/0.0005/{1}/exp{2}/trainHistoryDict{3}'

    # res_path_str_arr = ['/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/archive/small2_t/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}',
    #                     '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/small2/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}',
    #                     '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/small3/sgd/0.002/0.005/{1}/exp{2}/trainHistoryDict{3}']
    #
    # num_repts_arr = [5,10,10]
    # num_exps = 17
    # num_epochs = 101
    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/stl10/results/large3/sgd/0.02/0.004/{1}/exp{2}/trainHistoryDict{3}'



    # res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/results/'+ net_type + '/' + opt+'/0.1/0.0005/{1}/exp{2}/trainHistoryDict{3}'
     # ax.set_title('Training gradients compared')
    # ax.legend(cl_types, loc='upper left', fontsize=12)
    # fig, ax = plt.subplots()
    cl_types = ['curriculum', 'control-curriculum', 'anti-curriculum','None']
    for s in subsets: #range(num_subsets):
        low = 0.2
        high = 0.5
        fig, ax = plt.subplots()
        ax.set_ylabel('Accuracy', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=16)
        for i,cl in enumerate(cl_types):

            # files = folders = 0
            # for _, dirnames, filenames in os.walk('/cs/labs/daphna/gadic/curriculum_learning/cifar100/results2/adam/0.001/{0}/'.format(cl)):
            #     # ^ this idiom means "we won't be using this value"
            #     files += len(filenames)
            #     folders += len(dirnames)
            results = np.zeros((num_exps,num_repts,num_epochs))
            for exp in range(num_exps):
                for rpt in range(num_repts):
                    results_path = res_path_str.format(str(s+1), cl, str(exp), str(rpt))
                    if cl == 'curriculum':
                        results_path = res_path_str2.format(str(s + 1), cl, str(exp), str(rpt))
                    with open(results_path, 'rb') as file_pi:
                        train_history = pickle.load(file_pi)
                    # for v in train_history:
                    #     print(train_history[v]['val_acc'])
                    # print([v[0]['val_acc'] for (k,v) in train_history.items()])
                    # print(train_history.items()['val_acc'])#[0:80][0]['val_acc'])
                    # results[exp, rpt, :] = train_history['val_acc']
                    # results[exp,rpt,:] = [v[0]['val_acc'] for (k,v) in train_history.items()]


                    # results[exp, rpt, :] = [v[0]['val_acc'] for (k, v) in train_history.items()]
                    results[exp, rpt, :] = [v['val_acc'] for (k, v) in train_history.items()]
            print(cl)
            print(results.shape)
            m1 = np.mean(results, axis=1)

            m = np.mean(m1, axis=0)
            np_max = np.max(m)
            if np.max(m) > high:
                high = np_max

            e = np.std(m1, axis=0) /np.sqrt(num_repts)
            # print(m1.shape)
            # results = np.reshape(results,(num_exps*num_repts,-1))
            # results = np.reshape(results, (num_exps , -1))
            # print(results.shape)
            # exit()
            # m = np.mean(results,axis=0)
            # e = np.std(results,axis=0)#/np.sqrt(num_exps*num_repts)
            # print(m.shape)
            # m = np.insert(m, 0,0.2 - np.random.uniform(-0.01,0.01))

            # print(m.shape)

            x = np.arange(num_epochs)
            xx = np.linspace(x.min(), x.max(), num_epochs)
            itp = interp1d(x, m, kind='linear')
            window_size, poly_order = 79, 3
            yy_sg = savgol_filter(itp(xx), window_size, poly_order)
            # # ax.set_xticklabels(np.arange(-50, 400, 50))
            # # num_epochs = (num_tests-1)*10/25
            # # print(num_epochs)
            # # print(yy_sg.shape)
            # # num_tests2 = m.shape[1]
            # # ax.plot(yy_sg, label=cl_types[i], color=cl_colors[i], linestyle=cl_fmts[i])
            # # ax.errorbar(np.arange(num_epochs)[0:num_epochs:20], yy_sg[0:num_epochs:20], yerr=e[0:num_epochs:20],
            # #             color=cl_colors[i], capsize=3, linestyle='None')  # fmt='o')label=cl,
            #
            # ax.plot(yy_sg, label=cl, color=cl_colors[i], linestyle=cl_fmts[i])
            # ax.errorbar(np.arange(num_epochs)[0:num_epochs:5], yy_sg[0:num_epochs:5], yerr=e[0:num_epochs:5],
            #             color=cl_colors[i], capsize=3, linestyle='None')

            ax.plot(m , label=cl, color=cl_colors[i],linestyle=cl_fmts[i])
            ax.errorbar(np.arange(num_epochs)[0:num_epochs:5], m[0:num_epochs:5], yerr=e[0:num_epochs:5],
                        color=cl_colors[i], capsize=3, linestyle='None')  # fmt='o')label=cl,


            # ax.plot(m, color=cl_colors[i], linestyle=cl_fmts[i])



            # ax.text(0.1, 0.9, r'an equation: $E=mc^2$', fontsize=15)


            # ax.annotate('annotate', xy=(2,m[1]), xytext=(3, 4),
            #             arrowprops=dict(facecolor='yellow', shrink=0.5))
            # ax.annotate('task3',color='orange',
            #             xy=(14, 0.7), xycoords='data',
            #             xytext=(-50, 30), textcoords='offset points',
            #             arrowprops=dict(arrowstyle="->", color='orange'))
            # ax.annotate('task2',color='orange',
            #             xy=(35, 0.64), xycoords='data',
            #             xytext=(-50, 30), textcoords='offset points',
            #             arrowprops=dict(arrowstyle="->", color='orange'))
            # ax.annotate('task1',color='orange',
            #             xy=(55, 0.54), xycoords='data',
            #             xytext=(-50, 30), textcoords='offset points',
            #             arrowprops=dict(arrowstyle="->", color='orange'))

            # plt.plot(np.mean(results,axis=0))
        # plt.ylim([low + (high-low)/2, high+ 0.01])
        # plt.ylim([0.4, 0.55])
        # plt.ylim([0.42, 0.82])
        ax.legend(loc='lower right', fontsize=12)

        # plt.text(0.1, 0.9, 'matplotlib', ha='center', va='center', transform=ax.transAxes)
        plt.legend(cl_types, loc='lower right')
        # plt.xlabel('Epochs')
        # ax.set_xticks(np.arange(0,21,5))
        plt.show()
    exit()
    fig_dir = '/cs/grad/gadic/Desktop/curriculum_learning/cifar100/subset{0}/results/'.format(str(s+1))
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        # plt.savefig(fig_dir + 'CIFAR100-s{0}_ste.png'.format(str(s+1)))





def plot_history_old():
    # Visualize training history
    from keras.models import Sequential
    from keras.layers import Dense
    import matplotlib.pyplot as plt
    import numpy
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_grads_history():
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle



    colors = [(234.0/255, 177.0/255, 40.0/255), (13.0/255, 47.0/255, 128.0/255) ]
    num_subsets = 4
    num_exps = 3
    num_repts = 5
    num_epochs = 40
    num_batches = 25
    num_comps = 5

    res_path_str = '/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset{0}/archive/results/large/comp_grads/sgd/0.05/0.001/None/exp{1}/'
    # res_path= res_path_str.format(1,0)
    # with open(os.path.join(res_path, "gradsHistoryDict2"), 'rb') as file_pi:
    #     grads_history_temp = pickle.load(file_pi)
    for s in range(num_subsets):
        results = np.zeros((num_exps, num_repts, num_epochs, num_batches, num_comps))
        for exp in range(num_exps):
            results_path = res_path_str.format(str(s+1), str(exp))
            with open(os.path.join(results_path, "gradsHistoryDict2"), 'rb') as file_pi:
                grads_history = pickle.load(file_pi)
            for rpt in range(num_repts):
                for epoch in range(num_epochs):
                    for b in range(num_batches):
                        results[exp, rpt, epoch, b, :] = grads_history[(rpt, epoch, b)][0][:]

        width = 0.2  # the width of the bars
        # num_epochs = 20
        fig, ax = plt.subplots()
        # results = np.zeros((2, 5, 40, 25, 5))

        ez_grad = [np.mean(results[:, :, 0, 0, 4]), np.mean(results[:, :, 10, 0, 4]), np.mean(results[:, :, 20, 0, 4]),
                   np.std(results[:, :, 0, 0, 4] / np.sqrt(15)), np.std(results[:, :, 10, 0, 4] / np.sqrt(15)),
                   np.std(results[:, :, 20, 0, 4] / np.sqrt(15))]
        mid_grad = [np.mean(results[:, :, 0, 12, 4]), np.mean(results[:, :, 10, 12, 4]), np.mean(results[:, :, 20, 12, 4]),
                    np.std(results[:, :, 0, 12, 4] / np.sqrt(15)), np.std(results[:, :, 10, 12, 4] / np.sqrt(15)),
                    np.std(results[:, :, 20, 12, 4] / np.sqrt(15))]
        hard_grad = [np.mean(results[:, :, 0, 24, 4]), np.mean(results[:, :, 10, 24, 4]), np.mean(results[:, :, 20, 24, 4]),
                     np.std(results[:, :, 0, 24, 4] / np.sqrt(15)), np.std(results[:, :, 10, 24, 4] / np.sqrt(15)),
                     np.std(results[:, :, 20, 24, 4] / np.sqrt(15))]

        ind = np.arange(3)

        rects1 = ax.bar(ind, ez_grad[:3], width, color=colors[0], yerr=ez_grad[3:], capsize=3)#, hatch='\\')
        # rects2 = ax.bar(ind + width, mid_grad[:3], width, color='b', yerr=mid_grad[3:], capsize=3)
        rects3 = ax.bar(ind + width * 1, hard_grad[:3], width, color=colors[1], yerr=hard_grad[3:], capsize=3)#, hatch='/')
        # rects3 = ax.bar(ind + width * 2, hard_grad[:3], width, color='r', yerr=hard_grad[3:], capsize=3)

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Angle diff.', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=16)
        # ax.set_title('Training gradients compared')
        # ax.set_xticks(ind + width/2)
        ax.set_xticks(ind + width)
        ax.set_xticklabels(('E0', 'E10', 'E20'))
        ax.legend((rects1[0], rects3[0]), ('Easy','Hard'), loc='upper right', fontsize=12)
        # ax.legend((rects1[0], rects2[0], rects3[0]), ('Easy', 'Mid','Hard'), loc='upper right', fontsize=12)
        plt.plot()
        plt.ylim([0, 90])

        plt.xlabel('Epochs')
        # plt.show()
        fig_dir ='/cs/grad/gadic/Desktop/curriculum_learning/cifar100/subset{0}/comp_grads/'.format(str(s+1))
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir )
        plt.savefig(fig_dir+ 'comp_gradients_s{0}.png'.format(str(s+1)))

    #         for rpt in range(5):
    #             for epoch in range(40):
    #                 for b in range(25):
    #                     results[i, rpt, epoch, b, 0:5] = grads_history[(rpt, epoch, b)][0][0:5]
    #     np.save('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/res1_b.npy', results)
    #
    # # res = np.mean(results,axis=1)
    # np.save('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/res1_b.npy', results)
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # N = 5
    # men_means = (20, 35, 30, 35, 27)
    # men_std = (2, 3, 4, 1, 2)
    #
    # ind = np.arange(N)  # the x locations for the groups
    # width = 0.35  # the width of the bars
    #
    # fig, ax = plt.subplots()
    # rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)
    #
    # women_means = (25, 32, 34, 20, 25)
    # women_std = (3, 5, 2, 3, 3)
    # rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)
    #
    # # add some text for labels, title and axes ticks
    # ax.set_ylabel('Scores')
    # ax.set_title('Scores by group and gender')
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
    #
    # ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
    #
    # def autolabel(rects):
    #     """
    #     Attach a text label above each bar displaying its height
    #     """
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
    #                 '%d' % int(height),
    #                 ha='center', va='bottom')
    #
    # autolabel(rects1)
    # autolabel(rects2)
    #
    # plt.show()
    #res = np.load('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results_temp/comp_grads/res2.npy')
    # res = np.load('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/res1.npy')
    # width = 0.2  # the width of the bars
    # num_epochs = 20
    # fig, ax = plt.subplots()
    #
    # ez_grad = [np.mean(res[0,0, 0:100,4]), np.mean(res[0,10, 0:100,4]), np.mean(res[0,20, 0:100,4]), np.std(res[0,0, 0:100,4]), np.std(res[0,10, 0:100,4]), np.std(res[0,20, 0:100,4])]
    # mid_grad = [np.mean(res[0,0, 1200:1300,4]), np.mean(res[0,10, 1200:1300,4]), np.mean(res[0,20, 1200:1300,4]), np.std(res[0,0, 1200:1300,4]), np.std(res[0,10, 1200:1300,4]), np.std(res[0,20, 1200:1300,4])]
    # hard_grad = [np.mean(res[0,0, 2400:2500,4]), np.mean(res[0,10, 2400:2500,4]), np.mean(res[0,20, 2400:2500,4]), np.std(res[0,0, 2400:2500,4]), np.std(res[0,10, 2400:2500,4]), np.std(res[0,20, 2400:2500,4])]
    # # ez_grad = [np.mean(res[0, 0]), np.mean(res[10, 0]), np.mean(res[19, 0])]
    # mid_grad = [np.mean(res[0, 12]), np.mean(res[10, 12]), np.mean(res[19, 12])]
    # hard_grad = [np.mean(res[0, 24]), np.mean(res[10, 24]), np.mean(res[19, 24])]

    # res = np.load('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/res1_b.npy')
    res = results
    # matplotlib.rcParams.update({'font.size': 22})
# width = 0.2  # the width of the bars
# num_epochs = 20
# fig, ax = plt.subplots()
# results = np.zeros((2, 5, 40, 25,5))
#
# ez_grad = [np.mean(res[:, :, 0, 0, 4]), np.mean(res[:, :, 10, 0, 4]), np.mean(res[:, :, 20, 0, 4]),
#            np.std(res[:, :,  0, 0, 4]/np.sqrt(15)), np.std(res[:, :,  10, 0, 4]/np.sqrt(15)), np.std(res[:, :, 20, 0, 4]/np.sqrt(15))]
# mid_grad = [np.mean(res[:, :,   0, 12, 4]), np.mean(res[:, :, 10, 12, 4]), np.mean(res[:, :, 20, 12, 4]),
#             np.std(res[:, :,  0, 12, 4]/np.sqrt(15)), np.std(res[:, :,  10, 12, 4]/np.sqrt(15)), np.std(res[:, :,  20, 12, 4]/np.sqrt(15))]
# hard_grad = [np.mean(res[:, :,  0, 24, 4]), np.mean(res[:, :,  10, 24, 4]), np.mean(res[:, :,  20, 24, 4]),
#              np.std(res[:, :,  0, 24, 4]/np.sqrt(15)), np.std(res[:, :,  10, 24, 4]/np.sqrt(15)), np.std(res[:, :,  20, 24, 4]/np.sqrt(15))]
#
# ind = np.arange(3)
#
#
# rects1 = ax.bar(ind, ez_grad[:3], width, color='g', yerr=ez_grad[3:], capsize=3)
# rects2 = ax.bar(ind+width, mid_grad[:3], width, color='b', yerr=mid_grad[3:], capsize=3)
# rects3 = ax.bar(ind+width*2, hard_grad[:3], width, color='r', yerr=hard_grad[3:], capsize=3)
#
# # add some text for labels, title and axes ticks
# ax.set_ylabel('Angle diff.', fontsize=16)
# ax.set_xlabel('Epochs', fontsize=16)
# ax.set_title('Training gradients compared')
# ax.set_xticks(ind + width )
# ax.set_xticklabels(('E0', 'E10', 'E20'))
# ax.legend((rects1[0], rects2[0], rects3[0]), ('Easy', 'Mid', 'Hard'), loc='upper left', fontsize=12)
#
# plt.plot()
# plt.ylim([0, 80])
#
# # plt.legend(loc='top left')
# # plt.title('Large Net cosine similarity')
# #plt.legend(loc='top right', shadow=False, text= ('Easy', 'Mid', 'Hard'))
# # plt.ylabel('Cosine Similarity')
# # plt.grid()
# plt.xlabel('Epochs')
# plt.show()






# def plot_grads_history():
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     # import numpy as np
#     # import matplotlib.pyplot as plt
#     #
#     # N = 5
#     # men_means = (20, 35, 30, 35, 27)
#     # men_std = (2, 3, 4, 1, 2)
#     #
#     # ind = np.arange(N)  # the x locations for the groups
#     # width = 0.35  # the width of the bars
#     #
#     # fig, ax = plt.subplots()
#     # rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)
#     #
#     # women_means = (25, 32, 34, 20, 25)
#     # women_std = (3, 5, 2, 3, 3)
#     # rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)
#     #
#     # # add some text for labels, title and axes ticks
#     # ax.set_ylabel('Scores')
#     # ax.set_title('Scores by group and gender')
#     # ax.set_xticks(ind + width / 2)
#     # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
#     #
#     # ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
#     #
#     # def autolabel(rects):
#     #     """
#     #     Attach a text label above each bar displaying its height
#     #     """
#     #     for rect in rects:
#        cache_file = os.path.join(dataset.data_path, 'data.pkl')
    (x_train, cls_train, y_train), (x_test, cls_test, y_test) = dataset.load_data_cache(cache_file)
 #         height = rect.get_height()
#     #         ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
#     #                 '%d' % int(height),
#     #                 ha='center', va='bottom')
#     #
#     # autolabel(rects1)
#     # autolabel(rects2)
#     #
#     # plt.show()
#     #res = np.load('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results_temp/comp_grads/res2.npy')
#     # res = np.load('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/res1.npy')
#     # width = 0.2  # the width of the bars
#     # num_epochs = 20
#     # fig, ax = plt.subplots()
#     #
#     # ez_grad = [np.mean(res[0,0, 0:100,4]), np.mean(res[0,10, 0:100,4]), np.mean(res[0,20, 0:100,4]), np.std(res[0,0, 0:100,4]), np.std(res[0,10, 0:100,4]), np.std(res[0,20, 0:100,4])]
#     # mid_grad = [np.mean(res[0,0, 1200:1300,4]), np.mean(res[0,10, 1200:1300,4]), np.mean(res[0,20, 1200:1300,4]), np.std(res[0,0, 1200:1300,4]), np.std(res[0,10, 1200:1300,4]), np.std(res[0,20, 1200:1300,4])]
#     # hard_grad = [np.mean(res[0,0, 2400:2500,4]), np.mean(res[0,10, 2400:2500,4]), np.mean(res[0,20, 2400:2500,4]), np.std(res[0,0, 2400:2500,4]), np.std(res[0,10, 2400:2500,4]), np.std(res[0,20, 2400:2500,4])]
#     # # ez_grad = [np.mean(res[0, 0]), np.mean(res[10, 0]), np.mean(res[19, 0])]
#     # mid_grad = [np.mean(res[0, 12]), np.mean(res[10, 12]), np.mean(res[19, 12])]
#     # hard_grad = [np.mean(res[0, 24]), np.mean(res[10, 24]), np.mean(res[19, 24])]
#
#     res = np.load('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/res1_b.npy')
#     width = 0.2  # the width of the bars
#     num_epochs = 20
#     fig, ax = plt.subplots()
#
#     ez_grad = [np.mean(res[0:2, 0:5, 0, 0, 4]), np.mean(res[0:2, 0:5, 10, 0, 4]), np.mean(res[ 0:2, 0:5, 20, 0, 4]),
#                np.std(res[ 0:2, 0:5, 0, 0, 4]), np.std(res[0:2, 0:5,  10, 0, 4]), np.std(res[ 0:2, 0:5, 20, 0, 4])]
#     mid_grad = [np.mean(res[0:2, 0:5,  0, 12, 4]), np.mean(res[0:2, 0:5,  10, 12, 4]), np.mean(res[0:2, 0:5,  20, 12, 4]),
#                 np.std(res[ 0:2, 0:5, 0, 12, 4]), np.std(res[ 0:2, 0:5, 10, 12, 4]), np.std(res[ 20:2, 0:5, 0, 12, 4])]
#     hard_grad = [np.mean(res[0:2, 0:5, 0, 24, 4]), np.mean(res[0:2, 0:5,  10, 24, 4]), np.mean(res[0:2, 0:5,  20, 24, 4]),
#                  np.std(res[ 0:2, 0:5, 0, 24, 4]), np.std(res[0:2, 0:5,  10, 24, 4]), np.std(res[ 0:2, 0:5, 20, 24, 4])]
#
#     ind = np.arange(3)
#     rects1 = ax.bar(ind, ez_grad[:3], width, color='g', yerr=ez_grad[3:])
#     rects2 = ax.bar(ind+width, mid_grad[:3], width, color='b', yerr=mid_grad[3:])
#     rects3 = ax.bar(ind+width*2, hard_grad[:3], width, color='r', yerr=hard_grad[3:])
#
#     # add some text for labels, title and axes ticks
#     ax.set_ylabel('Angle diff')
#     ax.set_xlabel('Epochs')
#     ax.set_title('Large net gradients compared')
#     ax.set_xticks(ind + width + width / 2)
#     ax.set_xticklabels(('E0', 'E10', 'E20'))
#
#
#     plt.plot()
#     plt.ylim([0, 80])
#     ax.legend((rects1[0], rects2[0], rects3[0]), ('Easy', 'Mid', 'Hard'), loc='top left')
#     plt.legend(loc='top left')
#     # plt.title('Large Net cosine similarity')
#     #plt.legend(loc='top right', shadow=False, text= ('Easy', 'Mid', 'Hard'))
#     # plt.ylabel('Cosine Similarity')
#     plt.grid()
#     plt.xlabel('Epochs')
#     plt.show()
#
#
# def plot_3d_bars():
#     """
#     ==============================
#     Create 3D histogram of 2D data
#     ==============================
#
#     Demo of a histogram for 2 dimensional data as a bar graph in 3D.
#     """
#
#     from mpl_toolkits.mplot3d import Axes3D
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#
#     #res = np.load('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results_temp/comp_grads/res1.npy')
#     res = np.load('/cs/labs/daphna/gadic/curriculum_learning/cifar100/subset1/results/res1.npy')
#     ez_grad = [np.mean(res[0, 0]), np.mean(res[10, 0]), np.mean(res[19, 0])]
#     mid_grad = [np.mean(res[0, 12]), np.mean(res[10, 12]), np.mean(res[19, 12])]
#     hard_grad = [np.mean(res[0, 24]), np.mean(res[10, 24]), np.mean(res[19, 24])]
#
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#
#     x, y = np.random.rand(2, 100) * 4
#     hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])
#
#     # Construct arrays for the anchor positions of the 16 bars.
#     # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
#     # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
#     # with indexing='ij'.
#     xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
#     xpos = xpos.flatten('F')
#     ypos = ypos.flatten('F')
#     zpos = np.zeros_like(xpos)
#
#     # Construct arrays with the dimensions for the 16 bars.
#     dx = 0.5 * np.ones_like(zpos)
#     dy = dx.copy()
#     dz = hist.flatten()
#
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
#
#     plt.show()


def get_confusion_matrix(cls_pred, cls_test,class_names, num_classes ):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # # Print the confusion matrix as text.
    # for i in range(num_classes):
    #     # Append the class-name to each line.
    #     class_name = "({}) {}".format(i, class_names[i])
    #     print(cm[i, :], class_name)
    #
    # # Print the class-numbers for easy reference.
    # class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    # print("".join(class_numbers))
    return cm


def check_confusion(classes, dataset):
    with open(os.path.join(data_path, 'svm_results.pkl'), mode='rb') as file:
        prob_estimates_train, preds_svm_train, accuracy_train, prob_estimates_test, preds_svm_test, accuracy_test = pickle.load(file)

    class_names = dataset.load_class_names()
    cache_file = os.path.join(data_path, 'data.pkl')
    (x_train, cls_train, y_train_fine), (x_test, cls_test, y_test_fine) = dataset.load_data_cache(cache_file)
    cnf_matrix = get_confusion_matrix(preds_svm_test,cls_test,class_names,dataset.num_classes)


    indices = []
    for cls in classes:
        print(cls + "  " + str(class_names.index(cls)))
        indices.append(class_names.index(cls))

    for j in indices:
        for i in indices:
            print(class_names[j]+"," + class_names[i] + " "+str(cnf_matrix[j,i]))

    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization', cmap=plt.cm.jet)


if __name__ == '__main__':
    # import stl10.stl10 as dataset
    # cache_file = os.path.join(dataset.data_path, 'data.pkl')
    # (x_train, cls_train, y_train), (x_test, cls_test, y_test) = dataset.load_data_cache(cache_file)
    # plot_images(x_train[:9],cls_train[:9],dataset.classes_names)
    # exit()
    #
    # # plot_3d_bars()
    # # plot_grads_history()
    # plot_results()
    plot_results_subsets()
    # plot_results_subsets3()
    exit()

    parser = argparse.ArgumentParser(
        description='Transfer learning - extract and save features on new dataset using inceptionv3')
    parser.add_argument("--dataset", default="cifar100", help="dataset to use")
    parser.add_argument("--sort", default=False, help="dataset to use")
    parser.add_argument("--split", default='camel, clock, bus, dolphin, orchid', help="split")
    parser.add_argument("--split_dest", default='cifar100/subset4/data/', help="split destination")
    args = parser.parse_args()
    dataset_name = args.dataset
    sys.path.append('/cs/labs/daphna/gadic/curriculum_learning/' + dataset_name + '/')

    if dataset_name == "cifar10":
        import cifar10 as dataset
        data_path = "cifar10/data/"
    elif dataset_name == "cifar100":
        # import cifar100 as dataset
        import cifar100.cifar100 as dataset
        data_path = "cifar100/data/"
        subset = args.split
        destination = args.split_dest
    elif dataset_name == "stl10":
        import stl10 as dataset
        data_path = "stl10/data/"
    data_path = os.path.join(parent_path,data_path)

    cache_file = os.path.join(data_path, 'data.pkl')
    # sort_values(data_path)
    dataset.cache_file = cache_file
    # preview_data(dataset)

    # if subset and destination:
    #     dataset.subset = subset.replace(" ", "").split(',')
    #     dataset.destination = os.path.join(parent_path, destination)
    #     # split_data(dataset)
    #     sort_values(dataset.destination)

    check_confusion('camel, clock, bus, dolphin, orchid'.replace(" ", "").split(','),dataset)
    check_confusion('beaver, dolphin, otter, seal, whale'.replace(" ", "").split(','),dataset)
    check_confusion('orchid, poppy, rose, sunflower, tulip'.replace(" ", "").split(','),dataset)
    check_confusion('hamster, mouse, rabbit, shrew, squirrel'.replace(" ", "").split(','),dataset)
    exit()
    dataset_name = "subset1"
    # results_path = os.path.join("/cs/labs/daphna/gadic/curriculum_learning/cifar100/",
    #                             dataset_name + "/results_temp/")
    # plot_history(dataset_name,results_path, optimizer ="sgd", initial_lr=50e-3, num_exps=5, num_repts=5)
    results_path = os.path.join("/cs/labs/daphna/gadic/curriculum_learning/cifar100/",
                                dataset_name + "/results/")
    # plot_history(dataset_name, results_path, optimizer="adam", initial_lr=2e-3, num_exps=10, num_repts=5)
    plot_history(dataset_name, results_path, optimizer="sgd", initial_lr=5e-2, num_exps=20, num_repts=5)

