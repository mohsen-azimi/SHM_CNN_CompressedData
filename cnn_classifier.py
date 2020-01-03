import tensorflow as tf
import keras
from keras import Sequential
from keras.models import load_model

from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import  Dense, Flatten, Dropout, Conv1D, MaxPool1D, AvgPool1D,Conv2D, MaxPool2D, Activation
from keras.utils.vis_utils import plot_model
from utils import convert_matlab_file, load_dataset, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle
import hdf5storage
from keras.utils import np_utils

import numpy
import math
import scipy.io
import time
################################## for plot confusion imports
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))



def create_CNN_model():
    model = Sequential()

    # ................................................................................................................
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, input_shape=(lenSignal, nSensors)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  # Batch Normalization
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer

    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))
    # ................................................................................................................
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, input_shape=(lenSignal, nSensors)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  # Batch Normalization
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer

    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))
    # ................................................................................................................
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, input_shape=(lenSignal, nSensors)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  # Batch Normalization
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer

    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))
    # ................................................................................................................
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(nClasses*100))

    model.add(Dense(nClasses, activation='softmax'))

    ###################################################################################################################

    opt = keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])


    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    model.summary()


#   ...................................................................................................................
    return model


# #####################################################################################################################
loadMAT = 'A_SHM1'   # 1,2,   4,5,6,7
saveMAT = loadMAT

# saveModel = loadMAT + '_TL'



# #####################################################################################################################
def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# #####################################################################################################################

matlab_file = 'Data/'+loadMAT+'.mat'
mat = hdf5storage.loadmat(matlab_file)

InputData = np.transpose(mat['InputData'], (2, 0, 1))
TargetData = mat['TargetData']
nSamples = int(np.asscalar(mat['nSamples']))
lenSignal = int(np.asscalar(mat['lenSignal']))
nSensors = int(np.asscalar(mat['nSensors']))
nClasses = int(np.asscalar(mat['nClasses']))

X = InputData
Y = np_utils.to_categorical(TargetData)

X, Y = shuffle(X, Y)


# -----------------------------------------------------------------------------------------------------------------
# #####################################################################################################################
# -----------------------------------------------------------------------------------------------------------------




# ############ CNN ###############
# ################################
# ###############################

with tf.device("/gpu:1"):
    print("Training is started...")
    t0 = time.time()  # t0

    skfold = StratifiedKFold(numpy.argmax(Y, axis=-1), n_folds=10, shuffle=True, random_state=None)
    print("skfold:",skfold)

    for train_index, test_index  in skfold:

        model = create_CNN_model()

        history = History()
        # nepochs = 100
        nepochs = 100
        batchsize = 128

        model.fit(X[train_index], Y[train_index], epochs=nepochs, batch_size=batchsize, callbacks=[history], verbose=1)
        scores = model.evaluate(X[test_index], Y[test_index], verbose=1)


        print(scores)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        Y_testpred = model.predict_classes(X[test_index])
        Y_testpredScores = model.predict(X[test_index])




        # Compute confusion matrix
        Y_testtrue = np.argmax(Y[test_index], axis=1)
        cnf_matrix = confusion_matrix(Y_testtrue, Y_testpred)

        X_test2MAT = np.transpose(X[test_index], (1, 2, 0))

        scipy.io.savemat('saveMATs\A_SHM4_freq.mat', {
        scipy.io.savemat('saveMATs\A_SHM1_TH.mat', {
            'Y_true': Y_testtrue,
            'Y_pred': Y_testpred,
            'Y_predScores': Y_testpredScores,
            'AccuracyTH': history.acc,
            'LossTH': history.loss,
            'nClasses': nClasses,
            'nepochs': nepochs,
            'batchsize':  batchsize
        })
        #
        np.set_printoptions(precision=2)
        
        # Plot non-normalized confusion matrix
        plt.figure()
        class_names = np.array(["Intact", "Pattern1", "Pattern2"],dtype='<U10')
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')
        
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')
        
        plt.show()



        plt.figure()
        plt.plot(range(0, nepochs), history.acc, 'b--') # plt.plot(range(0, nepochs*math.ceil(0.9*nSamples/batchsize)), history.acc, 'b--')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
        plt.pause(0.05)


        # plt.figure()
        # plt.plot(range(0, nepochs), history.loss, 'r--')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.show()
        # plt.pause(0.05)

        t1 = time.time()  # t1 at the end
        print("Total Run Time: ", int(t1-t0), " seconds")

    # ####################################################################################


