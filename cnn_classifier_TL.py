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
        self.val_acc = []
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))



def create_CNN_model():
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, input_shape=(lenSignal, nSensors)))
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  # Batch Normalization
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer
    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))

    # ................................................................................................................
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  # Batch Normalization
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer
    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))

    # ................................................................................................................
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, input_shape=(lenSignal, nSensors)))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, input_shape=(lenSignal, nSensors)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  # Batch Normalization
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer
    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))

    # ................................................................................................................
    model.add(Flatten())
    model.add(BatchNormalization())
    #
    model.add(Dense(nClasses*1))
    model.add(Dense(nClasses*1))
    model.add(Dense(nClasses, activation='softmax'))

    # -----------------------------------------------------------------------------------------------------------------
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # model.summary()

#   ...................................................................................................................
    return model

# ...................................................................................................................
# #####################################################################################################################
loadMAT = 'A_SHM3_freq2'      # 1,2,   4,5,6,7

# #####################################################################################################################
def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# #####################################################################################################################
matlab_file = 'BenchMark/Outputs/'+loadMAT+'.mat'
mat = hdf5storage.loadmat(matlab_file)

InputData = np.transpose(mat['InputData2'], (2, 0, 1))
TargetData = mat['TargetData']    # Source Domain
# TargetData = mat['TargetData2']   # Target Domain
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
with tf.device("/gpu:1"):

    nFolds = 10
    print("Training is started")
    t0 = time.time()  # t0
    skfold = StratifiedKFold(numpy.argmax(Y, axis=-1), n_folds=nFolds, shuffle=True, random_state=None)
    # print("skfold:",skfold)

    fold = 1
    for train_index, test_index in skfold:

        # # I: Create a CNN
        # model = create_CNN_model()
        # for layer in model.layers:
        #     layer.trainable = True
        #
        # learnignRate = 0.00005

        # II:Load a pre-trained model
        model = load_model('saveModels/TL_' + loadMAT.replace("freq2", "freq1") + '_fold' + str(fold) + '.h5')
        model.summary()

        # remove last layers and add new layers
        model.pop() # remove last layer: model.add(Dense(nClasses, activation='softmax'))
        model.pop() # remove  layer:
        model.pop() # remove last
        model.pop() # remove  layer:


        # now we reached Flattern Layer

        for i, layer in enumerate(model.layers):
            print(i, layer.name)

        # # Add new layers
        model.add(Dense(nClasses*100, name='TL1_Dense'))
        model.add(BatchNormalization (name='TL2_BN'))
        model.add(LeakyReLU(alpha=.01,name='TL3_LeakyReLU'))

        model.add(Dense(nClasses*10, name='TL4_Dense'))
        model.add(Dropout(0.2,name='TL5_Dropout'))
        model.add(LeakyReLU(alpha=.01,name='TL6_LeakyReLU'))

        model.add(Dense(nClasses, activation='softmax',name='TL7_Dense'))

        for i, layer in enumerate(model.layers):
            print(i, layer.name)

        # Freeze the layers except the last 4 layers

        # for layer in model.layers:
        #     layer.trainable = True

        for layer in model.layers[:-7]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in model.layers:
            print(layer, layer.trainable)

        learnignRate = 0.00005
        #   End of Load a CNN

        # Do not forget to compile
        opt = keras.optimizers.Adam(lr=learnignRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse', 'mae', 'categorical_crossentropy'])
        model.summary()

        history = History()
        nepochs = 100 # 500
        batchsize = 256 # 1024, 256

        model.fit(X[train_index], Y[train_index], epochs=nepochs, batch_size=batchsize, callbacks=[history],
                  verbose=1)

        scores = model.evaluate(X[test_index], Y[test_index], verbose=1)

        print("*****  Fold {} *****".format(fold))

        print(scores)
        print("%s: %.2f" % (model.metrics_names[0], scores[0] ))
        print("%s: %.2f" % (model.metrics_names[1], scores[1] ))
        print("%s: %.2f" % (model.metrics_names[2], scores[2] ))
        print("%s: %.2f" % (model.metrics_names[3], scores[3] ))
        print("%s: %.2f" % (model.metrics_names[4], scores[4] ))

        Y_testpred = model.predict_classes(X[test_index])
        Y_testpredScores = model.predict(X[test_index])

        # Compute confusion matrix
        Y_testtrue = np.argmax(Y[test_index], axis=1)
        cnf_matrix = confusion_matrix(Y_testtrue, Y_testpred)

        X_test2MAT = np.transpose(X[test_index], (1, 2, 0))

        model.save('saveModels/TL_' + loadMAT + '_fold' + str(fold) + '.h5')

        scipy.io.savemat('saveMATs/TL_' + loadMAT + '_fold' + str(fold) + '.mat', {
            'Y_true': Y_testtrue,
            'Y_pred': Y_testpred,
            'Y_predScores': Y_testpredScores,
            'AccuracyTH': history.acc,
            'LossTH': history.loss,
            'nClasses': nClasses,
            'nepochs': nepochs,
            'batchsize':  batchsize,
            'scores': scores,
            'cnf_matrix': cnf_matrix,
            'nFolds': nFolds,
            'fold': fold,
            'learnignRate': learnignRate,
        })
        #

        t1 = time.time()  # t1 at the end
        print("Total Run Time: ", int(t1 - t0), " seconds")


        fold += 1


import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
winsound.MessageBeep()







