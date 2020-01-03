import hdf5storage
import numpy as np
# ##########################################################################
# https://scikit-learn.org/stable/modules/model_evaluation.html
from sklearn import metrics


for fold in range(1, 11):
    #  Load Matlab files
    # loadMAT = 'TL_A_SHM8_freq2_fold'+str(fold)
    # loadMAT = 'A_SHM1_TH_fold'+str(fold)
    loadMAT = 'A_SHM8_freq_fold'+str(fold)


    print('********* ' + loadMAT + ' *********')
    mat = hdf5storage.loadmat('saveMATs/' + loadMAT + '.mat')

    Y_pred = mat['Y_pred']
    ypredlist = Y_pred.tolist()
    y_pred = ypredlist[0]
    y_pred

    Y_true = mat['Y_true']
    ytruelist = Y_true.tolist()
    y_true = ytruelist[0]
    y_true

    # #############################################
    # Classification report
    # print(metrics.classification_report(y_true, y_pred))

    # #############################################
    # Accuracy score
    AS = metrics.accuracy_score(y_true, y_pred)
    print("Accuracy score = {0:.3f}".format(AS))

    # #############################################
    # Cohen’s kappa
    # CK = metrics.cohen_kappa_score(y_true, y_pred)
    # print("Cohen’s kappa = {0:.3f}".format(CK))

    # #############################################
    # Hamming loss
    HL = metrics.hamming_loss(y_true, y_pred)
    print("Hamming loss = {0:.3f}".format(HL))

    # #############################################
    ps = metrics.precision_score(y_true, y_pred, average='weighted')
    print("precision_score = {0:.3f}".format(ps))

    rs = metrics.recall_score(y_true, y_pred, average='weighted')
    print("recall_score = {0:.3f}".format(rs))

    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    print("f1_score = {0:.3f}".format(f1))

    fbeta = metrics.fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    print("fbeta_score = {0:.3f}".format(fbeta))

    # Save
    import scipy.io

    scipy.io.savemat('saveMetrics/' + loadMAT + '.mat', {
        'Y_true': y_true,
        'Y_pred': y_pred,
        'Accuracy_score': AS,
        # 'Cohen_kappa': CK,
        'Hamming_loss': HL,
        'precision_score': ps,
        'recall_score': rs,
        'f1_score': f1,
        'fbeta_score': fbeta,
    })

