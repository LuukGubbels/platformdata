import sys, getopt
sys.path.append('../')

if __name__ == "__main__":
    print()
    argv = sys.argv[1:]

    X_train_pos = '../platformSample_data2 processed.csv'
    X_train_neg = '../platformSample_data2 processed.csv'
    X_test_pos = '../randomSample_purified_data2 processed.csv'
    X_test_neg = '../randomSample_purified_data2 processed.csv'
    outfile = '../Results/EnsembleResults.csv'
    iters = 2
    size = 3
    avg = 2
    
    try:
        opts, args = getopt.getopt(argv,
                "trp:trn:tep:ten:o:n:s:a:",
                ["trpos=","trneg=","tepos=","teneg=","test=","ofile=","iters=","size=","avg="])
    except getopt.GetoptError:
        sys.exit(2)
    if '?' in args or 'help' in args:
        print('Help for "bootstrap testing 2.py"')
        print('This file is used to benchmark linear SVMs using bootstrapping.')
        print()
        print('Options:')
        print('-tr, --train:  Defines the file from which training data should be read. Input as a .csv file with extension.')
        print('-te, --test:   Defines the file from which testing data should be read. Input as a .csv file with extension.')
        # print('-i, --ifile:   Defines the files from which files should be read. Input as a python list with file extension.')
        print('-o, --ofile:   Defines the file in which results should be stored. Input with a file extension.')
        print('-s, --size:    Defines the number of machines should be used per ensemble. Non-integer numbers will be rounded down.')
        print('-a, --avg:     Defines the number of iterations needed to average the effects of stemming.')

        print()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-trp","--trpos"):
            X_train_pos = arg
        elif opt in ("-trn","--trneg"):
            X_train_neg = arg
        elif opt in ("-tep","--tepos"):
            X_test_pos = arg
        elif opt in ("-ten","--teneg"):
            X_test_neg = arg
        # if opt in ("-i","--ifile"):
        #     infiles = arg.strip('][').split(',')
        elif opt in ("-o","--ofile"):
            outfile = arg
        elif opt in ("-n","--iters"):
            iters = int(arg)
        elif opt in ("-s","--size"):
            size = int(arg)
        elif opt in ("-a","--avg"):
            avg = int(arg)

# take 80/20 split on training sets per machine
# use 0.2 validation set to give weight to machine

    try:
        fo = open(outfile, "wb")
        fo.close()
    except:
        print("Please close the outgoing file!")
        print()
        sys.exit(2)

import numpy as np
import sklearn.model_selection
import sklearn.metrics
import pandas as pd
from copy import copy
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from Modules import thesis_module as tm
from Modules import BayesCCal as bc
from multiprocessing import Value, Process
from time import time

class Machine(Process):
    def __init__(self, alg, id, X_train, y_train, X_test, y_test, size, avg):
        Process.__init__(self)
        self.alg = []
        for _ in range(size):
            self.alg.append(copy(alg))
        self.id = id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.size = size
        self.avg = avg
        self.TP = np.sum(y_test)

        self.w = Value('f',0)
        self.posest = Value('f',0)
        self.bias = Value('f',0)
        self.sPCC = Value('f',0)
        self.acc = Value('f',0)
        self.BA = Value('f',0)
        self.MCC = Value('f',0)
        self.AUROC = Value('f',0)

        self.w_P = Value('f',0)
        self.posest_P = Value('f',0)
        self.bias_P = Value('f',0)
        self.sPCC_P = Value('f',0)
        self.acc_P = Value('f',0)
        self.BA_P = Value('f',0)
        self.MCC_P = Value('f',0)
        self.AUROC_P = Value('f',0)

        self.w_C = Value('f',0)
        self.posest_C = Value('f',0)
        self.bias_C = Value('f',0)
        self.sPCC_C = Value('f',0)
        self.acc_C = Value('f',0)
        self.BA_C = Value('f',0)
        self.MCC_C = Value('f',0)
        self.AUROC_C = Value('f',0)

        self.w_CP = Value('f',0)
        self.posest_CP = Value('f',0)
        self.bias_CP = Value('f',0)
        self.sPCC_CP = Value('f',0)
        self.acc_CP = Value('f',0)
        self.BA_CP = Value('f',0)
        self.MCC_CP = Value('f',0)
        self.AUROC_CP = Value('f',0)
    
    def run(self):
        features = []
        tfidfvectorizer = copy(features)
        cv = copy(features)
        w = copy(features)
        w_P = copy(w)
        w_C = copy(w)
        w_CP = copy(w)
        y_pred = np.zeros((self.size,len(self.X_test)))
        y_pred_P = copy(y_pred)
        y_pred_C = copy(y_pred)
        y_pred_CP = copy(y_pred)
        for ind,i in enumerate(self.alg):
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(self.X_train, self.y_train, test_size = 0.2)
            X_train, feats, TFvec, Cvec = tm.processing(X_train)
            features.append(feats)
            tfidfvectorizer.append(TFvec)
            cv.append(Cvec)
            i.fit(X_train, y_train)
            X_val = tm.processing(X_val, tfidfvectorizer = TFvec, cv = Cvec)
            
            y_pred_v = i.predict(X_val, new_threshold = False, cal = False)
            w.append(sklearn.metrics.accuracy_score(y_val, y_pred_v))
            y_pred_v = i.predict_proba(X_val, cal = False)[:,1]
            w_P.append(sklearn.metrics.accuracy_score(y_val, y_pred_v>=0.5))
            y_pred_v = i.predict(X_val, new_threshold = False, cal = True)
            w_C.append(sklearn.metrics.accuracy_score(y_val, y_pred_v))
            y_pred_v = i.predict_proba(X_val, cal = True)[:,1]
            w_CP.append(sklearn.metrics.accuracy_score(y_val, y_pred_v>=i.threshold))

            X_test = tm.processing(self.X_test, tfidfvectorizer=TFvec, cv=Cvec)
            y_pred[ind] = i.predict(X_test, new_threshold=False, cal=False)
            y_pred_P[ind] += i.predict_proba(X_test, cal = False)[:,1]
            y_pred_C[ind] += i.predict(X_test, new_threshold=False, cal=True)
            y_pred_CP[ind] += i.predict_proba(X_test, cal = True)[:,1]


        weights = np.zeros((len(self.X_test),self.size))
        for _ in range(self.avg):
            test_tfidf, test_feats, _, _ = tm.processing(self.X_test, mindf=1)
            test_feats = np.append(test_feats,'')
            indices = (test_tfidf!=0)*1
            for i in range(len(indices)):
                index = np.unique(np.arange(len(indices[1]))*indices[i])
                if index[0]==0 and indices[i][0]!=1:
                    index = index[1:]
                feats = list(test_feats[index])
                for j in range(self.size):
                    weights[i][j] += len(set(features[j]).intersection(set(feats)))/len(set(feats))
        weights = weights/self.avg
        weights_P = weights*np.array(w_P)
        weights_C = weights*np.array(w_C)
        weights_CP = weights*np.array(w_CP)
        weights = weights*np.array(w)
        weights = sk.preprocessing.normalize(weights, 'l1')
        weights_P = sk.preprocessing.normalize(weights_P, 'l1')
        weights_C = sk.preprocessing.normalize(weights_C, 'l1')
        weights_CP = sk.preprocessing.normalize(weights_CP, 'l1')

        y_pred = np.sum(np.multiply(y_pred.reshape(-1,self.size), weights),axis=1).reshape(-1)
        y_pred_P = np.sum(np.multiply(y_pred_P.reshape(-1,self.size), weights_P),axis=1).reshape(-1)
        y_pred_C = np.sum(np.multiply(y_pred_C.reshape(-1,self.size), weights_C),axis=1).reshape(-1)
        y_pred_CP = np.sum(np.multiply(y_pred_CP.reshape(-1,self.size), weights_CP),axis=1).reshape(-1)

        # X_test = tm.processing(self.X_test, tfidfvectorizer = tfidfvectorizer[ind], cv=cv[ind])
        # y_pred = i.predict(X_test, new_threshold=False, cal=False)*weights[ind]
        # y_pred_P += i.predict_proba(X_test, cal = False)[:,1]*weights_P[ind]
        # y_pred_C += i.predict(X_test, new_threshold=False, cal=True)*weights_C[ind]
        # y_pred_CP += i.predict_proba(X_test, cal = True)[:,1]*weights_CP[ind]

        y_pred /= self.size
        y_pred = y_pred >= 0.5
        y_pred_P /= self.size
        y_pred_C /= self.size
        y_pred_C = y_pred_C >= 0.5
        y_pred_CP /= self.size

        threshold = 0.5
        self.posest.value = np.sum(y_pred)
        [_, FP], [FN, _] = tm.confusion_est(self.y_test, y_pred)
        self.bias.value = (FP-FN)/len(self.y_test)
        self.sPCC.value = np.round(tm.rho(self.y_test, y_pred),4)
        self.acc.value = np.round(sklearn.metrics.accuracy_score(self.y_test, y_pred>=threshold), 4)            
        self.BA.value = np.round(sklearn.metrics.balanced_accuracy_score(self.y_test, y_pred>=threshold),4)
        self.MCC.value = np.round(sklearn.metrics.matthews_corrcoef(self.y_test, y_pred>=threshold),4)
        self.AUROC.value = np.round(sklearn.metrics.roc_auc_score(self.y_test, y_pred),4)

        self.posest_P.value = np.sum(y_pred_P)
        [_, FP], [FN, _] = tm.confusion_est(self.y_test, y_pred_P)
        self.bias_P.value = (FP-FN)/len(self.y_test)
        self.sPCC_P.value = np.round(tm.rho(self.y_test, y_pred_P),4)
        self.acc_P.value = np.round(sklearn.metrics.accuracy_score(self.y_test, y_pred_P>=threshold), 4)            
        self.BA_P.value = np.round(sklearn.metrics.balanced_accuracy_score(self.y_test, y_pred_P>=threshold),4)
        self.MCC_P.value = np.round(sklearn.metrics.matthews_corrcoef(self.y_test, y_pred_P>=threshold),4)
        self.AUROC_P.value = np.round(sklearn.metrics.roc_auc_score(self.y_test, y_pred_P),4)

        self.posest_C.value = np.sum(y_pred_C)
        [_, FP], [FN, _] = tm.confusion_est(self.y_test, y_pred_C)
        self.bias_C.value = (FP-FN)/len(self.y_test)
        self.sPCC_C.value = np.round(tm.rho(self.y_test, y_pred_C),4)
        self.acc_C.value = np.round(sklearn.metrics.accuracy_score(self.y_test, y_pred_C>=threshold), 4)            
        self.BA_C.value = np.round(sklearn.metrics.balanced_accuracy_score(self.y_test, y_pred_C>=threshold),4)
        self.MCC_C.value = np.round(sklearn.metrics.matthews_corrcoef(self.y_test, y_pred_C>=threshold),4)
        self.AUROC_C.value = np.round(sklearn.metrics.roc_auc_score(self.y_test, y_pred_C),4)

        self.posest_CP.value = np.sum(y_pred_CP)
        [_, FP], [FN, _] = tm.confusion_est(self.y_test, y_pred_CP)
        self.bias_CP.value = (FP-FN)/len(self.y_test)
        self.sPCC_CP.value = np.round(tm.rho(self.y_test, y_pred_CP),4)
        self.acc_CP.value = np.round(sklearn.metrics.accuracy_score(self.y_test, y_pred_CP>=threshold), 4)            
        self.BA_CP.value = np.round(sklearn.metrics.balanced_accuracy_score(self.y_test, y_pred_CP>=threshold),4)
        self.MCC_CP.value = np.round(sklearn.metrics.matthews_corrcoef(self.y_test, y_pred_CP>=threshold),4)
        self.AUROC_CP.value = np.round(sklearn.metrics.roc_auc_score(self.y_test, y_pred_CP),4)       
        
if __name__ == "__main__":
    alg = LogisticRegression()
    alg = bc.calibrator_binary(alg, density = "test")
    
    X_train_pos = pd.read_csv(X_train_pos)
    y_train_pos = np.ones(len(X_train_pos))
    with open(X_train_neg) as f:
        tr_neg_n = sum(1 for line in f)
    y_train_neg = np.zeros(int(tr_neg_n/10))
    y_train = np.concatenate([y_train_pos, y_train_neg])
    with open(X_test_pos) as f:
        te_pos_n = sum(1 for line in f)
    y_test_pos = np.ones(int(te_pos_n/10))
    with open(X_test_neg) as f:
        te_neg_n = sum(1 for line in f)
    y_test_neg = np.zeros(int(te_neg_n/10))
    y_test = np.concatenate([y_test_pos, y_test_neg])

    processes = []
    for i in range(iters):
        tr_neg = np.random.choice(range(1,tr_neg_n), size=int(tr_neg_n/10), replace=False)
        tr_neg = np.concatenate([tr_neg,[0]])
        X_train_neg1 = pd.read_csv(X_train_neg, skiprows=lambda i: i not in tr_neg)
        X_train = pd.concat([X_train_pos, X_train_neg1])

        te_pos = np.random.choice(range(1,te_pos_n), size=int(te_pos_n/10), replace = False)
        te_pos = np.concatenate([te_pos, [0]])
        X_test_pos1 = pd.read_csv(X_test_pos, skiprows=lambda i: i not in te_pos)
        te_neg = np.random.choice(range(1,te_neg_n), size = int(te_neg_n/10), replace = False)
        te_neg = np.concatenate([te_neg, [0]])
        X_test_neg1 = pd.read_csv(X_test_neg, skiprows = lambda i: i not in te_neg)
        X_test = pd.concat([X_test_pos1, X_test_neg1])

        process = Machine(alg, i, X_train, y_train, X_test, y_test, size, avg)
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    TP = np.array([0,0,0,0], dtype='float64')
    posest = np.array([0,0,0,0], dtype='float64')
    bias = np.array([0,0,0,0], dtype='float64')
    sPCC = np.array([0,0,0,0], dtype='float64')
    acc = np.array([0,0,0,0], dtype='float64')
    BA = np.array([0,0,0,0], dtype='float64')
    MCC = np.array([0,0,0,0], dtype='float64')
    AUROC = np.array([0,0,0,0], dtype='float64')

    for process in processes:
        TP += np.array([process.TP, process.TP, process.TP, process.TP])
        posest += np.array([process.posest.value, process.posest_P.value, process.posest_C.value, process.posest_CP.value])
        bias += np.array([process.bias.value, process.bias_P.value, process.bias_C.value, process.bias_CP.value])
        sPCC += np.array([process.sPCC.value, process.sPCC_P.value, process.sPCC_C.value, process.sPCC_CP.value])
        acc += np.array([process.acc.value, process.acc_P.value, process.acc_C.value, process.acc_CP.value])
        BA += np.array([process.BA.value, process.BA_P.value, process.BA_C.value, process.BA_CP.value])
        MCC += np.array([process.MCC.value, process.MCC_P.value, process.MCC_C.value, process.MCC_CP.value])
        AUROC += np.array([process.AUROC.value, process.AUROC_P.value, process.AUROC_C.value, process.AUROC_CP.value])
    TP /= iters
    posest /= iters
    bias /= iters
    sPCC /= iters
    acc /= iters
    BA /= iters
    MCC /= iters
    AUROC /= iters

    dfmets = pd.DataFrame(columns= ['BayesCal?','Proba?',
                                    'TP','Pos. Est.','Bias','sPCC',
                                    'Acc','BA','MCC','AUROC'])
    bayes = ["No","No","Yes","Yes"]
    proba = ["No","Yes","No","Yes"]
    for i in range(4):
        mets = np.array([bayes[i],proba[i],TP[i],posest[i],bias[i],sPCC[i],acc[i],BA[i],MCC[i],AUROC[i]])
        dfmets.loc[len(dfmets)] = mets
    dfmets.to_csv(outfile)
