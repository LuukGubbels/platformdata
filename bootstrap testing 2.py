import sys, getopt
sys.path.append('../')
if __name__ == "__main__":
    print()
    argv = sys.argv[1:]

    X_train_pos = '../platformSample_data2 processed.csv'
    X_train_neg = '../platformSample_data2 processed.csv'
    X_test_pos = '../randomSample_purified_data2 processed.csv'
    X_test_neg = '../randomSample_purified_data2 processed.csv'
    outfile = '../Results/BootstrapResults.csv'
    iters = 11
    jobs = 5

    try:
        opts, args = getopt.getopt(argv,
                "trp:trn:tep:ten:o:n:j:",
                ["trpos=","trneg=","tepos=","teneg=","test=","ofile=","iters=","jobs="])
    except getopt.GetoptError:
        sys.exit(2)
    if '?' in args or 'help' in args:
        print('Help for "bootstrap testing 2.py"')
        print('This file is used to benchmark linear SVMs using bootstrapping.')
        print()
        print('Options:')
        print('-trp, --trpos:  Defines the file from which positive training data should be read. Input as a .csv file with extension.')
        print('-trn, --trneg:  Defines the file from which negative training data should be read. Input as a .csv file with extension.')
        print('-tep, --tepos:  Defines the file from which positive test data should be read. Input as a .csv file with extension.')
        print('-ten, --teneg:  Defines the file from which negative test data should be read. Input as a .csv file with extension.')
        print('-o, --ofile:   Defines the file in which results should be stored. Input with a file extension.')
        print('-n, --iters:   Defines the number of machines / bootstrap samples should be used. Non-integer numbers will be rounded down.')
        print('-j, --jobs:    Defines the number of jobs should be used.')

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
        elif opt in ("-j","--jobs"):
            jobs = int(arg)
    
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
from sklearn.linear_model import LogisticRegression
from copy import copy
import thesis_module as tm
import BayesCCal as bc
from tqdm import tqdm
from multiprocessing import Value, Process
from time import time

class Machine(Process):
    def __init__(self, alg, id, X_train, y_train, X_test, y_test):
        Process.__init__(self)
        self.alg = alg
        self.id = id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.TP = np.sum(y_test)

        self.posest = Value('f',0)
        self.bias = Value('f',0)
        self.sPCC = Value('f',0)
        self.acc = Value('f',0)
        self.BA = Value('f',0)
        self.MCC = Value('f',0)
        self.AUROC = Value('f',0)

        self.posest_P = Value('f',0)
        self.bias_P = Value('f',0)
        self.sPCC_P = Value('f',0)
        self.acc_P = Value('f',0)
        self.BA_P = Value('f',0)
        self.MCC_P = Value('f',0)
        self.AUROC_P = Value('f',0)

        self.posest_C = Value('f',0)
        self.bias_C = Value('f',0)
        self.sPCC_C = Value('f',0)
        self.acc_C = Value('f',0)
        self.BA_C = Value('f',0)
        self.MCC_C = Value('f',0)
        self.AUROC_C = Value('f',0)

        self.posest_CP = Value('f',0)
        self.bias_CP = Value('f',0)
        self.sPCC_CP = Value('f',0)
        self.acc_CP = Value('f',0)
        self.BA_CP = Value('f',0)
        self.MCC_CP = Value('f',0)
        self.AUROC_CP = Value('f',0)

    def run(self):
        self.X_train, _, tfidfvectorizer, cv = tm.processing(self.X_train)

        self.alg.fit(self.X_train,self.y_train)
        self.X_test = tm.processing(self.X_test, tfidfvectorizer=tfidfvectorizer, cv=cv)
        
        y_pred = self.alg.predict(self.X_test, new_threshold = False, cal = False)
        threshold = 0.5
        self.posest.value = np.sum(y_pred)
        [_, FP], [FN, _] = tm.confusion_est(self.y_test, y_pred)
        self.bias.value = (FP-FN)/len(self.y_test)
        self.sPCC.value = np.round(tm.rho(self.y_test, y_pred),4)
        self.acc.value = np.round(sklearn.metrics.accuracy_score(self.y_test, y_pred>=threshold), 4)            
        self.BA.value = np.round(sklearn.metrics.balanced_accuracy_score(self.y_test, y_pred>=threshold),4)
        self.MCC.value = np.round(sklearn.metrics.matthews_corrcoef(self.y_test, y_pred>=threshold),4)
        self.AUROC.value = np.round(sklearn.metrics.roc_auc_score(self.y_test, y_pred),4)

        y_pred = self.alg.predict_proba(self.X_test, cal = False)[:,1]
        self.posest_P.value = np.sum(y_pred)
        [_, FP], [FN, _] = tm.confusion_est(self.y_test, y_pred)
        self.bias_P.value = (FP-FN)/len(self.y_test)
        self.sPCC_P.value = np.round(tm.rho(self.y_test, y_pred),4)
        self.acc_P.value = np.round(sklearn.metrics.accuracy_score(self.y_test, y_pred>=threshold), 4)            
        self.BA_P.value = np.round(sklearn.metrics.balanced_accuracy_score(self.y_test, y_pred>=threshold),4)
        self.MCC_P.value = np.round(sklearn.metrics.matthews_corrcoef(self.y_test, y_pred>=threshold),4)
        self.AUROC_P.value = np.round(sklearn.metrics.roc_auc_score(self.y_test, y_pred),4)

        y_pred = self.alg.predict(self.X_test, new_threshold = False, cal = True)
        threshold = self.alg.threshold
        self.posest_C.value = np.sum(y_pred)
        [_, FP], [FN, _] = tm.confusion_est(self.y_test, y_pred)
        self.bias_C.value = (FP-FN)/len(self.y_test)
        self.sPCC_C.value = np.round(tm.rho(self.y_test, y_pred),4)
        self.acc_C.value = np.round(sklearn.metrics.accuracy_score(self.y_test, y_pred>=threshold), 4)            
        self.BA_C.value = np.round(sklearn.metrics.balanced_accuracy_score(self.y_test, y_pred>=threshold),4)
        self.MCC_C.value = np.round(sklearn.metrics.matthews_corrcoef(self.y_test, y_pred>=threshold),4)
        self.AUROC_C.value = np.round(sklearn.metrics.roc_auc_score(self.y_test, y_pred),4)

        y_pred = self.alg.predict_proba(self.X_test, cal = True)[:,1]
        self.posest_CP.value = np.sum(y_pred)
        [_, FP], [FN, _] = tm.confusion_est(self.y_test, y_pred)
        self.bias_CP.value = (FP-FN)/len(self.y_test)
        self.sPCC_CP.value = np.round(tm.rho(self.y_test, y_pred),4)
        self.acc_CP.value = np.round(sklearn.metrics.accuracy_score(self.y_test, y_pred>=threshold), 4)            
        self.BA_CP.value = np.round(sklearn.metrics.balanced_accuracy_score(self.y_test, y_pred>=threshold),4)
        self.MCC_CP.value = np.round(sklearn.metrics.matthews_corrcoef(self.y_test, y_pred>=threshold),4)
        self.AUROC_CP.value = np.round(sklearn.metrics.roc_auc_score(self.y_test, y_pred),4)       

if __name__ == "__main__":
    start = time()
    alg = LogisticRegression()
    alg = bc.calibrator_binary(alg, density = 'test')

    with open(X_train_pos) as f:
        tr_pos_n = sum(1 for line in f) - 1
    y_train_pos = np.ones(tr_pos_n)
    with open(X_train_neg) as f:
        tr_neg_n = sum(1 for line in f) - 1
    y_train_neg = np.zeros(tr_neg_n)
    y_train = np.concatenate([y_train_pos, y_train_neg])
    with open(X_test_pos) as f:
        te_pos_n = sum(1 for line in f) - 1
    y_test_pos = np.ones(te_pos_n)
    with open(X_test_neg) as f:
        te_neg_n = sum(1 for line in f) - 1
    y_test_neg = np.zeros(te_neg_n)
    y_test = np.concatenate([y_test_pos, y_test_neg])

    processes = []
    for i in range(iters):
        tr_pos = np.random.choice(range(1,tr_pos_n), size = tr_pos_n, replace = True)
        # tr_pos  =np.concatenate([tr_pos, [0]])
        X_train_pos1 = pd.read_csv(X_train_pos)
        X_train_pos1 = X_train_pos1.iloc[tr_pos-1]
        
        tr_neg = np.random.choice(range(1,tr_neg_n), size = int((tr_pos_n-1)*7/3), replace = False)
#         tr_neg = np.random.choice(range(1,tr_neg_n), size = int(tr_neg_n/10), replace = False)
        tr_neg = np.concatenate([tr_neg, [0]])
        X_train_neg1 = pd.read_csv(X_train_neg, skiprows= lambda i: i not in tr_neg)
        X_train = pd.concat([X_train_pos1, X_train_neg1])

        te_pos = np.random.choice(range(1,te_pos_n), size = int(te_pos_n/10), replace = False)
        te_pos = np.concatenate([te_pos,[0]])
        X_test_pos1 = pd.read_csv(X_test_pos, skiprows = lambda i: i not in te_pos)
        te_neg = np.random.choice(range(1,te_neg_n), size=int(te_neg_n/10), replace = False)
        te_neg = np.concatenate([te_neg,[0]])
        X_test_neg1 = pd.read_csv(X_test_neg, skiprows = lambda i: i not in te_neg)
        X_test = pd.concat([X_test_pos1, X_test_neg1])

        processes.append(Machine(alg, i, X_train, y_train,
                                    X_test, y_test))
    batches = range(int((iters-1)/jobs+1))
    for i in tqdm(batches):
        for process in processes[i*jobs:(i+1)*jobs]:
            process.start()
        for process in processes[i*jobs:(i+1)*jobs]:
            process.join()
        TP = np.array([0,0,0,0], dtype='float64')
        posest = np.array([0,0,0,0], dtype='float64')
        bias = np.array([0,0,0,0], dtype='float64')
        sPCC = np.array([0,0,0,0], dtype='float64')
        acc = np.array([0,0,0,0], dtype='float64')
        BA = np.array([0,0,0,0], dtype='float64')
        MCC = np.array([0,0,0,0], dtype='float64')
        AUROC = np.array([0,0,0,0], dtype='float64')


        for process in processes[i*jobs:(i+1)*jobs]:
            TP += np.array([process.TP, process.TP, process.TP, process.TP])
            posest += np.array([process.posest.value, process.posest_P.value, process.posest_C.value, process.posest_CP.value])
            bias += np.array([process.bias.value, process.bias_P.value, process.bias_C.value, process.bias_CP.value])
            sPCC += np.array([process.sPCC.value, process.sPCC_P.value, process.sPCC_C.value, process.sPCC_CP.value])
            acc += np.array([process.acc.value, process.acc_P.value, process.acc_C.value, process.acc_CP.value])
            BA += np.array([process.BA.value, process.BA_P.value, process.BA_C.value, process.BA_CP.value])
            MCC += np.array([process.MCC.value, process.MCC_P.value, process.MCC_C.value, process.MCC_CP.value])
            AUROC += np.array([process.AUROC.value, process.AUROC_P.value, process.AUROC_C.value, process.AUROC_CP.value])
        size = len(processes[i*jobs:(i+1)*jobs])
        TP /= size
        posest /= size
        bias /= size
        sPCC /= size
        acc /= size
        BA /= size
        MCC /= size
        AUROC /= size

        dfmets = pd.DataFrame(columns= ['Positive class weight','BayesCal?','Proba?',
                                    'TP','Pos. Est.','Bias','sPCC',
                                    'Acc','BA','MCC','AUROC'])
        bayes = ["No","No","Yes","Yes"]
        proba = ["No","Yes","No","Yes"]
        for j in range(4):
            mets = np.array([str(i),bayes[j],proba[j],TP[j],posest[j],bias[j],sPCC[j],acc[j],BA[j],MCC[j],AUROC[j]])
            dfmets.loc[len(dfmets)] = mets
        dfmets.to_csv(outfile[:-4]+' '+str(i)+'.csv')
    
    print(time()-start)
