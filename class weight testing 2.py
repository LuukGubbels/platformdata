
import sys, getopt
sys.path.append('../')

if __name__ == "__main__":
    print()
    argv = sys.argv[1:]
    
    X_train_pos = 'data/X_train_pos processed.csv'
    X_train_neg = 'data/X_train_neg processed.csv'
    X_test_pos = 'data/X_test_pos processed.csv'
    X_test_neg = 'data/X_test_neg processed.csv'
    outfile = 'results/ClassWeightResults.csv'
    steps = 2
    iters = 5
    jobs = 5
    feats = False

    try:
        opts, args = getopt.getopt(argv,
                "P:N:p:n:o:s:m:j:f:",
                ["trpos=","trneg=","tepos=","teneg=","ofile=","steps=","iters=","jobs=","feats="])
    except getopt.GetoptError:
        sys.exit(2)
    if '?' in args or 'help' in args:
        print('Help for "class weight testing.py"')
        print('This file is used to benchmark LogisticRegression using different class weights for positive cases in a logspace.')
        print('Note that all input files should be processed by processed.py before using.')
        print()
        print('Options:')
        print('-P, --trpos: Defines the file from which positive training data should be read. Input as a .csv file with extension. Defaults to "data/X_train_pos processed.csv".')
        print('-N, --trneg: Defines the file from which negative training data should be read. Input as a .csv file with extension. Defaults to "data/X_train_neg processed.csv".')
        print('-p, --tepos: Defines the file from which positive testing data should be read. Input as a .csv file with extension. Defaults to "data/X_test_pos processed.csv".')
        print('-n, --teneg: Defines the file from which negative testing data should be read. Input as a .csv file with extension. Defaults to "data/X_test_neg processed.csv".')
        print('-o, --ofile:   Defines the file in which results should be stored. Input with a file extension. Defaults to "results/ClassWeightResults.csv".')
        print('-s, --steps:   Defines the number of steps should be taken in the logspace. Non-integer numbers will be rounded down. Defaults to 2.')
        print('-m, --iters:   Defines the number of machines should be used per step in the logspace. Non-integer numbers will be rounded down. Defaults to 5.')
        print('-j, --jobs:    Defines the number of machines should be ran in parallel. Non-integer numbers will be rounded down. Defaults to 5.')
        print('-f, --feats:   Defines if the features should be stored. Defaults to False.')

        print()
        sys.exit(2)    
    for opt, arg in opts:
        if opt in ("-P","--trpos"):
            X_train_pos = arg
        elif opt in ("-N","--trneg"):
            X_train_neg = arg
        elif opt in ("-p","--tepos"):
            X_test_pos = arg
        elif opt in ("-n","--teneg"):
            X_test_neg = arg
        elif opt in ("-o","--ofile"):
            outfile = arg
        elif opt in ("-s","--steps"):
            steps = int(arg)
        elif opt in ("-m","--iters"):
            iters = int(arg)
        elif opt in ("-j","--jobs"):
            jobs = int(arg)
        elif otp in ("-f","--feats") and arg == 'True':
            feats = True

    try:
        fo = open(outfile, "wb")
        fo.close()
    except:
        print("Please close the outgoing file!")
        print()
        sys.exit(2)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import numpy as np
import sklearn.model_selection 
import sklearn.metrics
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import thesis_module as tm
import BayesCCal as bc
from tqdm import tqdm
from copy import copy
from multiprocessing import Lock, Value, Process
from time import time

class Machine(Process):
    # Initialize a process, storing the associated datasets and Values for storing metrics
    def __init__(self, alg, id, X_train_pos, X_train_neg, X_test_pos, X_test_neg):
        Process.__init__(self)
        self.alg = alg
        self.id = id
        self.X_train_pos = X_train_pos
        self.X_train_neg = X_train_neg
        self.X_test_pos = X_test_pos
        self.X_test_neg = X_test_neg
        self.TP = Value('f',0)
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
        
        # Form the training and testing data set
        X_train_pos1 = pd.read_csv(self.X_train_pos)
        tr_pos_n = len(X_train_pos1)
        y_train_pos = np.ones(tr_pos_n)

        with open(self.X_train_neg) as f:
            tr_neg_n = sum(1 for line in f)
        with open(self.X_test_pos) as f:
            te_pos_n = sum(1 for line in f)
        with open(self.X_test_neg) as f:
            te_neg_n = sum(1 for line in f)
        tr_neg = np.random.choice(range(1,tr_neg_n), size = int(tr_neg_n/10), replace = False) 
        # tr_neg = np.random.choice(range(1,tr_neg_n), size = int(tr_pos_n*3/7), replace = False)
        tr_neg = np.concatenate([tr_neg,[0]])
        X_train_neg1 = pd.read_csv(self.X_train_neg, skiprows=lambda i: i not in tr_neg)
        X_train = pd.concat([X_train_pos1, X_train_neg1])
        y_train_neg = np.zeros(len(X_train_neg1))
        y_train = np.concatenate([y_train_pos, y_train_neg])

        te_pos = np.random.choice(range(1,te_pos_n), size = int(te_pos_n/10), replace=False)
        te_pos = np.concatenate([te_pos,[0]])
        X_test_pos1 = pd.read_csv(self.X_test_pos, skiprows = lambda i: i not in te_pos)
        te_neg = np.random.choice(range(1,te_neg_n), size = int(te_neg_n/10), replace=False)
        te_neg = np.concatenate([te_neg,[0]])
        X_test_neg1 = pd.read_csv(self.X_test_neg, skiprows = lambda i: i not in te_neg)
        X_test = pd.concat([X_test_pos1, X_test_neg1])
        y_test_pos = np.ones(len(X_test_pos1))
        y_test_neg = np.zeros(len(X_test_neg1))
        y_test = np.concatenate([y_test_pos,y_test_neg])

        self.TP.value = np.sum(y_test)

        # Process the training set and fit the machine, storing features if needed
        X_train, features, tfidfvectorizer, cv = tm.processing(X_train)
        features = np.concatenate([features,['language']])
        
        self.alg.fit(X_train,y_train)
        if feats:
            features_w = np.vstack([features, self.alg.classifier.coef_[0]]).T
            pd.DataFrame(features_w, columns=["Features","Weights"]).to_csv('features/ClassWeightFeats'+str(np.round(self.alg.classifier.class_weight[1],2))+' '+str(self.id)+'.csv')

        # Process the testing set
        X_test = tm.processing(X_test, tfidfvectorizer=tfidfvectorizer, cv=cv)
        
        # Predict the labels/probabilities of the testing set per variant and calculate and store the metrics
        y_pred = self.alg.predict(X_test, new_threshold = False, cal = False)
        threshold = 0.5
        self.posest.value = np.sum(y_pred)
        [_, FP], [FN, _] = tm.confusion_est(y_test, y_pred)
        self.bias.value = (FP-FN)/len(y_test)
        self.sPCC.value = np.round(tm.rho(y_test, y_pred),4)
        self.acc.value = np.round(sklearn.metrics.accuracy_score(y_test, y_pred>=threshold), 4)            
        self.BA.value = np.round(sklearn.metrics.balanced_accuracy_score(y_test, y_pred>=threshold),4)
        self.MCC.value = np.round(sklearn.metrics.matthews_corrcoef(y_test, y_pred>=threshold),4)
        self.AUROC.value = np.round(sklearn.metrics.roc_auc_score(y_test, y_pred),4)
       

        y_predP = self.alg.predict_proba(X_test, cal = False)[:,1]
        self.posest_P.value = np.sum(y_predP)
        [_, FP], [FN, _] = tm.confusion_est(y_test, y_predP)
        self.bias_P.value = (FP-FN)/len(y_test)
        self.sPCC_P.value = np.round(tm.rho(y_test, y_predP),4)
        self.acc_P.value = np.round(sklearn.metrics.accuracy_score(y_test, y_predP>=threshold), 4)            
        self.BA_P.value = np.round(sklearn.metrics.balanced_accuracy_score(y_test, y_predP>=threshold),4)
        self.MCC_P.value = np.round(sklearn.metrics.matthews_corrcoef(y_test, y_predP>=threshold),4)
        self.AUROC_P.value = np.round(sklearn.metrics.roc_auc_score(y_test, y_predP),4)

        y_predC = self.alg.predict(X_test, new_threshold = False, cal = True)
        threshold = self.alg.threshold
        self.posest_C.value = np.sum(y_predC)
        [_, FP], [FN, _] = tm.confusion_est(y_test, y_predC)
        self.bias_C.value = (FP-FN)/len(y_test)
        self.sPCC_C.value = np.round(tm.rho(y_test, y_predC),4)
        self.acc_C.value = np.round(sklearn.metrics.accuracy_score(y_test, y_predC>=threshold), 4)            
        self.BA_C.value = np.round(sklearn.metrics.balanced_accuracy_score(y_test, y_predC>=threshold),4)
        self.MCC_C.value = np.round(sklearn.metrics.matthews_corrcoef(y_test, y_predC>=threshold),4)
        self.AUROC_C.value = np.round(sklearn.metrics.roc_auc_score(y_test, y_predC),4)

        y_predCP = self.alg.predict_proba(X_test, cal = True)[:,1]
        self.posest_CP.value = np.sum(y_predCP)
        [_, FP], [FN, _] = tm.confusion_est(y_test, y_predCP)
        self.bias_CP.value = (FP-FN)/len(y_test)
        self.sPCC_CP.value = np.round(tm.rho(y_test, y_predCP),4)
        self.acc_CP.value = np.round(sklearn.metrics.accuracy_score(y_test, y_predCP>=threshold), 4)            
        self.BA_CP.value = np.round(sklearn.metrics.balanced_accuracy_score(y_test, y_predCP>=threshold),4)
        self.MCC_CP.value = np.round(sklearn.metrics.matthews_corrcoef(y_test, y_predCP>=threshold),4)
        self.AUROC_CP.value = np.round(sklearn.metrics.roc_auc_score(y_test, y_predCP),4)       
        
        # Store the predictions of all four variants in a file (this file will be overwritten a lot of times, so only the last machine's predictions will be stored)
        pd.DataFrame([y_pred,y_predP,y_predC,y_predCP]).to_csv('results/ClassWeightPredictions.csv')

if __name__ == "__main__":
    start = time()

    print("Iterating over the log scale:")    
    scale = np.logspace(0.1,2,steps)
    for j in tqdm(scale, leave=False):
        alg = LogisticRegression(class_weight={0:1,1:j})
        alg = bc.calibrator_binary(alg, density='test')

        processes = [Machine(alg, i, X_train_pos, X_train_neg, X_test_pos, X_test_neg) for i in range(iters)]
        
        # Execute the processes in batches and wait until they are finished to start the new batch
        batches = range(int((iters-1)/jobs+1))
        for i in batches:
            for process in processes[i*jobs:(i+1)*jobs]:
                process.start()
            for process in processes[i*jobs:(i+1)*jobs]:
                process.join()
        
        # Calculate and store the average metrics per variant
        TP = np.array([process.TP.value,process.TP.value,process.TP.value,process.TP.value], dtype='float64')
        posest = np.array([0,0,0,0], dtype='float64')
        bias = np.array([0,0,0,0], dtype='float64')
        sPCC = np.array([0,0,0,0], dtype='float64')
        acc = np.array([0,0,0,0], dtype='float64')
        BA = np.array([0,0,0,0], dtype='float64')
        MCC = np.array([0,0,0,0], dtype='float64')
        AUROC = np.array([0,0,0,0], dtype='float64')

        for process in processes:
            # TP += np.array([process.TP, process.TP, process.TP, process.TP])
            posest += np.array([process.posest.value, process.posest_P.value, process.posest_C.value, process.posest_CP.value])
            bias += np.array([process.bias.value, process.bias_P.value, process.bias_C.value, process.bias_CP.value])
            sPCC += np.array([process.sPCC.value, process.sPCC_P.value, process.sPCC_C.value, process.sPCC_CP.value])
            acc += np.array([process.acc.value, process.acc_P.value, process.acc_C.value, process.acc_CP.value])
            BA += np.array([process.BA.value, process.BA_P.value, process.BA_C.value, process.BA_CP.value])
            MCC += np.array([process.MCC.value, process.MCC_P.value, process.MCC_C.value, process.MCC_CP.value])
            AUROC += np.array([process.AUROC.value, process.AUROC_P.value, process.AUROC_C.value, process.AUROC_CP.value])
        # TP /= iters
        posest /= iters
        bias /= iters
        sPCC /= iters
        acc /= iters
        BA /= iters
        MCC /= iters
        AUROC /= iters

        # Store the metrics as results in a .csv
        dfmets = pd.DataFrame(columns= ['Positive class weight','BayesCal?','Proba?',
                                       'TP','Pos. Est.','Bias','sPCC',
                                       'Acc','BA','MCC','AUROC'])
        bayes = ["No","No","Yes","Yes"]
        proba = ["No","Yes","No","Yes"]
        for i in range(4):
            mets = np.array([str(np.round(j,4)),bayes[i],proba[i],TP[i],posest[i],bias[i],sPCC[i],acc[i],BA[i],MCC[i],AUROC[i]])
            dfmets.loc[len(dfmets)] = mets
        dfmets.to_csv(outfile[:-4]+' '+str(np.round(j,4))+'.csv')
        
    print(time()-start)
