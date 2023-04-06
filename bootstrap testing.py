import sys, getopt
sys.path.append('../')
if __name__ == "__main__":
    print()
    argv = sys.argv[1:]

    X_train_pos = 'data/X_train_pos processed.csv'
    X_train_neg = 'data/X_train_neg processed.csv'
    X_test_pos = 'data/X_test_pos processed.csv'
    X_test_neg = 'data/X_test_neg processed.csv'
    outfile = 'results/BootstrapResults.csv'
    iters = 100
    jobs = 5
    feats = False

    try:
        opts, args = getopt.getopt(argv,
                "P:N:p:n:o:m:j:f:",
                ["trpos=","trneg=","tepos=","teneg=","test=","ofile=","iters=","jobs=","feats="])
    except getopt.GetoptError:
        sys.exit(2)
    if '?' in args or 'help' in args:
        print('Help for "bootstrap testing 2.py"')
        print('This file is used to benchmark LogisticRegression using bootstrapping.')
        print('Note that all input files should be processed by processed.py before using.')
        print()
        print('Options:')
        print('-P, --trpos: Defines the file from which positive training data should be read. Input as a .csv file with extension. Defaults to "data/X_train_pos processed.csv".')
        print('-N, --trneg: Defines the file from which negative training data should be read. Input as a .csv file with extension. Defaults to "data/X_train_neg processed.csv".')
        print('-p, --tepos: Defines the file from which positive testing data should be read. Input as a .csv file with extension. Defaults to "data/X_test_pos processed.csv".')
        print('-n, --teneg: Defines the file from which negative testing data should be read. Input as a .csv file with extension. Defaults to "data/X_test_neg processed.csv".')
        print('-o, --ofile:   Defines the file in which results should be stored. Input with a file extension. Defaults to "results/BootstrapResults.csv".')
        print('-m, --iters:   Defines the number of machines / bootstrap samples should be used. Non-integer numbers will be rounded down. Defaults to 100.')
        print('-j, --jobs:    Defines the number of machines should be ran in parallel. Non-integer numbers will be rounded down. Defaults to 5.')
        print('-f, --feats:   Defines if features should be stored. Defaults to False.')

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
        elif opt in ("-m","--iters"):
            iters = int(arg)
        elif opt in ("-j","--jobs"):
            jobs = int(arg)
        elif opt in ("-f","--feats") and arg == 'True':
            print("Features will be stored!")
            feats = True
    
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
from multiprocessing import Value, Process, Manager
from time import time

class Machine(Process):
    # Initialize a process, storing the associated datasets and arrays for storing predictions
    def __init__(self, alg, id, X_train_pos, X_train_neg, 
                 X_test,
                 y_pred, y_predP, y_predC, y_predCP):
        Process.__init__(self)
        self.alg = alg
        self.id = id
        self.X_train_pos = X_train_pos
        self.X_train_neg = X_train_neg
        self.X_test = X_test
        self.y_pred = y_pred
        self.y_predP = y_predP
        self.y_predC = y_predC
        self.y_predCP = y_predCP

    def run(self):
        
        # Form the training data set and process it
        with open(self.X_train_pos) as f:
            tr_pos_n = sum(1 for line in f) - 1
        y_train_pos = np.ones(tr_pos_n)
        with open(self.X_train_neg) as f:
            tr_neg_n = sum(1 for line in f) - 1
        
        tr_pos = np.random.choice(range(1,tr_pos_n), size = tr_pos_n, replace = True)
        X_train_pos1 = pd.read_csv(self.X_train_pos)
        X_train_pos1 = X_train_pos1.iloc[tr_pos-1]
        
        tr_neg = np.random.choice(range(1,tr_neg_n), size = int((tr_pos_n-1)*7/3), replace = False)
        y_train_neg = np.zeros(len(tr_neg))
        tr_neg = np.concatenate([tr_neg, [0]])
        X_train_neg1 = pd.read_csv(self.X_train_neg, skiprows= lambda i: i not in tr_neg)
        X_train = pd.concat([X_train_pos1, X_train_neg1])
        y_train = np.concatenate([y_train_pos, y_train_neg])
    
        X_train, features, tfidfvectorizer, cv = tm.processing(X_train)
        features = np.concatenate([features,['language']])
        
        # Fit the machine and store the features if needed
        self.alg.fit(X_train,y_train)
        if feats:
            features_w = np.vstack([features, self.alg.classifier.coef_[0]]).T
            pd.DataFrame(features_w, columns=["Features","Weights"]).to_csv('features/BootstrapFeats'+str(self.id)+'.csv')

        # Process the test set and use the different predictions variants
        self.X_test = tm.processing(self.X_test, tfidfvectorizer=tfidfvectorizer, cv=cv)
        
        self.y_pred.append(self.alg.predict(self.X_test, new_threshold = False, cal = False))
        self.y_predP.append(self.alg.predict_proba(self.X_test, cal = False)[:,1])
        self.y_predC.append(self.alg.predict(self.X_test, new_threshold = False, cal = True))
        self.y_predCP.append(self.alg.predict_proba(self.X_test, cal = True)[:,1]) 

if __name__ == "__main__":
    start = time()
    alg = LogisticRegression()
    alg = bc.calibrator_binary(alg, density = 'test')
    
    # Form the test set
    with open(X_test_pos) as f:
        te_pos_n = sum(1 for line in f) - 1
    y_test_pos = np.ones(int(te_pos_n/10))
    te_pos = np.random.choice(range(1,te_pos_n), size=int(te_pos_n/10), replace = False)
    te_pos = np.concatenate([te_pos, [0]])
    X_test_pos = pd.read_csv(X_test_pos, skiprows= lambda i: i not in te_pos)
    with open(X_test_neg) as f:
        te_neg_n = sum(1 for line in f) - 1
    y_test_neg = np.zeros(int(te_neg_n/10))
    y_test = np.concatenate([y_test_pos, y_test_neg])
    te_neg = np.random.choice(range(1,te_neg_n), size=int(te_neg_n/10), replace = False)
    te_neg = np.concatenate([te_neg,[0]])
    X_test_neg = pd.read_csv(X_test_neg, skiprows = lambda i: i not in te_neg)
    X_test = pd.concat([X_test_pos, X_test_neg])
    
    # Initialize a manager to manage the prediction arrays, such that the processes can store their predictions
    man = Manager()
    y_pred = man.list()
    y_predP = man.list()
    y_predC = man.list()
    y_predCP = man.list()
    processes = [Machine(alg, i, X_train_pos, X_train_neg,
                                    X_test,
                                    y_pred, y_predP,
                                    y_predC, y_predCP) for i in range(iters)]
    
    # Execute the processes in batches and wait until they are finished to start the new batch
    batches = range(int((iters-1)/jobs+1))
    for i in tqdm(batches):
        for process in processes[i*jobs:(i+1)*jobs]:
            process.start()
        for process in processes[i*jobs:(i+1)*jobs]:
            process.join()
        for process in processes[i*jobs:(i+1)*jobs]:
            process.terminate()
            
    # Store the metrics per variant and determine the average vote
    TP = np.sum(y_test)
    TP = np.array([TP,TP,TP,TP], dtype='float64')
    posest = np.array([0,0,0,0], dtype='float64')
    bias = np.array([0,0,0,0], dtype='float64')
    sPCC = np.array([0,0,0,0], dtype='float64')
    acc = np.array([0,0,0,0], dtype='float64')
    BA = np.array([0,0,0,0], dtype='float64')
    MCC = np.array([0,0,0,0], dtype='float64')
    AUROC = np.array([0,0,0,0], dtype='float64')
    
    y_pred = (np.mean(np.array(y_pred),axis=0) >= 0.5)
    y_predP = np.mean(np.array(y_predP), axis=0)
    y_predC = (np.mean(np.array(y_predC),axis=0) >= 0.5)
    y_predCP = np.mean(np.array(y_predCP), axis=0)
    pd.DataFrame([y_pred,y_predP,y_predC,y_predCP]).to_csv('results/BootstrapPredictions.csv')    
    
    posest = np.array([np.sum(y_pred),np.sum(y_predP),np.sum(y_predC), np.sum(y_predCP)])
    [_,FP],[FN,_] = tm.confusion_est(y_test, y_pred)
    bias[0] = (FP-FN)/len(y_test)
    sPCC[0] = tm.rho(y_test, y_pred)
    acc[0] = sklearn.metrics.accuracy_score(y_test,y_pred)
    BA[0] = sklearn.metrics.balanced_accuracy_score(y_test,y_pred)
    MCC[0] = sklearn.metrics.matthews_corrcoef(y_test,y_pred)
    AUROC[0] = sklearn.metrics.roc_auc_score(y_test,y_pred)

    [_,FP],[FN,_] = tm.confusion_est(y_test, y_predP)
    bias[1] = (FP-FN)/len(y_test)
    sPCC[1] = tm.rho(y_test, y_predP)
    acc[1] = sklearn.metrics.accuracy_score(y_test,y_predP >= 0.5)
    BA[1] = sklearn.metrics.balanced_accuracy_score(y_test,y_predP >= 0.5)
    MCC[1] = sklearn.metrics.matthews_corrcoef(y_test,y_predP >= 0.5)
    AUROC[1] = sklearn.metrics.roc_auc_score(y_test,y_predP)

    [_,FP],[FN,_] = tm.confusion_est(y_test, y_predC)
    bias[2] = (FP-FN)/len(y_test)
    sPCC[2] = tm.rho(y_test, y_predC)
    acc[2] = sklearn.metrics.accuracy_score(y_test,y_predC)
    BA[2] = sklearn.metrics.balanced_accuracy_score(y_test,y_predC)
    MCC[2] = sklearn.metrics.matthews_corrcoef(y_test,y_predC)
    AUROC[2] = sklearn.metrics.roc_auc_score(y_test,y_predC)

    [_,FP],[FN,_] = tm.confusion_est(y_test, y_predCP)
    bias[3] = (FP-FN)/len(y_test)
    sPCC[3] = tm.rho(y_test, y_predCP)
    acc[3] = sklearn.metrics.accuracy_score(y_test,y_predCP >= 0.5)
    BA[3] = sklearn.metrics.balanced_accuracy_score(y_test,y_predCP >= 0.5)
    MCC[3] = sklearn.metrics.matthews_corrcoef(y_test,y_predCP >= 0.5)
    AUROC[3] = sklearn.metrics.roc_auc_score(y_test,y_predCP)    

    
    # Store the metrics as results in a .csv
    dfmets = pd.DataFrame(columns= ['BayesCal?','Proba?',
                                'TP','Pos. Est.','Bias','sPCC',
                                'Acc','BA','MCC','AUROC'])
    bayes = ["No","No","Yes","Yes"]
    proba = ["No","Yes","No","Yes"]
    for j in range(4):
        mets = np.array([bayes[j],proba[j],TP[j],posest[j],bias[j],sPCC[j],acc[j],BA[j],MCC[j],AUROC[j]])
        dfmets.loc[len(dfmets)] = mets
    dfmets.to_csv(outfile)
    
    print(time()-start)
