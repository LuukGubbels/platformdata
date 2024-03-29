import sys, getopt
sys.path.append('../')

if __name__ == "__main__":
    print()
    argv = sys.argv[1:]

    X_train_pos = 'data/X_train_pos processed.csv'
    X_train_neg = 'data/X_train_neg processed.csv'
    X_test_pos = 'data/X_test_pos processed.csv'
    X_test_neg = 'data/X_test_neg processed.csv'
    outfile = 'results/EnsembleResults.csv'
    iters = 3
    size = 3
    avg = 2
    jobs = 5
    F = False
    
    try:
        opts, args = getopt.getopt(argv,
                "P:N:p:n:o:m:s:a:j:f:",
                ["trpos=","trneg=","tepos=","teneg=","test=","ofile=","iters=","size=","avg=","jobs=","feats="])
    except getopt.GetoptError:
        sys.exit(2)
    if '?' in args or 'help' in args:
        print('Help for "bootstrap testing 2.py"')
        print('This file is used to benchmark linear SVMs using bootstrapping.')
        print('Note that all input files should be processed by processed.py before using.')
        print()
        print('Options:')
        print('-P, --trpos: Defines the file from which positive training data should be read. Input as a .csv file with extension. Defaults to "data/X_train_pos processed.csv".')
        print('-N, --trneg: Defines the file from which negative training data should be read. Input as a .csv file with extension. Defaults to "data/X_train_neg processed.csv".')
        print('-p, --tepos: Defines the file from which positive testing data should be read. Input as a .csv file with extension. Defaults to "data/X_test_pos processed.csv".')
        print('-n, --teneg: Defines the file from which negative testing data should be read. Input as a .csv file with extension. Defaults to "data/X_test_neg processed.csv".')
        print('-o, --ofile:   Defines the file in which results should be stored. Input with a file extension. Defaults to "results/EnsembelResults.csv".')
        print('-m, --iters:   Defines the number of ensembles that should be used. Non-integer numbers will be rounded down. Defaults to 3.')
        print('-s, --size:    Defines the number of machines should be used per ensemble. Non-integer numbers will be rounded down. Defaults to 3.')
        print('-a, --avg:     Defines the number of iterations needed to average the effects of stemming. Non-integer numbers will be rounded down. Defaults to 2.')
        print('j, --jobs:     Defines the number of machines should be ran in parallel. Non-integer numbers will be rounded down. Defaults to 5.')
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
        elif opt in ("-m","--iters"):
            iters = int(arg)
        elif opt in ("-s","--size"):
            size = int(arg)
        elif opt in ("-a","--avg"):
            avg = int(arg)
        elif opt in ("-j","--jobs"):
            avg = int(arg)
        elif otp in ("-f","--feats") and arg == 'True':
            F = True

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
from tqdm import tqdm
import thesis_module as tm
import BayesCCal as bc
from multiprocessing import Value, Process
from time import time

class Machine(Process):
    # Initialize a process, storing the associated datasets and Values for storing metrics
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
        
        # Prepare arrays to store weights and predictions
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
        
        # Go over all machines in the ensemble, one by one
        for ind,i in enumerate(self.alg):
            
            # Make a training / validation split and process them, storing the features if needed
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(self.X_train, self.y_train, test_size = 0.2)
            X_train, feats, TFvec, Cvec = tm.processing(X_train)
            if F:
                Fe = np.concatenate([feats,['language']])
                pd.DataFrame(features_w, columns=["Features","Weights"]).to_csv('features/EnsembleFeats'+str(self.id)+'.'+str(ind)+'.csv')
                
            features.append(feats)
            tfidfvectorizer.append(TFvec)
            cv.append(Cvec)
            i.fit(X_train, y_train)
            X_val = tm.processing(X_val, tfidfvectorizer = TFvec, cv = Cvec)
            
            # Predict the labels of the validation set and storing the accuracy of that model per variant
            y_pred_v = i.predict(X_val, new_threshold = False, cal = False)
            w.append(sklearn.metrics.accuracy_score(y_val, y_pred_v))
            y_pred_v = i.predict_proba(X_val, cal = False)[:,1]
            w_P.append(sklearn.metrics.accuracy_score(y_val, y_pred_v>=0.5))
            y_pred_v = i.predict(X_val, new_threshold = False, cal = True)
            w_C.append(sklearn.metrics.accuracy_score(y_val, y_pred_v))
            y_pred_v = i.predict_proba(X_val, cal = True)[:,1]
            w_CP.append(sklearn.metrics.accuracy_score(y_val, y_pred_v>=i.threshold))

            # Predict the labels/probabilities of the testing set per variant
            X_test = tm.processing(self.X_test, tfidfvectorizer=TFvec, cv=Cvec)
            y_pred[ind] = i.predict(X_test, new_threshold=False, cal=False)
            y_pred_P[ind] += i.predict_proba(X_test, cal = False)[:,1]
            y_pred_C[ind] += i.predict(X_test, new_threshold=False, cal=True)
            y_pred_CP[ind] += i.predict_proba(X_test, cal = True)[:,1]


        # Add another weight per machine based on how much the features of a machine overlap with that of each point in the test set.
        # We create a weight matrix of size (# machines, # samples)
        weights = np.zeros((len(self.X_test),self.size))
        for _ in range(self.avg):
            test_tfidf, test_feats, _, _ = tm.processing(self.X_test, mindf=50)
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
        
        # Multiply these matrices with the accuracy score per machine and normalize them
        weights_P = weights*np.array(w_P)
        weights_C = weights*np.array(w_C)
        weights_CP = weights*np.array(w_CP)
        weights = weights*np.array(w)
        weights = sk.preprocessing.normalize(weights, 'l1')
        weights_P = sk.preprocessing.normalize(weights_P, 'l1')
        weights_C = sk.preprocessing.normalize(weights_C, 'l1')
        weights_CP = sk.preprocessing.normalize(weights_CP, 'l1')

        # Obtain the weighted vote for label/probability per sample
        y_pred = np.sum(np.multiply(y_pred.reshape(-1,self.size), weights),axis=1).reshape(-1)
        y_pred_P = np.sum(np.multiply(y_pred_P.reshape(-1,self.size), weights_P),axis=1).reshape(-1)
        y_pred_C = np.sum(np.multiply(y_pred_C.reshape(-1,self.size), weights_C),axis=1).reshape(-1)
        y_pred_CP = np.sum(np.multiply(y_pred_CP.reshape(-1,self.size), weights_CP),axis=1).reshape(-1)

        y_pred /= self.size
        y_pred = y_pred >= 0.5
        y_pred_P /= self.size
        y_pred_C /= self.size
        y_pred_C = y_pred_C >= 0.5
        y_pred_CP /= self.size
        
        # Store the predictions of all four variants in a file (this file will be overwritten, so only the last ensemble's predictions will be stored).
        pd.DataFrame([y_pred,y_pred_P,y_pred_C,y_pred_CP]).to_csv('results/EnsemblePredictions.csv')

        # Calculate the metrics per variant
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
    
    # Count the size of the training and testing populations
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

    # Execute the processes in batches and wait until they are finished to start the new batch
    processes = []
    batches = range(int((iters-1)/jobs+1))
    l = copy(iters)
    for j in tqdm(batches):
        k = min(jobs, l)
        while k > 0:
            # Create a training and testing set per ensemble
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

            process = Machine(alg, k, X_train, y_train, X_test, y_test, size, avg)
            processes.append(process)
            process.start()
            k = k-1
        for process in processes[j*jobs:(j+1)*jobs]:
            process.join()
        l = l - jobs
    
    # Initialize arrays used for storing metrics
    TP = np.array([0,0,0,0], dtype='float64')
    posest = np.array([0,0,0,0], dtype='float64')
    bias = np.array([0,0,0,0], dtype='float64')
    sPCC = np.array([0,0,0,0], dtype='float64')
    acc = np.array([0,0,0,0], dtype='float64')
    BA = np.array([0,0,0,0], dtype='float64')
    MCC = np.array([0,0,0,0], dtype='float64')
    AUROC = np.array([0,0,0,0], dtype='float64')

    # Average the metrics over all ensembles
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

    # Store the metrics as results in a .csv
    dfmets = pd.DataFrame(columns= ['BayesCal?','Proba?',
                                    'TP','Pos. Est.','Bias','sPCC',
                                    'Acc','BA','MCC','AUROC'])
    bayes = ["No","No","Yes","Yes"]
    proba = ["No","Yes","No","Yes"]
    for i in range(4):
        mets = np.array([bayes[i],proba[i],TP[i],posest[i],bias[i],sPCC[i],acc[i],BA[i],MCC[i],AUROC[i]])
        dfmets.loc[len(dfmets)] = mets
    dfmets.to_csv(outfile)
