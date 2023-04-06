import sys, getopt
sys.path.append('../')
if __name__ == "__main__":
    print()
    argv = sys.argv[1:]

    X_train_pos = 'data/X_train_pos processed.csv'
    X_train_neg = 'data/X_train_neg processed.csv'
    X_test_pos = 'data/X_test_pos processed.csv'
    X_test_neg = 'data/X_test_neg processed.csv'
    outfile = 'results/StateArtResults.csv'
    iters = 2
    jobs = 5
    feats = False

    try:
        opts, args = getopt.getopt(argv,
                "P:N:p:n:o:m:j:f",
                ["trpos=","trneg=","tepos=","teneg=","ofile=","iters=","jobs=","feats="])
    except getopt.GetoptError:
        sys.exit(2)
    if '?' in args or 'help' in args:
        print('Help for stateart.py')
        print('This file is used to benchmark the current state of the art.')
        print('Note that all input files should be processed by processed.py before using.')
        print()
        print('Options:')
        print('-P, --trpos: Defines the file from which positive training data should be read. Input as a .csv file with extension. Defaults to "data/X_train_pos processed.csv".')
        print('-N, --trneg: Defines the file from which negative training data should be read. Input as a .csv file with extension. Defaults to "data/X_train_neg processed.csv".')
        print('-p, --tepos: Defines the file from which positive testing data should be read. Input as a .csv file with extension. Defaults to "data/X_test_pos processed.csv".')
        print('-n, --teneg: Defines the file from which negative testing data should be read. Input as a .csv file with extension. Defaults to "data/X_test_neg processed.csv".')
        print('-o, --ofile:   Defines the file in which results should be stored. Input with a file extension. Defaults to "results/StateArtResults.csv".')
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
            feats = True    
    try:
        fo = open(outfile, "wb")
        fo.close()
    except:
        print("Could not open outfile!")
        print()
        sys.exit(2)
    
import numpy as np
import sklearn.model_selection
import sklearn.metrics
import pandas as pd
from sklearn.linear_model import LogisticRegression
import thesis_module as tm
import BayesCCal as bc
from tqdm import tqdm
from multiprocessing import Value, Process
from time import time

class Machine(Process):
    # Initialize a process, storing the associated datasets and Values for storing metrics
    def __init__(self, alg, id, X_train_pos, X_train_neg,X_test_pos, X_test_neg):    
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
        
        # Form the training dataset and process it
        with open(self.X_train_pos) as f:
            tr_pos_n = sum(1 for line in f) - 1
        y_train_pos = np.ones(tr_pos_n)
        with open(self.X_train_neg) as f:
            tr_neg_n = sum(1 for line in f) - 1
        y_train_neg = np.zeros(tr_pos_n)
        y_train = np.concatenate([y_train_pos, y_train_neg])

        X_train_pos1 = pd.read_csv(self.X_train_pos)
        
        tr_neg = np.random.choice(range(1,tr_neg_n), size = tr_pos_n, replace = False)
        tr_neg = np.concatenate([tr_neg, [0]])
        X_train_neg1 = pd.read_csv(self.X_train_neg, skiprows = lambda i: i not in tr_neg)
        X_train = pd.concat([X_train_pos1, X_train_neg1])

        X_train, features, tfidfvectorizer, cv = tm.processing(X_train)
        features = np.concatenate([features,['language']])
        
        # Fit the machine and store the features if needed
        self.alg.fit(X_train,y_train)
        if feats:
            features_w = np.vstack([features, self.alg.classifier.coef_[0]]).T
            pd.DataFrame(features_w, columns=["Features","Weights"]).to_csv('features/StateArtFeats'+str(self.id)+'.csv')

        # Form the testing set and process it
        with open(self.X_test_pos) as f:
            te_pos_n = sum(1 for line in f) - 1
        y_test_pos = np.ones(int(te_pos_n/10))
        with open(self.X_test_neg) as f:
            te_neg_n = sum(1 for line in f) - 1
        y_test_neg = np.zeros(int(te_neg_n/10))
        y_test = np.concatenate([y_test_pos, y_test_neg])

        te_pos = np.random.choice(range(1,te_pos_n), size = int(te_pos_n/10), replace = False)
        te_pos = np.concatenate([te_pos,[0]])
        X_test_pos1 = pd.read_csv(self.X_test_pos, skiprows = lambda i: i not in te_pos)
        te_neg = np.random.choice(range(1,te_neg_n), size=int(te_neg_n/10), replace = False)
        te_neg = np.concatenate([te_neg,[0]])
        X_test_neg1 = pd.read_csv(self.X_test_neg, skiprows = lambda i: i not in te_neg)
        X_test = pd.concat([X_test_pos1, X_test_neg1]) 
        X_test = tm.processing(X_test, tfidfvectorizer=tfidfvectorizer, cv=cv)

        # Predict the labels/probabilities per variant and calculate the metrics
        y_pred = self.alg.predict(X_test, cal = False)
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
        
        # Store the predictions of all four variants in a file (this file will be overwritten, so only the last machine's predictions will be stored).
        pd.DataFrame([y_pred,y_predP,y_predC,y_predCP]).to_csv('results/StateArtPredictions.csv')
if __name__ == "__main__":
    start = time()
    alg = LogisticRegression()
    alg = bc.calibrator_binary(alg, density = 'test')

    # Exectue the processes in batches and wait until they are finished to start a new batch
    processes = [Machine(alg, i, X_train_pos, X_train_neg, 
                         X_test_pos, X_test_neg) for i in range(iters)]
    batches = range(int((iters-1)/jobs+1))
    for i in tqdm(batches):
        for process in processes[i*jobs:(i+1)*jobs]:
            process.start()
        for process in processes[i*jobs:(i+1)*jobs]:
            process.join()
            
    # Initialize arrays usef for storing metrics
    TP = np.array([process.TP.value, process.TP.value, process.TP.value, process.TP.value])
    posest = np.zeros(4)
    bias = np.zeros(4)
    sPCC = np.zeros(4)
    acc = np.zeros(4)
    BA = np.zeros(4)
    MCC = np.zeros(4)
    AUROC = np.zeros(4)

    # Average the metrics over all machines
    for process in processes:
        posest[0] += process.posest.value
        bias[0] += process.bias.value
        sPCC[0] += process.sPCC.value
        acc[0] += process.acc.value
        BA[0] += process.BA.value
        MCC[0] += process.MCC.value
        AUROC[0] += process.AUROC.value

        posest[1] += process.posest_P.value
        bias[1] += process.bias_P.value
        sPCC[1] += process.sPCC_P.value
        acc[1] += process.acc_P.value
        BA[1] += process.BA_P.value
        MCC[1] += process.MCC_P.value
        AUROC[1] += process.AUROC_P.value

        posest[2] += process.posest_C.value
        bias[2] += process.bias_C.value
        sPCC[2] += process.sPCC_C.value
        acc[2] += process.acc_C.value
        BA[2] += process.BA_C.value
        MCC[2] += process.MCC_C.value
        AUROC[2] += process.AUROC_C.value

        posest[3] += process.posest_CP.value
        bias[3] += process.bias_CP.value
        sPCC[3] += process.sPCC_CP.value
        acc[3] += process.acc_CP.value
        BA[3] += process.BA_CP.value
        MCC[3] += process.MCC_CP.value
        AUROC[3] += process.AUROC_CP.value

    posest /= iters
    bias /= iters
    sPCC /= iters
    acc /= iters
    BA /= iters
    MCC /= iters
    AUROC /= iters

    # Store the metrics as results in a .csv
    dfmets = pd.DataFrame(columns =['BayesCal?','Proba?',
                                'TP','Pos. Est.','Bias','sPCC',
                                'Acc','BA','MCC','AUROC'])
    bayes = ["No","No","Yes","Yes"]
    proba = ["No","Yes","No","Yes"]
    for j in range(4):
        mets = np.array([bayes[j],proba[j],TP[j],posest[j],bias[j],sPCC[j],acc[j],BA[j],MCC[j],AUROC[j]])
        dfmets.loc[len(dfmets)] = mets
    dfmets.to_csv(outfile)
