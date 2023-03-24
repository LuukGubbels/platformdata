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
    iters = 2
    jobs = 5

    try:
        opts, args = getopt.getopt(argv,
                "trp:trn:tep:ten:o:n:j:",
                ["trpos=","trneg=","tepos=","teneg=","ofile=","iters=","jobs="])
    except getopt.GetoptError:
        sys.exit(2)
    if '?' in args or 'help' in args:
        print('Help for stateart.py')
        print('This file is used to benchmark the current state of the art.')
        print()
        print('Options:')
        print('-trp, --trpos:  Defines the file from which positive training data should be read.')
        print('-trn, --trneg:  Defines the file from which negative training data should be read.')
        print('-tep, --tepos:  Defines the file from which positive testing data should be read.')
        print('-ten, --teneg:  Defines the file from which negative testing data should be read.')
        print('-o, --ofile:    Defines the file to which results should be written.')
        print('-n, --iters:    Defines the number of machines should be used.')
        print('-j, --jobs:     Defines the number of machines that should be run in parallel.')

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

        X_train, _, tfidfvectorizer, cv = tm.processing(X_train)
        self.alg.fit(X_train,y_train)

        with open(self.X_test_pos) as f:
            te_pos_n = sum(1 for line in f) - 1
        y_test_pos = np.ones(te_pos_n)
        with open(self.X_test_neg) as f:
            te_neg_n = sum(1 for line in f) - 1
        y_test_neg = np.zeros(te_neg_n)
        y_test = np.concatenate([y_test_pos, y_test_neg])

        te_pos = np.random.choice(range(1,te_pos_n), size = int(te_pos_n/10), replace = False)
        te_pos = np.concatenate([te_pos,[0]])
        X_test_pos1 = pd.read_csv(self.X_test_pos, skiprows = lambda i: i not in te_pos)
        te_neg = np.random.choice(range(1,te_neg_n), size=int(te_neg_n/10), replace = False)
        te_neg = np.concatenate([te_neg,[0]])
        X_test_neg1 = pd.read_csv(self.X_test_neg, skiprows = lambda i: i not in te_neg)
        X_test = pd.concat([X_test_pos1, X_test_neg1]) 
        X_test = tm.processing(self.X_test, tfidfvectorizer=tfidfvectorizer, cv=cv)

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

        y_pred = self.alg.predict_proba(X_test, cal = False)[:,1]
        self.posest_P.value = np.sum(y_pred)
        [_, FP], [FN, _] = tm.confusion_est(y_test, y_pred)
        self.bias_P.value = (FP-FN)/len(y_test)
        self.sPCC_P.value = np.round(tm.rho(y_test, y_pred),4)
        self.acc_P.value = np.round(sklearn.metrics.accuracy_score(y_test, y_pred>=threshold), 4)            
        self.BA_P.value = np.round(sklearn.metrics.balanced_accuracy_score(y_test, y_pred>=threshold),4)
        self.MCC_P.value = np.round(sklearn.metrics.matthews_corrcoef(y_test, y_pred>=threshold),4)
        self.AUROC_P.value = np.round(sklearn.metrics.roc_auc_score(y_test, y_pred),4)

        y_pred = self.alg.predict(X_test, new_threshold = False, cal = True)
        threshold = self.alg.threshold
        self.posest_C.value = np.sum(y_pred)
        [_, FP], [FN, _] = tm.confusion_est(y_test, y_pred)
        self.bias_C.value = (FP-FN)/len(y_test)
        self.sPCC_C.value = np.round(tm.rho(y_test, y_pred),4)
        self.acc_C.value = np.round(sklearn.metrics.accuracy_score(y_test, y_pred>=threshold), 4)            
        self.BA_C.value = np.round(sklearn.metrics.balanced_accuracy_score(y_test, y_pred>=threshold),4)
        self.MCC_C.value = np.round(sklearn.metrics.matthews_corrcoef(y_test, y_pred>=threshold),4)
        self.AUROC_C.value = np.round(sklearn.metrics.roc_auc_score(y_test, y_pred),4)

        y_pred = self.alg.predict_proba(X_test, cal = True)[:,1]
        self.posest_CP.value = np.sum(y_pred)
        [_, FP], [FN, _] = tm.confusion_est(y_test, y_pred)
        self.bias_CP.value = (FP-FN)/len(y_test)
        self.sPCC_CP.value = np.round(tm.rho(y_test, y_pred),4)
        self.acc_CP.value = np.round(sklearn.metrics.accuracy_score(y_test, y_pred>=threshold), 4)            
        self.BA_CP.value = np.round(sklearn.metrics.balanced_accuracy_score(y_test, y_pred>=threshold),4)
        self.MCC_CP.value = np.round(sklearn.metrics.matthews_corrcoef(y_test, y_pred>=threshold),4)
        self.AUROC_CP.value = np.round(sklearn.metrics.roc_auc_score(y_test, y_pred),4)       

if __name__ == "__main__":
    start = time()
    alg = LogisticRegression()
    alg = bc.calibrator_binary(alg, density = 'test')

    processes = [Machine(alg, i, X_train_pos, X_train_neg, 
                         X_test_pos, X_test_neg) for i in range(iters)]
    batches = range(int((iters-1)/jobs+1))
    for i in tqdm(batches):
        for process in processes[i*jobs:(i+1)*jobs]:
            process.start()
        for process in processes[i*jobs:(i+1)*jobs]:
            process.join()
    TP = np.array([process.TP.value, process.TP.value, process.TP.value, process.TP.value])
    posest = np.zeros(4)
    bias = np.zeros(4)
    sPCC = np.zeros(4)
    acc = np.zeros(4)
    BA = np.zeros(4)
    MCC = np.zeros(4)
    AUROC = np.zeros(4)

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

    dfmets = pd.DataFrame(columns =['BayesCal?','Proba?',
                                'TP','Pos. Est.','Bias','sPCC',
                                'Acc','BA','MCC','AUROC'])
    bayes = ["No","No","Yes","Yes"]
    proba = ["No","Yes","No","Yes"]
    for j in range(4):
        mets = np.array([bayes[j],proba[j],TP[j],posest[j],bias[j],sPCC[j],acc[j],BA[j],MCC[j],AUROC[j]])
        dfmets.loc[len(dfmets)] = mets
    dfmets.to_csv(outfile)
