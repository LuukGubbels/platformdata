
import sys, getopt
sys.path.append('../')

if __name__ == "__main__":
    print()
    argv = sys.argv[1:]

    # infiles = ['../platformSample_data2.csv','../randomSample_purified_data2.csv']
    X_train_pos = '../platformSample_data2 processed.csv'
    X_train_neg = '../platformSample_data2 processed.csv'
    X_test_pos = '../randomSample_purified_data2 processed.csv'
    X_test_neg = '../randomSample_purified_data2 processed.csv'
    outfile = '../Results/ClassWeightResults.csv'
    steps = 2
    iters = 5
    jobs = 10

    try:
        opts, args = getopt.getopt(argv,
                "trp:trn:tep:ten:o:s:n:",
                ["trpos=","trneg=","tepos=","teneg=","ofile=","steps=","iters="])
    except getopt.GetoptError:
        sys.exit(2)
    if '?' in args or 'help' in args:
        print('Help for "class weight testing.py"')
        print('This file is used to benchmark linear SVMs using different class weights for positive cases in a logspace.')
        print()
        print('Options:')
        print('-tr, --train:  Defines the file from which training data should be read. Input as a .csv file with extension.')
        print('-te, --test:   Defines the file from which testing data should be read. Input as a .csv file with extension.')
        # print('-i, --ifile:   Defines the files from which files should be read. Input as a python list with file extension.')
        print('-o, --ofile:   Defines the file in which results should be stored. Input with a file extension.')
        print('-s, --steps:   Defines the number of steps should be taken in the logspace. Non-integer numbers will be rounded down.')
        print('-n, --iters:   Defines the number of machines should be used per step in the logspace. Non-integer numbers will be rounded down.')

        print()
        sys.exit(2)    
    for opt, arg in opts:
        if opt in ("-trp","--trpos"):
            X_train_pos = arg
        elif opt in ("-trn","--trneg"):
            X_train_neg = arg
        elif opt in ("-tep","--tepos"):
            X_test_pos = arg
        elif opt in ("-ten","--tenog"):
            X_test_neg = arg
        # if opt in ("-i","--ifile"):
        #     infiles = arg.strip('][').split(',')
        elif opt in ("-o","--ofile"):
            outfile = arg
        elif opt in ("-s","--steps"):
            steps = int(arg)
        elif opt in ("-n","--iters"):
            iters = int(arg)

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

        X_train, _, tfidfvectorizer, cv = tm.processing(X_train)

        self.alg.fit(X_train,y_train)
        X_test = tm.processing(X_test, tfidfvectorizer=tfidfvectorizer, cv=cv)
        
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

# Arguments
# infile (file or list of files to )
# outfile
# steps
# iters

# dfsvm = pd.DataFrame(columns= ['Positive class weight', 'BayesCal?','TP','Pos. Est.','Bias','Pos. Est. P','BiasP', 'sPCC','Acc','AUROC','BA','MCC'])
# print("Iterating over the log scale:")

# Multithread -> gebruik multiprocessing ipv multithreading
# size = iters
# per machine eigen test set samplen, n/10, stratified (class dist behouden)
# per sample van de test set predicten

if __name__ == "__main__":
    start = time()

    print("Iterating over the log scale:")
    # df_train = tm.preprocess(train)
    # y_train = np.array([df_train['platform']])[0]
    # X_train = df_train.drop(['platform'], axis=1)
    # X_test_pos = X_test_neg = X_train_pos = X_train_neg = X_train
    
    scale = np.logspace(0.1,2,steps)

    # X_train_pos = pd.read_csv(X_train_pos)
    # y_train_pos = np.ones(len(X_train_pos))

    # with open(X_train_neg) as f:
    #     tr_neg_n = sum(1 for line in f)
    # y_train_neg = np.zeros(int(tr_neg_n/10))

    # with open(X_test_pos) as f:
    #     te_pos_n = sum(1 for line in f)
    # y_test_pos = np.zeros(int(te_pos_n/10))
    
    # with open(X_test_neg) as f:
    #     te_neg_n = sum(1 for line in f)
    # y_test_neg = np.zeros(int(te_neg_n/10))

    for j in tqdm(scale, leave=False):
        alg = LogisticRegression(class_weight={0:1,1:j})
        alg = bc.calibrator_binary(alg, density='test')

        processes = [Machine(alg, i, X_train_pos, X_train_neg, X_test_pos, X_test_neg) for i in range(iters)]
        
        # processes = []
        # for i in range(iters):
        #     tr_neg = np.random.choice(range(1,tr_neg_n), size = int(tr_neg_n/10),replace=False)
        #     tr_neg = np.concatenate([tr_neg, [0]])
        #     X_train_neg1 = pd.read_csv(X_train_neg, skiprows=lambda i: i not in tr_neg)
        #     X_train = pd.concat([X_train_pos, X_train_neg1])
        #     y_train = np.concatenate([y_train_pos, y_train_neg])

        #     te_pos = np.random.choice(range(1,te_pos_n), size=int(te_pos_n/10), replace=False)
        #     te_pos = np.concatenate([te_pos,[0]])
        #     X_test_pos1 = pd.read_csv(X_test_pos, skiprows = lambda i: i not in te_pos)
        #     te_neg = np.random.choice(range(1,te_neg_n), size=int(te_neg_n/10), replace=False)
        #     te_neg = np.concatenate([te_neg,[0]])
        #     X_test_neg1 = pd.read_csV(X_test_neg, skiprows = lambda i: i not in te_neg)
        #     X_test = pd.concat([X_test_pos1, X_test_neg1])
        #     y_test = np.concatenate([y_test_pos, y_test_neg])            

        #     process = Machine(alg, i, X_train, y_train, X_test, y_test)
        #     processes.append(process)
        batches = range(int((iters-1)/jobs+1))
        for i in batches:
            for process in processes[i*jobs:(i+1)*jobs]:
                process.start()
            for process in processes[i*jobs:(i+1)*jobs]:
                process.join()
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

# print("Iterating over the log scale: Done!")
# print()

# print("Storing metrics: ...")
# # dfsvm.to_csv(outfile)
# print("Storing metrics: Done!")
# print("Finished!")
# print()
