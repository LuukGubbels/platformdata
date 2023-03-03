import sys, getopt
sys.path.append('../')

print()
argv = sys.argv[1:]

# infiles = ['../platformSample_data2.csv','../randomSample_purified_data2.csv']
train = 'platformSample_data2.csv'
test = 'platformSample_data2.csv'
outfile = '../Results/ClassWeightResults.csv'
steps = 1
iters = 1

try:
    opts, args = getopt.getopt(argv,
            "tr:te:o:s:n:",
            ["train=","test=","ofile=","steps=","iters="])
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
    if opt in ("-tr","--train"):
        train = arg
    elif opt in ("-te","--test"):
        test = arg
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
import thesis_module as tm
from tqdm import tqdm
from copy import copy

# Arguments
# infile (file or list of files to )
# outfile
# steps
# iters

df_train = tm.preprocess(train)


y_train = np.array([df_train['platform']])[0]
X_train = df_train.drop(['platform'], axis=1)

scale = np.logspace(0.1,2,steps)

dfsvm = pd.DataFrame(columns= ['Positive class weight', 'BayesCal?','TP','Pos. Est.','Bias','Pos. Est. P','BiasP', 'sPCC','Acc','AUROC','BA','MCC'])
print("Iterating over the log scale:")
for i in tqdm(scale, leave=False):
    mets = 0
    metsC = 0

    posest, bias, posestP, biasP = np.zeros(4)
    posestC, biasC, posestPC, biasPC = np.zeros(4)
    FP,FN,FPP,FNP,FPC,FNC,FPCP,FNCP = np.zeros(8)
    for j in tqdm(range(iters), desc="Training and predicting for weight " + str(np.round(i,3)), leave=False):
        #Training Phase
        # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.2)
        X_train1, _, tfidfvectorizer, cv = tm.processing(X_train)
        alg = SVC(kernel = 'linear', probability=True, class_weight={0:1,1:i})
        alg.fit(X_train1, y_train)
        algC = tm.apply_BayesCCal(alg, X_train1, y_train, density="test")
        algC.fit(X_train1,y_train)

        #Testing Phase
        y_test = []
        file = pd.read_csv(test, sep = ',', chunksize=1)

        y_pred = []
        y_predP = []
        y_predC = []
        y_predCP = []

        while True:
            try:
                df = next(file)
                df = df.fillna(" ")
                df = df[df['text'].str.split().apply(len)>=10]
                dflen = len(df)
                while dflen == 0:
                    df = next(file)
                    df = df.fillna(" ")
                    df = df[df['text'].str.split().apply(len)>=10]
                    dflen = len(df)
                df['text'].str.findall('\w{3,}').str.join(' ')
                df['text'] = df['text'].str.replace("  "," ")
                y_test.append(np.array([df['platform']])[0][0])
                df = df.drop(['platform'],axis=1)
                df = tm.processing(df, tfidfvectorizer=tfidfvectorizer, cv=cv)
                y_pred.append(alg.predict(df)[0])
                y_predP.append(alg.predict_proba(df)[:,1][0])
                y_predC.append(algC.predict(df, new_threshold = False)[0])
                y_predCP.append(algC.predict_proba(df)[:,1][0])
            except:
                break
        TP = np.sum(y_test)

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        y_predP = np.array(y_predP)
        y_predC = np.array(y_predC)
        y_predCP = np.array(y_predCP)

        ### Prediction ###
        try:
            mets += np.round(tm.metrics(y_test, y_predP),4)
        except:
            mets = np.round(tm.metrics(y_test, y_predP),4)
        [_, FP1],[FN1,_] = tm.confusion_est(y_test, y_pred)
        FP += FP1
        FN += FN1
        [_, FP1],[FN1,_] = tm.confusion_est(y_test, y_predP)
        FPP += FP1
        FNP += FN1
        try:
            posest += np.sum(y_pred)
        except:
            posest = np.sum(y_pred)
        try:
            posestP += np.sum(y_predP)
        except:
            posestP = np.sum(y_predP)

        try:
            metsC += np.round(tm.metrics(y_test, y_predCP),4)
        except:
            metsC = np.round(tm.metrics(y_test,y_predCP), 4)
        [_, FP1],[FN1,_] = tm.confusion_est(y_test, y_predC)
        FPC += FP1
        FNC += FN1
        [_, FP1],[FN1,_] = tm.confusion_est(y_test, y_predCP)
        FPCP += FP1
        FNCP += FN1
        try:
            posestC += np.sum(y_predC)
        except:
            posestC = np.sum(y_predC)
        try:
            posestPC += np.sum(y_predCP)
        except:
            posestPC = np.sum(y_predCP)
        
        if j == iters-1:
            posest /= iters
            posestP /= iters
            if FP==FN:
                bias = 0
            else:
                bias = (FP-FN)/len(y_test)
            if FPP==FNP:
                biasP = 0
            else:
                biasP = (FPP-FNP)/len(y_test)
            mets /= iters
            newrow = np.concatenate([[str(i)],["No"],[TP],[posest],[bias],[posestP],[biasP], mets])
            dfsvm.loc[len(dfsvm)] = newrow
        # if j%2==1:
        #     alg = SVC(kernel = 'linear', probability=True, class_weight={0:1,1:i})
        #     alg = tm.apply_BayesCCal(alg, X_train1, y_train, density="test")

        #     ### Prediction (BayesCal) ###
        #     splits = int(len(X_test1)/batch_size + 1)
        #     y_pred = alg.predict(X_test1)
        #     y_predP = alg.predict_proba(X_test1)[:,1]
        #     try:
        #         metsC += np.round(tm.metrics(y_test, y_predP),4)
        #     except:
        #         metsC = np.round(tm.metrics(y_test,y_predP), 4)
        #     [_, FP1],[FN1,_] = tm.confusion_est(y_test, y_pred)
        #     FPC += FP1
        #     FNC += FN1
        #     [_, FP1],[FN1,_] = tm.confusion_est(y_test, y_predP)
        #     FPCP += FP1
        #     FNCP += FN1
        #     try:
        #         posestC += np.sum(y_pred)
        #     except:
        #         posestC = np.sum(y_pred)
        #     try:
        #         posestPC += np.sum(y_predP)
        #     except:
        #         posestPC = np.sum(y_predP)
        
            posestC /= iters
            posestPC /= iters
            if FPC==FNC:
                biasC = 0
            else:
                biasC = (FPC-FNC)/len(y_test)
            if FPCP==FNCP:
                biasPC = 0
            else:
                biasPC = (FPCP - FNCP)/len(y_test)
            metsC /=iters
            newrow = np.concatenate([[str(i)],["Yes"],[TP],[posestC],[biasC],[posestPC],[biasPC], metsC])
            dfsvm.loc[len(dfsvm)] = newrow
print("Iterating over the log scale: Done!")
print()

print("Storing metrics: ...")
dfsvm.to_csv(outfile)
print("Storing metrics: Done!")
print("Finished!")
print()
