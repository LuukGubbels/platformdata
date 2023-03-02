import sys, getopt
sys.path.append('../')

print()
argv = sys.argv[1:]

# Arguments
# infiles
# outfile
# iters

# infiles = ["../platformSample_data2.csv", "../randomSample_purified_data2.csv"]
train = "platformSample_data2.csv"
test = "platformSample_data2.csv"
outfile = 'results/BootstrapResults.csv'
iters = 1

try:
    opts, args = getopt.getopt(argv,
            "tr:te:o:n:",
            ["train=","test=","ofile=","iters="])
except getopt.GetoptError:
    sys.exit(2)
if '?' in args or 'help' in args:
    print('Help for "bootstrap testing.py"')
    print('This file is used to benchmark linear SVMs using bootstrapping.')
    print()
    print('Options:')
    print('-tr, --train:  Defines the file from which training data should be read. Input as a .csv file with extension.')
    print('-te, --test:   Defines the file from which testing data should be read. Input as a .csv file with extension.')
    # print('-i, --ifile:   Defines the files from which files should be read. Input as a python list with file extension.')
    print('-o, --ofile:   Defines the file in which results should be stored. Input with a file extension.')
    print('-n, --iters:   Defines the number of machines / bootstrap samples should be used. Non-integer numbers will be rounded down.')

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
    elif opt in ("-n","--iters"):
        iters = int(arg) 

if train == 0:
    raise Exception("No training data inserted!")
if test == 0:
    raise Exception("No testing data inserted!")

try:
    fo = open(outfile, "wb")
    fo.close()
    fo = open(outfile[:-4] + ' (calibrated).csv', "wb")
    fo.close()
except:
    print("Please close the outgoing file(s)!")
    print()
    sys.exit(2)

def warn(*args, **kwargs):
    print
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import numpy as np
import sklearn.model_selection
import sklearn.metrics
import pandas as pd
from sklearn.svm import SVC
from copy import copy
import thesis_module as tm
from tqdm import tqdm

dfmet = pd.DataFrame(columns = ["Iteration", "TP", "Pos. Est.","Bias", "sPCC", "Acc","AUROC","BA", "MCC"])
dfcal = copy(dfmet)

df_train = tm.preprocess(train)

y_train = np.array([df_train['platform']])[0]
X_train = df_train.drop(['platform'], axis=1)

###
# Train test split section
# - Make a normal train test split
# - for every machine do:
# -> make a bootstrap sample from the positive data 
# -> take a normal sample from the negative data
# ->> such that pos/neg ratio in training set is 25/75
# ->> also try without taking a sample from the negative data, but just take the entire set
# ->>> this depends on the size of the dataset
###

X_train_pos = X_train[y_train==1]
X_train_neg = X_train[y_train==0]
del(X_train)

poslen = len(X_train_pos)
neglen = len(X_train_neg)
ypos = np.ones(poslen)
yneg = np.zeros(neglen)

y_pred = 0
y_predP = 0
y_predC = 0
y_predPC = 0

print("Training and predicting:")
for i in tqdm(range(iters), leave=False):

    #Training Phase

    # Make a bootstrap sample (30/70 split)
    Xpos_train = np.random.randint(0,poslen, poslen)
    ypos_train = ypos[Xpos_train]
    Xpos_train = X_train_pos.iloc[Xpos_train]
    Xneg_train, _, yneg_train, _ = sklearn.model_selection.train_test_split(X_train_neg,yneg, test_size=int(7*poslen/3)) #int(7*poslen/3))

    X_train = pd.concat([Xpos_train, Xneg_train])
    X_train, features, tfidfvectorizer, cv = tm.processing(X_train)
    y_train = np.concatenate((np.ones(poslen),np.zeros(len(Xneg_train))))

    alg = SVC(kernel = 'linear', probability=True)
    alg.fit(X_train, y_train)
    algC = tm.apply_BayesCCal(alg, X_train, y_train, density="test")
    algC.fit(X_train, y_train)

    #Testing Phase

    line = 0
    y_test = []
    file = pd.read_csv(test, sep=';', chunksize=1)

    y_pred = []
    y_predP = []
    y_predC = []
    y_predCP = []
    
    while True:
        # Load dataframe part
        # preprocess
        # process
        # predict
        # concat predictions
        # delete part
        try:

            df = next(file)
            line += 1
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
            df = df.drop(['platform'], axis=1)
            df = tm.processing(df, tfidfvectorizer=tfidfvectorizer, cv=cv)
            y_pred.append(alg.predict(df)[0])
            y_predP.append(alg.predict_proba(df)[:,1][0])
            y_predC.append(algC.predict(df)[0])
            y_predCP.append(algC.predict_proba(df)[:,1][0])
        except:
            break
    print(y_test)
    TP = np.sum(y_test)
    y_pred = np.array(y_pred)
    y_predP = np.array(y_predP)
    y_predC = np.array(y_predC)
    y_predCP = np.array(y_predCP)

    mets = tm.metrics(y_test, y_pred)
    posest = np.sum(y_pred)
    [_, FP],[FN,_] = tm.confusion_est(y_test, y_pred)
    bias = (FP-FN)/len(y_test)
    mets = np.concatenate([[str(i)],[TP],[posest],[bias],mets])
    dfmet.loc[len(dfmet)] = mets

    mets = tm.metrics(y_test, y_predP)
    posest = np.sum(y_predP)
    [_,FP],[FN,_] = tm.confusion_est(y_test, y_predP)
    bias = (FP-FN)/len(y_test)
    mets = np.concatenate([[str(i) + 'P'],[TP],[posest],[bias],mets])
    dfmet.loc[len(dfmet)] = mets
    
    mets = tm.metrics(y_test, y_predC, threshold = algC.threshold)
    posest = np.sum(y_predC)
    [_, FP],[FN,_] = tm.confusion_est(y_test, y_predC)
    bias = (FP-FN)/len(y_test)
    mets = np.concatenate([[str(i)],[TP],[posest],[bias],mets])
    dfcal.loc[len(dfcal)] = mets

    mets = tm.metrics(y_test, y_predCP, threshold = algC.threshold)
    posest = np.sum(y_predCP)
    [_, FP],[FN,_] = tm.confusion_est(y_test,y_predCP)
    bias = (FP-FN)/len(y_test)
    mets = np.concatenate([[str(i) + 'P'],[TP],[posest],[bias],mets])
    dfcal.loc[len(dfcal)] = mets


print("Training and predicting: Done!")
print()
print("Processing metrics:")
total_pred = [y_pred, y_predP, y_predC, y_predCP]
j=0
for i in tqdm(total_pred, leave=False):
    i = i / iters
    mets = tm.metrics(y_test, i)
    posest = np.sum(i)
    [_,FP],[FN,_] = tm.confusion_est(y_test, i)
    bias = (FP-FN)/len(y_test)
    if j%2==0:
        mets = np.concatenate([["Average"],[TP],[posest],[bias],mets])
    else:
        mets = np.concatenate([["AverageP"],[TP],[posest],[bias],mets])
    if j<2:
        dfmet.loc[len(dfmet)] = mets
    else:
        dfcal.loc[len(dfcal)] = mets
    j += 1

print("Processing metrics: Done!")
print()
print("Storing metrics: ...")
dfmet.to_csv(outfile)
outfile = outfile[:-4] + ' (calibrated).csv'
dfcal.to_csv(outfile)
print("Storing metrics: Done!")
print()
print("Finished!")
print()
