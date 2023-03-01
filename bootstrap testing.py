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
outfile = '../Results/BootstrapResults.csv'
iters = 5

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
from Modules import thesis_module as tm
from tqdm import tqdm

dfmet = pd.DataFrame(columns = ["Iteration", "TP", "Pos. Est.","Bias", "sPCC", "Acc","AUROC","BA", "MCC"])
dfcal = copy(dfmet)

# frames = []
# for i in infiles:
#     df = pd.read_csv(i,sep=";")
#     df = df.fillna(" ")
#     df = df[df['text'].str.split().apply(len) >= 10]
#     frames.append(df)
# df3 = pd.concat(frames, sort=True)
# del(frames)

# char = 3
# if char == 3:
#     df3['text'].str.findall('\w{3,}').str.join(' ')

# df3['text'] = df3['text'].str.replace("  ", " ")

df_train = tm.preprocess(train)
df_test = tm.preprocess(test)

y_train = np.array([df_train['platform']])[0]
X_train = df_train.drop(['platform'], axis=1)
y_test = np.array([df_test['platform']])[0]
X_test = df_test.drop(['platform'], axis=1)

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


# Xpos = X[y==1]
# poslen = len(Xpos)
# ypos = np.ones(poslen)
# Xneg = X[y==0]
# neglen = len(Xneg)
# yneg = np.zeros(neglen)

# Xpos, Xpos_test, ypos, ypos_test = sklearn.model_selection.train_test_split(Xpos, ypos, test_size=0.2)
# poslen = len(Xpos)
# Xneg, Xneg_test, yneg, yneg_test = sklearn.model_selection.train_test_split(Xneg, yneg, test_size=0.2)
# neglen = len(Xneg)
# X_test = pd.concat([Xpos_test,Xneg_test])
# y_test = np.concatenate((ypos_test, yneg_test))
TP = np.sum(y_test)

y_pred = 0
y_predP = 0
y_predC = 0
y_predPC = 0

print("Training and predicting:")
for i in tqdm(range(iters), leave=False):
    alg = SVC(kernel = 'linear', probability=True)


    # Make a bootstrap sample (30/70 split)
    Xpos_train = np.random.randint(0,poslen, poslen)
    ypos_train = ypos[Xpos_train]
    Xpos_train = X_train_pos.iloc[Xpos_train]
    Xneg_train, _, yneg_train, _ = sklearn.model_selection.train_test_split(X_train_neg,yneg, test_size=0.2) #int(7*poslen/3))

    X_train = pd.concat([Xpos_train, Xneg_train])
    X_train, features, tfidfvectorizer, cv = tm.processing(X_train)
    y_train = np.concatenate((np.ones(poslen),np.zeros(len(Xneg_train))))
    X_test1 = tm.processing(X_test, tfidfvectorizer=tfidfvectorizer, cv=cv)

    alg.fit(X_train, y_train)
    pred = alg.predict(X_test1)
    try: 
        y_pred += pred
    except:
        y_pred = pred
    mets = tm.metrics(y_test, pred)
    posest = np.sum(pred)
    [_, FP],[FN,_] = tm.confusion_est(y_test, pred)
    bias = (FP-FN)/len(y_test)
    mets = np.concatenate([[str(i)],[TP],[posest],[bias],mets])
    dfmet.loc[len(dfmet)] = mets

    pred = alg.predict_proba(X_test1)[:,1]
    try:
        y_predP += pred
    except:
        y_predP = pred
    mets = tm.metrics(y_test, (pred))
    posest = np.sum(pred)
    [_,FP],[FN,_] = tm.confusion_est(y_test, pred)
    bias = (FP-FN)/len(y_test)
    mets = np.concatenate([[str(i) + 'P'],[TP],[posest],[bias],mets])
    dfmet.loc[len(dfmet)] = mets
    
    alg = tm.apply_BayesCCal(alg, X_train, y_train, density="test")
    pred = alg.predict(X_test1)
    try: 
        y_predC += pred
    except:
        y_predC = pred
    mets = tm.metrics(y_test, pred, threshold = alg.threshold)
    posest = np.sum(pred)
    [_, FP],[FN,_] = tm.confusion_est(y_test, pred)
    bias = (FP-FN)/len(y_test)
    mets = np.concatenate([[str(i)],[TP],[posest],[bias],mets])
    dfcal.loc[len(dfcal)] = mets
    pred = alg.predict_proba(X_test1)[:,1]
    try:
        y_predPC += pred
    except:
        y_predPC = pred
    mets = tm.metrics(y_test, pred, threshold = alg.threshold)
    posest = np.sum(pred)
    [_, FP],[FN,_] = tm.confusion_est(y_test,pred)
    bias = (FP-FN)/len(y_test)
    mets = np.concatenate([[str(i) + 'P'],[TP],[posest],[bias],mets])
    dfcal.loc[len(dfcal)] = mets


print("Training and predicting: Done!")
print()
print("Processing metrics:")
total_pred = [y_pred, y_predP, y_predC, y_predPC]
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