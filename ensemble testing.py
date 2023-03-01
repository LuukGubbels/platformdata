import sys, getopt
sys.path.append('../')
print()
argv = sys.argv[1:]

# infiles = ['../platformSample_data2.csv','../randomSample_purified_data2.csv']
train = 'platformSample_data2.csv'
test = 'platformSample_data2.csv'
outfile = '../Results/EnsembleResults.csv'
iters = 2
size = 2
n_jobs = size

try:
    opts, args = getopt.getopt(argv,
            "tr:te:o:n:s:j",
            ["train=","test=","ofile=","iters=","size=","jobs="])
except getopt.GetoptError:
    sys.exit(2)
if '?' in args or 'help' in args:
    print('Help for "ensemble testing.py"')
    print('This file is used to benchmark linear SVM Ensembles.')
    print()
    print('Options:')
    print('-tr,--train:  Defines the file from which training data should be read. Input as a .csv file with file extension')
    print('-te,--test:   Defines the file from which test data should be read. Input as a .csv file with file extension')
    print('-o,--ofile:   Defines the file in which results should be stored. Input with a file extension.')
    print('-n,--iters:   Defines the number of ensembles should be trained. Non-integer numbers will be rounded down.')
    print('-s,--size:    Defines the number of machines should be trained per ensemble. Non-integer numbers will be rounded down.')
    print('-j,--jobs:    Defines the number of jobs/workers that should be used for training and predicing. Non-integer numbers will be rounded down. This will be the size of the ensembles at most.')
    print()
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-tr","--train"):
        train = arg
    elif opt in ("-te","--test"):
        test = arg
    elif opt in ("-o","--ofile"):
        outfile = arg
    elif opt in ("-n","--iters"):
        iters = int(arg)
    elif opt in ("-s","--size"):
        size = int(arg)
    elif opt in ("-j","--jobs"):
        n_jobs = int(arg)

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
    pass
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

n_jobs = np.min((size,n_jobs)) #Remove unneccessary strain on CPU

dfens = pd.DataFrame(columns = ["Ensemble", "TP","Pos. Est.","Bias","Pos. Est. P","BiasP","sPCC","Acc","AUROC","BA","MCC"])
dfcal = copy(dfens)
alg = SVC(kernel='linear',probability=True)

### Loading Data ###
# frames = []
# for j in infiles:
#     df = pd.read_csv(j, sep=";")
#     df = df.fillna(" ")
#     df = df[df['text'].str.split().apply(len) >= 10]
#     frames.append(df)
# df3 = pd.concat(frames, sort=True)
# del(df)

# char = 3
# ##Check if only words with 3 or more characters should be included
# if char == 3:
#     df3['text'].str.findall('\w{3,}').str.join(' ')
        
# ##Check and remove double spaces from texts
# df3['text'] = df3['text'].str.replace("  ", " ")

# y = np.array(df3['platform'])
# X = df3.drop(['platform'], axis=1)

df_train = tm.preprocess(train)
df_test = tm.preprocess(test)

y_train = np.array([df_train['platform']])[0]
X_train = df_train.drop(['platform'], axis=1)
y_test = np.array([df_test['platform']])[0]
X_test = df_test.drop(['platform'], axis=1)

print("Training and predicting:")
for i in tqdm(range(iters), leave=False):
    ### Train Ensemble ###
    ensemble = tm.Ensemble(alg=alg, size = size)
    ensemble.fit(X_train,y_train,n_jobs=n_jobs)
    TP = np.sum(y_test)

    ### Prediction ###
    # Can be changed to not use batch prediction if needed
    # Batch prediction is not slower per say
    batch_size = 200
    splits = int(len(X_test)/batch_size + 1)

    y_pred = np.array([])
    y_predT = copy(y_pred)
    y_predP = copy(y_pred)
    y_predPT = copy(y_pred)
    # for m in range(splits):
    #     batch = X_test[m*batch_size:(m+1)*batch_size]
    #     if len(batch) != 0:
    #         predT, pred = ensemble.predict(batch, n_jobs=n_jobs, avg_size=10, indiv_preds=True, total_preds = True)
    #         y_predT = np.concatenate((y_predT, predT))
    #         try:
    #             y_pred = np.hstack((y_pred,pred))
    #         except:
    #             y_pred = pred
    #         predT, pred = ensemble.predict_proba(batch, 
    #                                 n_jobs=n_jobs, avg_size=10, indiv_preds=True)
    #         y_predPT = np.concatenate((y_predPT, predT))
    #         try:
    #             y_predP = np.hstack((y_predP, pred))
    #         except:
    #             y_predP = pred
    #     else:
    #         break

    y_predT, y_pred = ensemble.predict(X_test, n_jobs=n_jobs, avg_size=10, indiv_preds=True, total_preds = True)
    y_predPT, y_predP = ensemble.predict_proba(X_test, n_jobs=n_jobs, avg_size=10, indiv_preds=True, total_preds = True)
        
    mets = np.round(tm.metrics(y_test, y_predT),4)
    posest = np.sum(y_predT)
    [_,FP],[FN,_] = tm.confusion_est(y_test,y_predT)
    bias = (FP-FN)/len(y_test)
    posestP = np.sum(y_predPT)
    [_,FP],[FN,_] = tm.confusion_est(y_test,y_predPT)
    biasP = (FP-FN)/len(y_test)

    newrow = np.concatenate([[str(i)],[TP],[posest],[bias],[posestP],[biasP],mets])
    dfens.loc[len(dfens)] = newrow
    for number, j in enumerate(zip(y_pred,y_predP)):
        mets = tm.metrics(y_test, j[1])
        posest = np.sum(j[0])
        [_,FP],[FN,_] = tm.confusion_est(y_test, j[0])
        bias = (FP-FN)/len(y_test)
        posestP = np.sum(j[1])
        [_,FP],[FN,_] = tm.confusion_est(y_test, j[1])
        biasP = (FP-FN)/len(y_test)
        
        newrow = np.concatenate([[str(i)+"."+str(number+1)],[TP],[posest],[bias],[posestP],[biasP],mets])
        dfens.loc[len(dfens)] = newrow

    ### Train Calibrated Ensemble ###
    ensemble = tm.Ensemble(alg=alg, size=size, bayescal=True)
    ensemble.fit(X_train,y_train,n_jobs=n_jobs)

    y_pred = np.array([])
    y_predT = copy(y_pred)
    y_predP = copy(y_pred)
    y_predPT = copy(y_pred)
    # for m in range(splits):
    #     batch = X_test[m*batch_size:(m+1)*batch_size]
    #     if len(batch) != 0:
    #         predT, pred = ensemble.predict(batch, n_jobs=n_jobs, avg_size=10, indiv_preds=True)
    #         y_predT = np.concatenate((y_predT, predT))
    #         try:
    #             y_pred = np.hstack((y_pred,pred))
    #         except:
    #             y_pred = pred
    #         predT, pred = ensemble.predict_proba(batch, n_jobs=n_jobs, avg_size=10, indiv_preds=True)
    #         y_predPT = np.concatenate((y_predPT, predT))
    #         try:
    #             y_predP = np.hstack((y_predP,pred))
    #         except:
    #             y_predP = pred
    #     else:
    #         break
    y_predT, y_pred = ensemble.predict(X_test, n_jobs=n_jobs, avg_size=10, indiv_preds=True, total_preds = True)
    y_predPT, y_predP = ensemble.predict_proba(X_test, n_jobs=n_jobs, avg_size=10, indiv_preds=True, total_preds = True)

    mets = np.round(tm.metrics(y_test, y_predT),4)
    posest = np.sum(y_predT)
    [_,FP],[FN,_] = tm.confusion_est(y_test,y_predT)
    bias = (FP-FN)/len(y_test)
    posestP = np.sum(y_predPT)
    [_,FP],[FN,_] = tm.confusion_est(y_test,y_predPT)
    biasP = (FP-FN)/len(y_test)
    
    newrow = np.concatenate([[str(i)],[TP],[posest],[bias],[posestP],[biasP],mets])
    dfcal.loc[len(dfcal)] = newrow
    for number, j in enumerate(zip(y_pred,y_predP)):
        mets = tm.metrics(y_test, j[1])
        posest = np.sum(j[0])
        [_,FP],[FN,_] = tm.confusion_est(y_test, j[0])
        bias = (FP-FN)/len(y_test)
        posestP = np.sum(j[1])
        [_,FP],[FN,_] = tm.confusion_est(y_test, j[1])
        biasP = (FP-FN)/len(y_test)

        newrow = np.concatenate([[str(i)+"."+str(number+1)],[TP],[posest],[bias],[posestP],[biasP],mets])
        dfcal.loc[len(dfcal)] = newrow

print("Training and predicting: Done!")
print()
print("Processing ensemble metrics:")
avg_bias, avg_biasP, avg_sPCC, avg_acc, avg_auroc, avg_ba, avg_phi = np.zeros(7)
avg_bias_cal, avg_biasP_cal, avg_sPCC_cal, avg_acc_cal, avg_auroc_cal, avg_ba_cal, avg_phi_cal = np.zeros(7)
for i in tqdm(range(iters), leave=False):
    avg_bias = avg_bias + float(dfens['Bias'][i*(1+size)])
    avg_biasP = avg_biasP + float(dfens['BiasP'][i*(1+size)])
    avg_sPCC = avg_sPCC + float(dfens['sPCC'][i*(1+size)])
    avg_acc = avg_acc + float(dfens['Acc'][i*(1+size)])
    # avg_prec = avg_prec + float(dfens['Prec'][i*(1+size)])
    # avg_rec = avg_rec + float(dfens['Rec'][i*(1+size)])
    # avg_f1 = avg_f1 + float(dfens['F1'][i*(1+size)])
    avg_auroc = avg_auroc + float(dfens['AUROC'][i*(1+size)])
    avg_ba = avg_ba + float(dfens['BA'][i*(1+size)])
    avg_phi = avg_phi + float(dfens['MCC'][i*(1+size)])

    avg_bias_cal += float(dfcal['Bias'][i*(1+size)])
    avg_biasP_cal += float(dfcal['BiasP'][i*(1+size)])
    avg_sPCC_cal += float(dfcal['sPCC'][i*(1+size)])
    avg_acc_cal += float(dfcal['Acc'][i*(1+size)])
    # avg_prec_cal += float(dfcal['Prec'][i*(1+size)])
    # avg_rec_cal += float(dfcal['Rec'][i*(1+size)])
    # avg_f1_cal += float(dfcal['F1'][i*(1+size)])
    avg_auroc_cal += float(dfcal['AUROC'][i*(1+size)])
    avg_ba_cal += float(dfcal['BA'][i*(1+size)])
    avg_phi_cal += float(dfcal['MCC'][i*(1+size)])

avg_bias = avg_bias/iters
avg_biasP = avg_biasP/iters
avg_sPCC = avg_sPCC/iters
avg_acc = avg_acc/iters
# avg_prec = avg_prec/iters
# avg_rec = avg_rec/iters
# avg_f1 = avg_f1/iters
avg_auroc = avg_auroc/iters
avg_ba = avg_ba/iters
avg_phi = avg_phi/iters
newrow = ['Ensemble Average','-','-',avg_bias,'-',avg_biasP, avg_sPCC, avg_acc, avg_auroc, avg_ba, avg_phi]
dfens.loc[len(dfens)] = newrow

avg_bias_cal = avg_bias_cal/iters
avg_biasP_cal = avg_biasP_cal/iters
avg_sPCC_cal = avg_sPCC_cal/iters
avg_acc_cal = avg_acc_cal/iters
# avg_prec_cal = avg_prec_cal/iters
# avg_rec_cal = avg_rec_cal/iters
# avg_f1_cal = avg_f1_cal/iters
avg_auroc_cal = avg_auroc_cal/iters
avg_ba_cal = avg_ba_cal/iters
avg_phi_cal = avg_phi_cal/iters
newrow = ['Ensemble Average','-','-',avg_bias_cal,'-',avg_biasP_cal, avg_sPCC_cal, avg_acc_cal, avg_auroc_cal, avg_ba_cal, avg_phi_cal]
dfcal.loc[len(dfcal)] = newrow

print("Processing ensemble metrics: Done!")
print()
print("Processing individual metrics:")
avg_bias, avg_biasP, avg_sPCC, avg_acc, avg_auroc, avg_ba, avg_phi = np.zeros(7)
avg_bias_cal, avg_biasP_cal, avg_sPCC, avg_acc_cal, avg_auroc_cal, avg_ba_cal, avg_phi_cal = np.zeros(7)
for i in tqdm(range(iters), leave=False):
    avg_bias += np.sum(np.array(dfens['Bias'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_biasP += np.sum(np.array(dfens['BiasP'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_sPCC += np.sum(np.array(dfens['sPCC'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_acc += np.sum(np.array(dfens['Acc'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    # avg_prec += np.sum(np.array(dfens['Prec'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    # avg_rec += np.sum(np.array(dfens['Rec'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    # avg_f1 += np.sum(np.array(dfens['F1'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_auroc += np.sum(np.array(dfens['AUROC'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_ba += np.sum(np.array(dfens['BA'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_phi += np.sum(np.array(dfens['Phi'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))

    avg_bias_cal += np.sum(np.array(dfcal['Bias'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_biasP_cal += np.sum(np.array(dfcal['BiasP'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_sPCC_cal += np.sum(np.array(dfcal['sPCC'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_acc_cal += np.sum(np.array(dfcal['Acc'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    # avg_prec_cal += np.sum(np.array(dfcal['Prec'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    # avg_rec_cal += np.sum(np.array(dfcal['Rec'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    # avg_f1_cal += np.sum(np.array(dfcal['F1'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_auroc_cal += np.sum(np.array(dfcal['AUROC'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_ba_cal += np.sum(np.array(dfcal['BA'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))
    avg_phi_cal += np.sum(np.array(dfcal['Phi'][i*(1+size)+1:(i+1)*(1+size)],dtype=float))

avg_bias = avg_bias/(iters*size)
avg_biasP = avg_biasP/(iters*size)
avg_sPCC = avg_sPCC/(iters*size)
avg_acc = avg_acc/(iters*size)
# avg_prec = avg_prec/(iters*size)
# avg_rec = avg_rec/(iters*size)
# avg_f1 = avg_f1/(iters*size)
avg_auroc = avg_auroc/(iters*size)
avg_ba = avg_ba/(iters*size)
avg_phi = avg_phi/(iters*size)
newrow = ['Individual Average','-','-',avg_bias,'-',avg_biasP, avg_sPCC, avg_acc, avg_auroc, avg_ba, avg_phi]
dfens.loc[len(dfens)] = newrow

avg_bias_cal = avg_bias_cal/(iters*size)
avg_biasP_cal = avg_biasP_cal/(iters*size)
avg_sPCC_cal = avg_sPCC_cal/(iters*size)
avg_acc_cal = avg_acc_cal/(iters*size)
# avg_prec_cal = avg_prec_cal/(iters*size)
# avg_rec_cal = avg_rec_cal/(iters*size)
# avg_f1_cal = avg_f1_cal/(iters*size)
avg_auroc_cal = avg_auroc_cal/(iters*size)
avg_ba_cal = avg_ba_cal/(iters*size)
avg_phi_cal = avg_phi_cal/(iters*size)
newrow = ['Individual Average','-','-',avg_bias,'-',avg_biasP_cal,avg_sPCC_cal, avg_acc_cal, avg_auroc_cal, avg_ba_cal, avg_phi_cal]
dfcal.loc[len(dfcal)] = newrow

print("Processing individual metrics: Done!")
print()
print("Storing metrics: ...")
dfens.to_csv(outfile)
outfile = outfile[:-4] + ' (calibrated).csv'
dfcal.to_csv(outfile)
print("Storing metrics: Done!")
print()
print("Finished!")
print()
