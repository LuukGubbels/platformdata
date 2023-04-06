import sys, getopt

print()
argv = sys.argv[1:]

infiles = []

try:
    opts, args = getopt.getopt(argv,
            "P:N:n:O:o",
            ["pos=","trainneg=","testneg=","otrain=","otest="])
except getopt.GetoptError:
    sys.exit(2)
if '?' in args or 'help' in args:
    print('Help for traintestsplit.py')
    print('this file is used to form a train and test set from separate files.')
    print()
    print('Options:')
    print('-P, --pos:       Defines the files containing positives. Input as a python list as a string with .csv files with extension. No default.')
    print('-N, --trainneg:   Defines the files containing negatives needed for training. Input as a python list as a string with .csv files with extension. No default.')
    print('-n, --testneg:    Defines the files containing negatives needed for testing. Input as a python list as a string with .csv files with extension. No default.')
    print("-O, --otrain:     Defines the file in which the training set should be saved. Input with a file extension. No default.")
    print("-o, --otest:      Defines the file in which the testing set should be saved. Input with a file extension. No default.")

    print()
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-P","--pos"):
        pos = arg.strip('][').split(',')
    elif opt in ("-N","--trainneg"):
        trainneg = arg.strip('][').split(',')
    elif opt in ("-n","--testneg"):
        testneg = arg.strip('][').split(',')
    elif opt in ("-O","--otrain"):
        otrain = arg
    elif opt in ("-o","--otest"):
        otest = arg

try:
    fo = open(otrain, "wb")
    fo.close()
    fo = open(otest, "wb")
    fo.close()
except:
    print("Please close the outgoing file(s)!")
    print()
    sys.exit()

import pandas as pd

# Read and concatenate all positives. Split them 50/50 into a training and test set
frames = []
for i in pos:
    df = pd.read_csv(i, sep=";")
    frames.append(df)
X_pos = pd.concat(frames, sort=True)
del(frames)
npos = int(len(X_pos))
X_pos = X_pos.sample(frac=1)
X_train_pos = X_pos.head(npos)
X_test_pos = X_pos.tail(npos)

# Read and concatenate the training negatives and store them
frames = []
for i in trainneg:
    df = pd.read_csv(i, sep=";")
    frames.append(df)
X_train_neg = pd.concat(frames, sort=True)
del(frames)
nneg = int(7*npos/3) # create approx. 30/70 split
X_train_neg = X_train_neg.sample(n=nneg)

X_train = pd.concat([X_train_pos, X_train_neg])
X_train.to_csv(otrain)

# Read and concatenate the testing negatives and store them
frames = []
for i in testneg:
    df = pd.read_csv(i, sep=";")
    frames.append(df)
X_test_neg = pd.concat(frames, sort=True)
del(frames)
X_test = pd.concat([X_test_pos, X_test_neg])
X_test.to_csv(otest)
