#----------------------------- IMPORTS ---------------------------------#

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
# import BayesCCal as bc
from threading import Thread, Lock
from queue import Queue
from copy import copy
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import sys
sys.path.append('../')
import BayesCCal as bc

#----------------------------- CLASSES ---------------------------------#

class MeanEmbeddingVectorizer(object):
    """
    Class used for ???
    """
    def __init__(self, word2vec):
        """
        Initialization of the class and definition of the dimension of the class depends on the word2vec model).
        Input:
            word2vec: word2vec model
        """
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(word2vec))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        """
        Pass the data and labels to the word2vec model.
        Input:
            X: data
            y: labels
        """
        return self

    def transform(self, X):
        """
        Calculate the mean embeddings of all the words in X using the word2vec model.
        Input:
            X: data
        Output:
            return ...: mean embeddings
        """
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class Ensemble(object):
    """
    Class to form an ensemble of machine learning algorithms based on text data.
    Includes a fitting and prediction method.
    """

    def __init__(self, alg, size:int, bayescal: bool = False):
        """
        Initialisation function
        """

        self.alg = alg
        self.size = size
        self.bayescal = bayescal
    
    def fit(self, X,y, n_jobs = 2):
        """
        Method to fit algorithms in parallel.
        """
        q = Queue()
        lock = Lock()
        ens = {}
        features_list = {}
        vectorizers = {}
        cvs = {}
        clf_scores = {}
        for _ in range(n_jobs):
            worker = TrainWorker(q = q, lock = lock, alg = self.alg, 
                                X = X, y = y, ens = ens, 
                                features_list = features_list, 
                                vectorizers = vectorizers, cvs = cvs, 
                                clf_scores = clf_scores, bayescal = self.bayescal)
            worker.daemon = True
            worker.start()
        for i in range(self.size):
            q.put(i)
        q.join()

        #Sort the outputs and assign them as Ensemble attributes
        sort_i = np.argsort(np.array([*ens.keys()]))
        self.ens = np.array([*ens.values()])[sort_i]
        sort_i = np.argsort(np.array([*features_list.keys()]))
        self.features = np.array([*features_list.values()])[sort_i]
        sort_i = np.argsort(np.array([*vectorizers.keys()]))
        self.vectorizers = np.array([*vectorizers.values()])[sort_i]
        sort_i = np.argsort(np.array([*cvs.keys()]))
        self.cvs = np.array([*cvs.values()])[sort_i]
        sort_i = np.argsort(np.array([*clf_scores.keys()]))
        self.clf_scores = np.array([*clf_scores.values()])[sort_i]

    def predict(self, X, n_jobs = 2, indiv_preds = False, 
                total_preds = True, threshold = 0.5, 
                mindf:int=50,char:int=3, avg_size:int = 10):
        """
        Method to predict class labels in parallel.
        Averages prediction labels using weights based on the intersection of the feature space of a sample.
        """
        proba = False
        q = Queue()
        lock = Lock()
        y_pred = {}

        for _ in range(n_jobs):
            worker = PredictWorker(q, lock, X, y_pred, proba)
            worker.daemon = True
            worker.start()

        for i in range(self.size):
            i = (i, self.ens[i], self.vectorizers[i], self.cvs[i])
            q.put(i)
        q.join()

        sort_i = np.argsort(np.array([*y_pred.keys()]))
        y_pred = np.array([*y_pred.values()])[sort_i]
        if indiv_preds and not total_preds:
            return y_pred
        if total_preds:
            weights = np.zeros((len(X),self.size))
            for k in range(avg_size):
                test_tfidf, test_feats, _, _ = processing(X, mindf=1)
                test_feats = np.append(test_feats,'')
                indices = (test_tfidf!=0)*1
                for i in range(len(indices)):
                    index = np.unique(np.arange(len(indices[1]))*indices[i])
                    if index[0]==0 and indices[i][0]!=1:
                        index = index[1:]
                    feats = list(test_feats[index])
                    for j in range(self.size):
                        weights[i][j] += len(set(self.features[j]).intersection(set(feats)))/len(set(feats))
            weights = weights/avg_size
            weights = normalize(weights,'l1')
            weights = weights*self.clf_scores
            y_predT = np.sum(weights*(y_pred.T),axis=1)
            y_predT = (y_predT >= threshold)
        if not indiv_preds and total_preds:
            return y_predT
        if indiv_preds and total_preds:
            return y_predT, y_pred

    def predict_proba(self, X, n_jobs=2, indiv_preds = False,
                    total_preds = True, mindf:int=50, char:int=3, avg_size:int=10):
        """
        Method to predict class probabilities in parallel.
        Averages prediction probabilities using weights based on the intersection of the feature space of a sample.
        """
        
        proba = True
        q = Queue()
        lock = Lock()
        y_pred = {}

        for _ in range(n_jobs):
            worker = PredictWorker(q, lock, X, y_pred, proba)
            worker.daemon = True
            worker.start()
        for i in range(self.size):
            i = (i, self.ens[i], self.vectorizers[i], self.cvs[i])
            q.put(i)
        q.join()

        sort_i = np.argsort(np.array([*y_pred.keys()]))
        y_pred = np.array([*y_pred.values()])[sort_i]
        if indiv_preds and not total_preds:
            return y_pred
        if total_preds:
            weights = np.zeros((len(X),self.size))
            for k in range(avg_size):
                test_tfidf, test_feats, _, _ = processing(X, mindf=1)
                test_feats = np.append(test_feats,'')
                indices = (test_tfidf!=0)*1
                for i in range(len(indices)):
                    index = np.unique(np.arange(len(indices[1]))*indices[i])
                    if index[0]==0 and indices[i][0]!=1:
                        index = index[1:]
                    feats = list(test_feats[index])
                    for j in range(self.size):
                        weights[i][j] += len(set(self.features[j]).intersection(set(feats)))/len(set(feats))
            weights = weights/avg_size
            weights = normalize(weights,'l1')
            weights = weights*self.clf_scores
            y_predT = np.sum(weights*(y_pred.T),axis=1)
            if indiv_preds:
                return y_predT, y_pred
            else:
                return y_predT

class TrainWorker(Thread):
    """
    Class used for fitting algorithms in parallel
    """
    def __init__(self, q, lock, alg, X, y, 
                ens, features_list, vectorizers, cvs, clf_scores, 
                scorer = metrics.accuracy_score, bayescal:bool = False):
        """
        Initialisation function
        """
        Thread.__init__(self)
        self.q = q
        self.lock = lock
        self.alg = alg
        self.X = X
        self.y = y
        self.ens = ens
        self.features_list = features_list
        self.vectorizers = vectorizers
        self.cvs = cvs
        self.clf_scores = clf_scores
        self.scorer = scorer
        self.bayescal = bayescal

    def run(self):
        """ 
        This method gets called when an item is put in the queue and the Thread is started.
        """
        while True:
            tag = self.q.get()
            try:
                self.fit_clf(self.X, self.y, self.alg, tag,
                self.ens, self.features_list, self.vectorizers, 
                self.cvs, self.clf_scores, self.lock, self.scorer,
                self.bayescal)
            finally:
                self.q.task_done()
                break

    def fit_clf(self, X, y, alg, tag, 
                ens, features_list, vectorizers, cvs, clf_scores, 
                lock, scorer = metrics.accuracy_score, bayescal:bool = False):
        """
        Function that gets called during the run method. 
        Only is called whenever an item has been put into the queue.
        """
        temp_alg = copy(alg)
        X, X_val, y, y_val = train_test_split(X,y, test_size = 0.2)
        X, features, vectorizer, cv = processing(X)
        X_val = processing(X_val, tfidfvectorizer = vectorizer, cv=cv)

        if bayescal:
            temp_alg = apply_BayesCCal(temp_alg, X, y, density="test")
        else:
            temp_alg.fit(X,y)
        y_pred = temp_alg.predict(X_val)
        score = scorer(y_val, y_pred)

        lock.acquire()
        ens[tag] = temp_alg
        features_list[tag] = features
        vectorizers[tag] = vectorizer
        cvs[tag] = cv
        clf_scores[tag] = score
        lock.release()

class PredictWorker(Thread):
    """
    Class used to predict labels in parallel.
    """
    def __init__(self, q, lock, X, y_pred, proba):
        """
        Initialization function.
        """
        Thread.__init__(self)
        self.q = q
        self.lock = lock
        self.X = X
        self.y_pred = y_pred
        self.proba = proba
    
    def run(self):
        """
        This method gets called when an item is put in the queue and the Thread is started.
        """
        while True:
            tag, alg, vectorizer, cv = self.q.get()
            try:
                if self.proba:
                    self.predict_proba_clf(self.X, alg, tag, vectorizer, cv, self.y_pred, self.lock)
                else:
                    self.predict_clf(self.X, alg, tag, vectorizer, cv, self.y_pred, self.lock)
            finally:
                self.q.task_done()
                break
    
    def predict_clf(self, X, alg, tag, vectorizer, cv, y_pred, lock):
        """
        Function that gets called during the run method (when proba is False).
        Only is called whenver an item has been put into the queue.
        """
        X = processing(X, tfidfvectorizer = vectorizer, cv = cv)
        y_pred_temp = alg.predict(X)
        lock.acquire()
        y_pred[tag] = y_pred_temp
        lock.release()

    def predict_proba_clf(self, X, alg, tag, vectorizer, cv, y_pred, lock):
        """
        Function that gets called during the run method (when proba is True).
        Only is called whenever an item has been put into the queue.
        """
        X = processing(X, tfidfvectorizer = vectorizer, cv = cv)
        y_pred_temp = alg.predict_proba(X)[:,1]
        lock.acquire()
        y_pred[tag] = y_pred_temp
        lock.release()

class NeuralNetProba():
    """
    Adds the predict_proba method to binary classification Neural Networks.
    This neural network has to have an output layer with a dimension of 1.
    """
    def __init__(self, classifier):
        # classifier.__init__(self, classifier.inputs, classifier.outputs)
        self.classifier = classifier
    def predict(self,X):
        return self.classifier.predict(X)
    def predict_proba(self, X):
        preds = self.classifier.predict(X)
        return np.array([1-preds, preds]).T[0]
    def fit(self, X,y):
        return self.classifier.fit(X,y)

#-------------------------- Testing ----------------------------#

def genData(d_prime, N, ppos):
    """
    Function used to generate data (used for examples).

    Input:
        d_prime:    the average difference between the negative and positive class samples
        N:          number of samples to be generated
        ppos:       estimated number of positive samples to be generated

    Output:
        X:          set of normally distributed samples.
                    Class 0 samples are normally distributed as N(0,1)
                    Class 1 samples are normally distributed as N(d_prime,1)
        y:          set of random binary class labels
    """
    X = np.random.normal(0, 1, N)
    y = np.random.rand(N)<=ppos
    X[y] += d_prime
    X = X.reshape(-1,1)
    return X,y

def test_function():
    """
    Function for examples
    """
    X,y = genData(2,4000,0.5)
    clf = sk.linear_model.LogisticRegression(random_state=0, fit_intercept=True)
    clf.fit(X,y)
    cal = bc.calibrator_binary(clf)
    cal.fit(X,y)
    Xcal, _ = genData(2,1000,0.2)
    cal.determineThreshold(Xcal)
    X_test, y_test = genData(2,100,0.2)
    y_pred = clf.predict(X_test)
    print(metrics(y_test,y_pred))
    y_predP = clf.predict_proba(X_test)[:,1]
    print(metrics(y_test,y_predP>=0.5))
    y_pred = cal.predict(X_test)
    print(metrics(y_test,y_pred))
    y_predP = cal.predict_proba(X_test)[:,1]
    print(metrics(y_test,y_predP>=0.5))

#-------------------------- Utility ----------------------------#

def print_classifier(coef:np.array = None,
                     y_predP: np.array = None,
                     feature_names: np.array = None,
                     topnumcoefs: int = 20):
    """
    Function to present metrics of the classifier algorithm.

    Input:
        coef:   coefficients of the features of the classifier algorithm (optional)
        y_predP: prediction probabilities of the corresponding test set y_test
        feature_names:  names of the used features
        topnumcoefs:    number of coefficients that should be looked at

    Output:
        ---
    """

    if (type(y_predP) == type(None)) & (type(coef) == type(None)) & (type(feature_names) == type(None)):
        raise Exception('No input was given')
    
    if type(y_predP) != type(None):
        plt.hist(y_predP, color = 'blue', edgecolor = 'black', bins = 50)
        plt.show()
    
    if type(coef) != type(None):
        print("Number of features: " + str(len(coef)))
        print("")
        if type(feature_names) != type(None):
            top_pos_coef = np.argsort(coef)[-20:]
            top_neg_coef = np.argsort(coef)[:20]
            print("The top " + str(topnumcoefs) + " positive features:")
            print(np.array(feature_names)[top_pos_coef])
            print("")
            print("The top " + str(topnumcoefs) + " negative features:")
            print(np.array(feature_names)[top_neg_coef])

def file_importer(path: str, sep: str = ';'):
    """
    Function to import the stored y_test, y_pred and y_predP values from the files generated in the model_creation() function.
    
    Input:
        path:   path to the file to be imported
        sep:    seperator (defaults to ';')
    
    Output:
        y_test:  y_test as an np.array
        y_pred:  y_pred as an np.array
        y_predP: y_predP as an np.array
    """
    dataf = pd.read_csv(path, sep = ';')
    return np.array(dataf['y_test']), np.array(dataf['y_pred']),np.array(dataf['y_predP'])

def contains_string(X: list, text: str):
    """
    A function to find the strings in a list that contain a certain string.

    Input:
        X: list     list of strings
        text: str   string
    
    Output:
        y: list     list of strings
    """
    y = []
    for i in X:
        if text in i:
            y.append(i)
    return y

def print_metrics(y_true, y_pred, y_predP):
    """
    Function to print several metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - AUROC
    - Balanced Accuracy
    Plus:
    - the confusion matrix
    - the estimated confusion matrix
    - the ROC curve

    Input:
        y_true:     true labels of the samples
        y_pred:     predicted labels of the samples
        y_predP:    predicted probabilities of class 1 of the samples

    Output:
        ---
    """
    mets_ = ["Accuracy","Precision","Recall","F1 Score","AUROC","BA"]
    mets = metrics(y_true, y_pred)
    for i,j in zip(mets_,mets):
        i = i + ': '
        j = np.round(j,4)
        print('{:<15}{:<6}'.format(i,j))

    fig, ax = plt.subplots(1,3,figsize=(15,5))
    plt.figure()
    sk.metrics.ConfusionMatrixDisplay.from_predictions(y_true,y_pred, ax=ax[0], display_labels=["Non-Platform","Platform"])
    cm = confusion_est(y_true,y_predP, normalize=True)
    disp = sk.metrics.ConfusionMatrixDisplay(cm)
    disp.plot(values_format='',ax=ax[1])
    roc = sk.metrics.roc_curve(y_true,y_predP)
    ax[2].plot(roc[0],roc[1])
    ax[0].set_title('Confusion Matrix')
    ax[1].set_title('Est. Confusion Matrix')
    ax[2].set_title('ROC Curve')
    plt.show()

def metrics(y_true, y_pred, threshold=0.5):
    """
    Function to calculate several metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - AUROC
    - Balanced Accuracy
    - Matthews Correlation Coefficient (aka Phi coefficient)
    All metrics are calculated using their relative functions in sk.metrics

    Input:
        y_true:     true labels of the samples
        y_pred:     predicted labels of the samples
        y_predP:    predicted probabilities of class 1 of the samples

    Output:
        acc:        accuracy
        prec:       precision
        rec:        recall
        F1:         F1
        auroc:      AUROC
        BA:         balanced accuracy
    """
    sPCC = rho(y_true, y_pred)
    acc = sk.metrics.accuracy_score(y_true,y_pred>=threshold)
    # prec = sk.metrics.precision_score(y_true,y_pred>=threshold)
    # rec = sk.metrics.recall_score(y_true,y_pred>=threshold)
    # F1 = sk.metrics.f1_score(y_true,y_pred>=threshold)
    auroc = sk.metrics.roc_auc_score(y_true,y_pred)
    BA = sk.metrics.balanced_accuracy_score(y_true,y_pred>=threshold)
    phi = sk.metrics.matthews_corrcoef(y_true, y_pred>=threshold)
    return sPCC, acc, auroc, BA, phi

def opt_treshold(y_true, y_predP):
    """
    Function to determine the optimal threshold based of the estimated confusion matrix.
    Note that this function does optimizes the threshold naively and will (more often than not) result in a less than optimal threshold.

    Input:
        y_true:     true labels of the samples
        y_pred:     predicted labels of the samples

    Output:
        opt:        the threshold for which the mse is minimal
        acc:        the accuracy of the associated optimal threshold
    """
    [tn,fp,fn,tp] = np.zeros(4)
    for i,j in np.array([y_true,y_predP]).T:
        if i:
            tp += j
            fn += 1-j
        else:
            fp += j
            tn += 1-j
    cmP = np.array([[tn,fp],[fn,tp]])
    thresholds = np.linspace(0,1,100)
    mse = []
    acc = []
    for i in thresholds:
        y_pred = (y_predP > i)
        cm = sk.metrics.confusion_matrix(y_true,y_pred)
        mse.append(np.mean(np.power(cm-cmP,2)))
        acc.append(sk.metrics.accuracy_score(y_true,y_pred))
    opt = np.argmin(mse)/100
    return opt, acc

def rho(y_true, y_pred, OR="OvO", logit = False, array = False):
    if logit and list(np.unique(y_pred)) != [0,1]:
        y_pred = np.log(y_pred/(1-y_pred))
    try:
        dims = y_true.shape[1]
    except:
        N = np.sum(y_true)
        P = len(y_true) - N
        r = ((N*P)/((N+P)))**0.5
        muN = np.mean(y_pred[y_true==0])
        muP = np.mean(y_pred[y_true==1])
        std = np.sqrt(np.sum((y_pred-np.mean(y_pred))**2))
        if std == 0:
            return 0
        r *= (muP-muN)/std
        return r
    labels = np.argmax(y_true,axis=1)
    r = []
    for i in range(dims):
        if OR == "OvO":
            for j in range(dims):
                if i!=j:
                    Psamples = y_pred[labels==i][:,i]
                    Nsamples = y_pred[labels==j][:,i]
                    r.append(rho(np.concatenate([np.ones(len(Psamples)),np.zeros(len(Nsamples))]), np.concatenate([Psamples,Nsamples])))
        if OR == "OvR":
            temp_y_true = y_true[:,i]
            temp_y_pred = y_pred[:,i]
            r.append(rho(temp_y_true, temp_y_pred))
            # Psamples = y_pred[labels==i][i]
            # Nsamples = y_pred[labels!=i][i]
            # r.append(rho(np.concatenate([np.ones(len(Psamples)),np.zeros(len(Nsamples))]), np.concatenate([Psamples,Nsamples])))
    if array:
        return r
    r = np.tanh(np.mean(np.arctanh(r)))
    return r

#---------------------- Data Processing ------------------------#

def addlanguagefeature(X, dataset):
    """
    Function used to determine the language of a vector X. 
    Appends a 0 if the language is a 0 and a 1 otherwise.

    Input:
        X:          TF-IDF matrix
        dataset:    the original dataframe containing text

    Output:
        X:          Original vector X including a language feature
    """
    lang = dataset['lang'].tolist()
    language_vector = []
    for item in lang:
        if item=="dutch":
            language_vector.append(0)
        else:
            language_vector.append(1)
    language_vector = np.array(language_vector, ndmin=2, dtype="float64").T
    X = np.c_[X, language_vector]
    return(X)

def text_transformer(X):
    """
    Process the text for the training set. 
    Add an element from X['text'] if it is a string.
    Add a space otherwise.

    Input:
        X:      pandas.DataFrame containing a column 'text'

    Output:
        text:   text that is seperated by a space
    """
    text = []
    for file in X['text'].tolist():
        if isinstance(file,str):
            text.append(file.split())
        else:
            text.append(" ")
    return text
            
def processing(X, char:int=3, mindf:int=50, 
                tfidfvectorizer=None, cv=None):
    """
    Transforms the data from a DataFrame containing text to a tf-idf matrix and stores the features, vectorizer and countvectorizer,
    such that these can be used later on to transform other (test) data.

    Transforms a DataFrame containing text to a tf-idf matrix and returns this matrix in addition to: 
    - the observed features
    - the TF-IDF vectorizer
    - the count vectorizer

    These last two can be fed back to this function in order to apply the same transformation to a different DataFrame.

    Input:
        X:      pandas.DataFrame with 
        char:   the minimum number of characters strings should have for them to be included
        mindf:  minimum document frequency for generating the TF-IDF matrix
        tfidfvectorizer:    vectorizer used for generating the TF-IDF matrix
        cv:     countvectorizer for generating the TF matrix

    Output:
        X_3:    the TF-IDF matrix
        features:   the observed features
        tfidfvectorizer:    the TF-IDF vectorizer
        cv:     the count vectorizer
    """
    if type(tfidfvectorizer) == type(None) and type(cv) == type(None):  
        _ = text_transformer(X)
        cv = CountVectorizer(input='content', min_df=mindf,
                            token_pattern=u'\w{'+str(char)+',}')
        word_count_vector = cv.fit_transform(X['text'].tolist())
        tfidfvectorizer = TfidfTransformer(smooth_idf=True,use_idf=True,sublinear_tf=True)
        _ = tfidfvectorizer.fit(word_count_vector)
        features = cv.get_feature_names_out()
        X_2 = tfidfvectorizer.transform(word_count_vector)
        X_2 = np.c_[X_2.toarray()]
        X_3 = addlanguagefeature(X_2,X)
        return [X_3, features, tfidfvectorizer, cv]
    else:
        word_count_vector = cv.transform(X['text'].tolist())
        X_2 = tfidfvectorizer.transform(word_count_vector)
        X_2 = np.c_[X_2.toarray()]
        X_3 = addlanguagefeature(X_2,X)
        return X_3   

def strat_split(X,y, test_size = np.array([0.2,0.2])):
    """
    The function used to split different classes into different splits.
    For example: 
    if test_size = np.array([0.1,0.5]), 0.1 of class 0 is used for testing and 0.5 of class 1 is used for testing.
    
    Input:
        X:  dataset with samples
        y:  labels of the samples
        test_size:  the respective test sizes per class.
    
    Output:
        X_train:    train split of dataset X
        X_test:     test split of dataset X
        y_train:    labels of X_train
        y_test:     labels of X_test
    """

    #Add an output ratio: for example, output is 50/50 split
    #-> calculate the corresponding test_size ratios
    if len(test_size) == 1:
        test_size = np.repeat([test_size],2)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i,j in enumerate(test_size):
        X_temp = X[y==i]
        length = len(X_temp)
        y_temp = np.repeat(i, length)
        np.random.shuffle(X_temp)
        ind = int(length*j)

        try:
            X_test = np.concatenate((X_test, X_temp[:ind]))
        except:
            X_test = X_temp[:ind]
        y_test = np.concatenate((y_test, y_temp[:ind]))

        try:
            X_train = np.concatenate((X_train, X_temp[ind:]))
        except:
            X_train = X_temp[ind:]
        y_train = np.concatenate((y_train, y_temp[ind:]))
        
    ind = np.arange(len(X_test))
    np.random.shuffle(ind)
    X_test = X_test[ind]
    y_test = y_test[ind].astype(int)

    ind = np.arange(len(X_train))
    np.random.shuffle(ind)
    X_train = X_train[ind]
    y_train = y_train[ind].astype(int)

    return X_train, X_test, y_train, y_test

def preprocess(file):
    df = pd.read_csv(file, sep=";")
    df = df.fillna(" ")
    df = df[df['text'].str.split().apply(len)>=10]
    df['text'].str.findall('\w{3,}').str.join(' ')
    df['text'] = df['text'].str.replace("  "," ")

    return df

#--------------------------- Other ----------------------------#

def cv(clf, X_train, y_train, X_test, y_test, scoring,
        results: dict = {}, cval: int = 10, 
        cores: int = 2, save_files: bool = False,
        file_prefix: str = 'results', clf_alias: str = None, **kwargs):
    """
    Function used to perform cross validation and store predictions (labels and probabilities).

    Input:
        clf:        classifier
        X_train:    dataset used for training
        y_train:    labels of X_train
        X_test:     dataset used for testing
        y_test:     labels of X_test
        scoring:    function used for scoring (as per sklearn's scorers)
        results:    dictionary used for storing results
        cval:       number of folds used for cross validation
        cores:      number of workers used for cross validation
        save_files: boolean to determine whether or not to save predictions
        file_prefix:    string for naming the prediction files
        clf_alias:  string to give a different name to the classifier
    
    Output:
        ---
    """

    clf.fit(X_train, y_train)
    scores = cross_validate(clf, X_train, y_train, cv = cval, n_jobs =cores, scoring=scoring)
    score_names = contains_string(list(scores.keys()), 'test')
    for i in score_names:
        try:
            results[i].append(np.mean(scores[i]))
        except:
            results[i] = [np.mean(scores[i])]
    
    y_pred = clf.predict(X_test)
    y_predP = clf.predict_proba(X_test)

    if save_files:
        if clf_alias == None:
            fileRes = file_prefix + 'cv' + str(clf) + '.txt'
        else:  
            fileRes = file_prefix + 'cv' + clf_alias + '.txt'
        ## Create file
        fi = open(fileRes, "w")
        fi.write("y_test;y_pred;y_predP\n")
        for l in range(len(y_test)):
            fi.write(str(y_test[l]) + ";" + str(y_pred[l]) + ";" + str(y_predP[:,1][l]) + "\n")
        fi.close()
    
def benchmark_cv(clfs: list, clfs_aliases: list, dataset, scoring, BayesCal: bool = True, **kwargs):
    """
    Function to benchmark several classifiers using cross-validation.

    Input:
        clfs:   list of different sklearn classifiers
        clfs_aliases:   list of strings containing the aliases for the classifiers supplied in clfs
        dataset:    dataframe containing text
        scoring:    scoring function (as per sklearn's scorer functions)
        BayesCal:   boolean to indicate whether or not to also perform cross-validation using Bayes Calibration
    
    Output:
        results:    dictionary containing the classifier aliases and their corresponding scores from cross-validation
    """
    results = {}
    if BayesCal: 
        bayeslist = []
        temp_list = list(np.repeat(clfs_aliases, 2))
    else:
        temp_list = clfs_aliases
    results["Clf"] = temp_list
    del(temp_list)
    for i,j in np.vstack((clfs, clfs_aliases)).T:
        processing(dataset=dataset, **kwargs)
        [X_train, X_test, y_train, y_test] = processing.traintest
        cv(i, X_train, y_train, X_test, y_test, clf_alias=j, results=results, scoring=scoring, **kwargs)
        
        if BayesCal:
            bayeslist.extend(["No","Yes"])
            i = apply_BayesCCal(i, X_train, y_train)
            cv(i, X_train, y_train, X_test, y_test, clf_alias=str(j)+"Bayes", results=results, scoring=scoring, **kwargs)
    
    results["BayesCal"] =  bayeslist
    del(bayeslist)
    results = pd.DataFrame(results)
    cols = results.columns.tolist()
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]
    results = results[cols]
    return results

def apply_BayesCCal(alg, X_train, y_train, density="dens", NN=False):
    """
    Function for applying BayesCCal.

    Input: 
        alg:    classifier algorithm to which BayesCCal has to be applied
        X_train:    training dataset used to apply BayesCCal
        y_train:    training labels used to apply BayesCCal
        density:    passes the density argument to the calibrator_binary function from BayesCal
    
    Output:
        cal:    classifier algorithm callibrated according to BayesC and the provided dataset.
    """
    # Check the needed attributes (predict_proba and fit) -> This is done by the calibrator_binary class in __init__
    # bc.checkattr(alg)

    # Create a callibrated classifier
    cal = bc.calibrator_binary(alg, density = density, NN=NN)

    # Fit the callibrated classifier
    cal.fit(X_train, y_train)
    
    return cal

def confusion_est(y_true,y_predP, **kwargs):
    """
    Function to determine the estimated confusion matrix:
    Instead of containing the TP, FN, FP, TN, 
    the estimated confusion matrix contains estimates of these metrics.
    Such an estimate is calculated using the predicted probabilities of all the samples.

    Input:
        y_true:     the true labels
        y_predP:    predicted probabilities
    
    Output:
        cm:     estimated confusion matrix
    """


    [tn,fp,fn,tp] = np.zeros(4,dtype=int)
    for i,j in np.array([y_true,y_predP]).T:
        if i:
            tp += j
            fn += 1-j
        else:
            fp += j
            tn += 1-j
    cm = np.array([[tn,fp],[fn,tp]])
    if 'normalize' in kwargs:
        if kwargs['normalize']:
            N = len(y_true)
            # Np = np.sum(y_true)
            # Nn = N - Np
            cm[0] = cm[0]/N
            cm[1] = cm[1]/N
    if 'decimals' in kwargs:
        cm = np.round(cm,**kwargs)
    else:
        cm = np.round(cm,2)
    return cm
