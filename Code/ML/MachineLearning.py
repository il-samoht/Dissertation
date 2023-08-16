import pandas as pd
import numpy as np
import sklearn
import nltk
import contractions
import math
import sklearn.neighbors
import sklearn.metrics
import sklearn.linear_model
import sklearn.svm
import sklearn.naive_bayes
import sklearn.discriminant_analysis
import sklearn.tree
nltk.download('stopwords')

## Read Dataset
dataset = pd.read_csv("1Day_advanced.csv")
dataset.drop(columns=dataset.columns[0], axis=1, inplace=True)
news = dataset.iloc[:,1:26]



## Feature Extraction and Selection
#tokenization
print("Tokenization...")
tokenized_news = []
for index, rows in news.iterrows():
    rows.dropna(inplace=True) 
    #removes contractions eg. "I'd like to" -> "I would like to"
    rows = rows.apply(contractions.fix)
    #removes punctuations as well during tokenization
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokenized_news.append(rows.apply(tokenizer.tokenize))
news = pd.DataFrame(tokenized_news) 

#text cleaning
print("Text cleaning...")
stop_words = nltk.corpus.stopwords.words('english')
def textCleaning(sentence):
    try:
        #transform lowercase
        sentence = [x.lower() for x in sentence]
        #remove stopword
        sentence = [word for word in sentence if word not in stop_words]  
        #remove numbers
        sentence = [word for word in sentence if not word.isdigit()] 
    except:
        sentence = " "
    return sentence

headlines = []
for index, rows in news.iterrows():
    #tokenized version
    news.iloc[index] = rows.apply(textCleaning)
    #un-tokenized version
    headlines.append(' '.join((news.iloc[index].apply(' '.join)).tolist()))
news.to_csv("testFiltered.csv")
pd_headlines = pd.DataFrame(headlines)     
pd_headlines.to_csv("headlines.csv")

#split training data and testing data
splitIndex = math.floor(len(headlines) * 0.8)
train_data = headlines[:splitIndex]
train_labels = dataset.loc[:splitIndex-1, 'Label'].tolist()
test_data = headlines[splitIndex:len(headlines)]
test_labels = dataset.loc[splitIndex:len(headlines), 'Label'].tolist()

# Embedding
tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer='word', ngram_range=(3,3))
hash_vectorizer = sklearn.feature_extraction.text.HashingVectorizer(n_features=(2**18))
count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer='word', ngram_range=(3,3))


## ML
#train model
def train_model_old(vectorizer):
    print("Model training...")
    traindataset = vectorizer.fit_transform(train_data)
    
    print("knn training...")
    knn_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 10)
    knn_classifier.fit(traindataset, train_labels)

    print("svc training...")
    svc_classifier = sklearn.svm.SVC(kernel = 'linear')
    svc_classifier.fit(traindataset, train_labels)

    print("gaussian naive bayes training...")
    gnb_classifier = sklearn.naive_bayes.GaussianNB()
    gnb_classifier.fit(traindataset.toarray(), train_labels)

    print("logistic regression training...")
    lr_classifier = sklearn.linear_model.LogisticRegression()
    lr_classifier.fit(traindataset.toarray(), train_labels)

    print("linear discriminant analysis training...")
    lda_classifier = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda_classifier.fit(traindataset.toarray(), train_labels)
    
    print("decision tree classifier training...")
    dtc_classifier = sklearn.tree.DecisionTreeClassifier()
    dtc_classifier.fit(traindataset.toarray(), train_labels)


    #test results
    print("Model testing...")
    print("knn testing...")
    test_results = vectorizer.transform(test_data)
    knn_predictions = knn_classifier.predict(test_results)
    print("Score: ", sklearn.metrics.accuracy_score(test_labels, knn_predictions))
    print("Report: ", sklearn.metrics.classification_report(test_labels, knn_predictions))

    print("svc testing...")
    test_results = vectorizer.transform(test_data)
    svc_predictions = svc_classifier.predict(test_results)
    print("Score: ", sklearn.metrics.accuracy_score(test_labels, svc_predictions))
    print("Report: ", sklearn.metrics.classification_report(test_labels, svc_predictions))

    print("gaussian naive bayes testing...")
    test_results = vectorizer.transform(test_data)
    gnb_predictions = gnb_classifier.predict(test_results.toarray())
    print("Score: ", sklearn.metrics.accuracy_score(test_labels, gnb_predictions))
    print("Report: ", sklearn.metrics.classification_report(test_labels, gnb_predictions))
    
    print("logistic regression testing...")
    test_results = vectorizer.transform(test_data)
    lr_predictions = lr_classifier.predict(test_results.toarray())
    print("Score: ", sklearn.metrics.accuracy_score(test_labels, lr_predictions))
    print("Report: ", sklearn.metrics.classification_report(test_labels, lr_predictions))
    
    print("linear discriminant analysis testing...")
    test_results = vectorizer.transform(test_data)
    lda_predictions = lda_classifier.predict(test_results.toarray())
    print("Score: ", sklearn.metrics.accuracy_score(test_labels, lda_predictions))
    print("Report: ", sklearn.metrics.classification_report(test_labels, lda_predictions))
    
    print("decision tree classifier testing...")
    test_results = vectorizer.transform(test_data)
    dtc_predictions = dtc_classifier.predict(test_results.toarray())
    print("Score: ", sklearn.metrics.accuracy_score(test_labels, dtc_predictions))
    print("Report: ", sklearn.metrics.classification_report(test_labels, dtc_predictions))

# Output Scores
def evaluate_results(predictions):
    print("Score: ", sklearn.metrics.accuracy_score(test_labels, predictions))
    print("Report: ", sklearn.metrics.classification_report(test_labels, predictions))
    
# Model training and testing
def train_test_model(vectorizer):
    # preprocess traindataset so it can be used for our vectorizer
    traindataset = vectorizer.fit_transform(train_data)
    test_results = vectorizer.transform(test_data)
    
    # k-nearest neighbors training
    classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 10)
    classifier.fit(traindataset, train_labels)
    # k-nearest neighbors testing
    predictions = classifier.predict(test_results)
    evaluate_results(predictions)

    # support vector classifier training
    classifier = sklearn.svm.SVC(kernel = 'linear')
    classifier.fit(traindataset, train_labels)
    # support vector classifier testing
    predictions = classifier.predict(test_results)
    evaluate_results(predictions)

    # gaussian naive bayes training
    classifier = sklearn.naive_bayes.GaussianNB()
    classifier.fit(traindataset.toarray(), train_labels)
    # gaussian naive bayes testing
    predictions = classifier.predict(test_results.toarray())
    evaluate_results(predictions)

    # logistic regression training
    classifier = sklearn.linear_model.LogisticRegression()
    classifier.fit(traindataset.toarray(), train_labels)
    # logistic regression testing
    predictions = classifier.predict(test_results.toarray())
    evaluate_results(predictions)
    
    # decision tree classifier training
    classifier = sklearn.tree.DecisionTreeClassifier()
    classifier.fit(traindataset.toarray(), train_labels)
    # decision tree classifier testing
    predictions = classifier.predict(test_results.toarray())
    evaluate_results(predictions)
    

train_test_model(tfidf_vectorizer)
train_test_model(count_vectorizer)
train_test_model(hash_vectorizer)

