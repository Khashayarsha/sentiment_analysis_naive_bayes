# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:44:44 2019

@author: XH
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:21:01 2019

@author: XH
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:12:18 2019

@author: XH
"""
import numpy as np 
import pandas as pd 

import time
#implements a Bernoulli (not multinomial) Naive Bayes Classifier.
#Maybe implement TF-IDF? 
#Implement bigrams
#https://stats.stackexchange.com/questions/57337/naive-bayes-feature-probabilities-should-i-double-count-words
#data = [,reviewerID,asin,reviewerName,helpful,reviewText,overall(score),summary,unixReviewTime,reviewTime, 'stemmed' 	]

#data  =pd.read_csv('stemmedData10k.csv', delimiter = ',' )
dataset = pd.read_csv('train60_test15_only1-5_cleaned.csv', delimiter = ',') 


reviews_test_clean = []  
reviews_train_clean = []

datatrain = 14504

datatest = 3626

for i in range(0, datatrain):  
    
    review = str(dataset['cleaned_review'][i])
    # append each string to create 
    # array of clean text  
    reviews_train_clean.append(review) 


#print("reviews_train_clean :", reviews_train_clean)

for i in range(datatrain, datatrain+datatest):  
    
    review = str(dataset['cleaned_review'][i])
    # append each string to create 
    # array of clean text  
    reviews_test_clean.append(review) 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer 

cv = CountVectorizer()
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)
#Predict Output

#print( "Predicted Value:", y_predicted)

#assess prediction
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test, y_predicted) 

#correctly predicted
from sklearn.metrics import classification_report
print('confusion_matrix:  ')  
#print(cm)
print('sum of trace/total: ', cm.trace()/cm.sum())
target_names = ['score 1', 'score 2',  'score 3', 'score 4', 'score 5']
#print(classification_report(y_test, y_predicted, target_names=target_names))
from sklearn import metrics
acc = metrics.accuracy_score(y_test, y_predicted)
print('accuracy score: ', acc)




