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
data  =pd.read_csv('stemmedData10k.csv', delimiter = ',' )


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
accScores = []
#for i in range(1000, 7000, 500):
cv = CountVectorizer(max_features = 3000, binary = False)#, ngram_range=(1, 2))
#cv = TfidfVectorizer(max_features = 3000)
print('amount of features used : ', 3000)

X = cv.fit_transform(data['stemmed']).toarray()

y = data.iloc[:,7]#.values #scores

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)

from sklearn.naive_bayes import MultinomialNB


model = MultinomialNB()

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)
#Predict Output

#print( "Predicted Value:", y_predicted)

#assess prediction
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score
print('------------------starting cross validation:-----------')
scores = cross_val_score(model, X_train, y_train, cv=10)
                                            

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('------------------cross validation has ended ------------')
cm = confusion_matrix(y_test, y_predicted) 

#correctly predicted
from sklearn.metrics import classification_report
print('confusion_matrix:  ')  
#print(cm)
print('sum of trace/total: ', cm.trace()/cm.sum())
target_names = ['score 1', 'score 2',  'score 3', 'score 4', 'score 5']
print(classification_report(y_test, y_predicted, target_names=target_names))
from sklearn import metrics
acc = metrics.accuracy_score(y_test, y_predicted)
print('accuracy score: ', acc)







    
