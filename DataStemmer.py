import numpy as np 
import pandas as pd 
import nltk
import time
import pandas




#
#data = [,reviewerID,asin,reviewerName,helpful,reviewText,overall(score),summary,unixReviewTime,reviewTime, 'stemmed' 	]
#last column contains the stemmed reviewText								



tic = time.time()
data  =pd.read_csv('balancedData.csv', delimiter = ',' )
#n = 3000
#data = data.iloc[:n]
#print(data)
data['stemmed'] = ''
print('length of stemmedReviews = ', len(data['stemmed']))
data['reviewText'] =data['reviewText'].apply(str)

import re 

#nltk.download('stopwords')
#to remove stopwords (a, the... etc)

from nltk.corpus import stopwords
#for stemming purposes
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

    
def stem_sentences(sentence):
    
    global count
    count+=1
    print(count)
    print('type of sentence: ', type(sentence))
    if type(sentence)==str:
       
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        sentence = sentence.lower()
        tokens = sentence.split()
        stemmed_tokens = [ps.stem(token)  for token in tokens if not token in set(stopwords.words('english'))]
        
        return ' '.join(stemmed_tokens)
    else:
        
        return ' '.join(stemmed_tokens)


    
def main():
    global count 
    count = 0     
    data['stemmed'] =  data['reviewText'].apply(stem_sentences)

    data.to_csv('SmallBalancedStemmed.csv')
    print('done')

if __name__ == '__main__':
    main()
    
# =============================================================================
# 
# for j in range(1,10):
#     print('busy stemming documents: ', j)
#     data['stemmed'].iloc[(j-1)*6043:j*6043] = data['reviewText'].apply(stem_sentences)
#     name = 'stemmedBalancedData'+str(j)+'.csv'
#     data.to_csv(name)
# 
# #exports the dataset with stemmed sentences in the final column data['stemmed'] 
# =============================================================================








#following code is replaced with def stem_sentences(sentence)
# =============================================================================
# for i in range(0, 10):#len(data)):  
#     #print(i)
#     # column : "Review", row ith 
#     review = re.sub('[^a-zA-Z]', ' ', data['reviewText'][i])  
#       
#     # convert all cases to lower cases 
#     review = review.lower()  
#       
#     # split to array(default delimiter is " ") 
#     review = review.split()  
#       
#     # creating PorterStemmer object to 
#     # take main stem of each word 
#     ps = PorterStemmer()  
#       
#     # loop for stemming each word 
#     # in string array at ith row     
#     review = [ps.stem(word) for word in review 
#                 if not word in set(stopwords.words('english'))]  
#                   
#     # rejoin all string array elements 
#     # to create back into a string 
#     review = ' '.join(review)   
#       
#     # append each string to create 
#     # array of clean text  
#     reviews.append(review)  
# =============================================================================


    
toc = time.time()
elapsed = toc-tic
print('elepased time: ', elapsed)
