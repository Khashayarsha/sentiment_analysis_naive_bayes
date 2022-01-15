# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:14:59 2019

@author: XH
"""
import time
import numpy as np 
import pandas as pd 

#this code balances a dataset in the following way: 
#find score with minimum amount of reviews, sample that minimum amount of for every score, using a random_state = 1 (for reproducibility)
#puts them all together and saves it as a csv file in the directory wherefrom the code is ran.

#Enter a different .csv in the pd.read_csv(....) function if you want to balance other datasets. 





tic = time.time()
#[,reviewerID,asin,reviewerName,helpful,reviewText,overall(score),summary,unixReviewTime,reviewTime	]				
bigDataSet  =pd.read_csv('.\Project_random_200000_average_4.11.csv', delimiter = ',' )   #we use the biggeset dataset
balancedData = pd.DataFrame() #creates new empty DataFrame 
scores= range(1,6)     #possible review scores
#for i in range(1,6):
    #print('amount of data with score = ',i, '---> ',len(bigDataSet[bigDataSet.overall==i]))



dataPerScore = [len(bigDataSet[bigDataSet.overall ==score]) for score in scores] #array of amount of reviews per score
print('for pre-balance data: ', dataPerScore)

a = [0,0,0,0,0]       #placeholder array
for score in scores:
    #the following line takes a random sample of the size of the amount of reviews that are present the least in the dataset
    a[score-1] = bigDataSet[bigDataSet.overall ==score].sample(n=min(dataPerScore),random_state=1) 

balancedData = pd.concat(a)
#Checking if balancedData is balanced:
print('balancedData: ', balancedData)

    
balancedData = pd.concat(a)

#Checking if balancedData is balanced: if balanced, all array entries are of same length
check = [len(balancedData[balancedData.overall ==score]) for score in scores]

print('for balancedData: ', check, "all entries should be same if balanced")


balancedData.to_csv('balancedDataOut.csv')


toc = time.time()  # stop timing
elapsed = toc-tic
print("elapsed time:", "%.6f" % elapsed  )