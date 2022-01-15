# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:29:32 2019

@author: XH
"""

#df  =pd.read_csv('balancedData.csv', delimiter = ',' )

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
  
# Reads 'Youtube04-Eminem.csv' file  
df = pd.read_csv(r"train60_test15_only1-5_cleaned.csv", encoding ="latin-1") 
reviews = df['cleaned_review']
  
#comment_words = ' '
stopwords = set(STOPWORDS) 

#I get errors if I use more than 7000 reviews 
negativeReviewWords = df[df['overall'] == 1]['cleaned_review'].iloc[:7000].sum()
positiveReviewWords = df[df['overall'] == 5]['cleaned_review'].iloc[:7000].sum()
a = set(negativeReviewWords.split())
b = set(positiveReviewWords.split())

negs = ' '.join([word for word in negativeReviewWords.split() if word not in b])
poss = ' '.join([word for word in positiveReviewWords.split() if word not in a])#])

  
wordcloud = WordCloud(width = 800, height = 800, 
                 background_color ='white', 
                 stopwords = stopwords,
                min_font_size = 10).generate(poss) 
   
 # plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show() 
wordcloud.to_file('PositiveWords.png')

