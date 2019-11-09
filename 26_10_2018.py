import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re


import gensim
from gensim import corpora
import string
# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import FreqDist
reviews_df = pd.read_csv("train.csv")
def clean_text(text):
    # chuyển thành chữ thường
    text = text.lower()
    # chuyen thanh token
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # xoa từ chứa số
    text = [word for word in text if not any(c.isdigit() for c in word)]
    text = " ".join(text)
    return(text)

def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms)
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()
reviews_df["content_clean"] = reviews_df["content"].apply(lambda x: clean_text(x))
# reviews_df["content_clean"] = reviews_df["content_clean"].str.replace("[^a-zA-Z#]", " ")
# freq_words(reviews_df["content_clean"])
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

test_reviews_df = pd.read_csv("test.csv")
test_reviews_df["content_clean"] = reviews_df["content"].apply(lambda x: clean_text(x))
train_vectors = vectorizer.fit_transform(reviews_df["content_clean"])
test_vectors = vectorizer.transform(test_reviews_df["content_clean"])
import time
from sklearn import svm
from sklearn.metrics import classification_report
# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
print("start training")
t0 = time.time()
classifier_linear.fit(train_vectors,reviews_df["label"])
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(test_reviews_df["label"], prediction_linear, output_dict=True)
# import pickle
import pickle
#pickling the vectorizer
pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))
# pickling the model
pickle.dump(classifier_linear, open('classifier.sav', 'wb'))
def test():
    test_reviews_df = pd.read_csv("test.csv")
    vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
    classifier = pickle.load(open('classifier.sav', 'rb'))
    test_reviews_df["content_clean"] = test_reviews_df["content"].apply(lambda x: clean_text(x))
    text_vector = vectorizer.transform(test_reviews_df['content_clean'])
    result = classifier.predict(text_vector)
    return result
def result():
    test_data={}
    test_reviews_df = pd.read_csv("test.csv")
    test_reviews_df['train']  = test()
    test_reviews_df['True or False']=test_reviews_df['train']==test_reviews_df['label']
    test_reviews_df.pop("content")
    export_csv = test_reviews_df.to_csv('result.csv')
    right=0
    for x in test_reviews_df['True or False']:
      if x==True:
        right=right+1
    divisions = ['Đúng']
    print(right/len( test_reviews_df['True or False']))
    divisions_marks = [  right/len( test_reviews_df['True or False'])]
    plt.bar(divisions,divisions_marks,color="green")
    plt.title("%")
    plt.xlabel('label')
    plt.ylabel('so luong ban gi')
    plt.show()

result()