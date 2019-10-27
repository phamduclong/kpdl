#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
reviews_df = pd.read_csv("data.csv")
contents = reviews_df['content']

def clean_text(text):
    # chuyển thành chữ thường
    text = text.lower()
    # chuyen thanh token
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # xoa từ chứa số
    text = [word for word in text if not any(c.isdigit() for c in word)]
    text = " ".join(text)
    return(text)
reviews_df["content_clean"] = reviews_df["content"].apply(lambda x: clean_text(x))
# # danh gia tinh cam
# sid = SentimentIntensityAnalyzer()
# reviews_df["sentiments"] = reviews_df["review"].apply(lambda x: sid.polarity_scores(x))
# reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)

# do dai cua cau
reviews_df["nb_chars"] = reviews_df["content"].apply(lambda x: len(x))
# so tu
reviews_df["nb_words"] = reviews_df["content"].apply(lambda x: len(x.split(" ")))
# chuyen chu thanh vector
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews_df["content_clean"].apply(lambda x: x.split(" ")))]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

doc2vec_df = reviews_df["content_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(reviews_df["content_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews_df.index
reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)

print (reviews_df["label"].value_counts(normalize = True))
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def show_wordcloud(data, title=None):
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=42
    ).generate(str(data))

    fig = plt.figure(1, figsize=(20, 20))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# print wordcloud
show_wordcloud(reviews_df["content_clean"])
test = reviews_df[reviews_df["nb_words"] >= 5]
print (test)
import seaborn as sns

for x in ["__label__xuat_sac","__label__tot","__label__trung_binh","__label__kem","__label__rat_kem"]:
    subset = reviews_df[reviews_df['label'] == x]

    # Draw the density plot
    if x == "__label__xuat_sac":
        label = "xuat xac"
    elif x== "__label__tot":
        label = "tot"
    elif x =="__label__trung_binh":
        label ="trung binh"
    elif x =="__label__kem":
        label ="kem"
    else:
        label="rat kem"
    print (subset)
    #sns.distplot(subset['compound'], hist=False, label=label)
# feature selection
label = "label"
ignore_cols = [label, "content", "content_clean"]
features = [c for c in reviews_df.columns if c not in ignore_cols]
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(reviews_df[features], reviews_df[label], test_size = 0.20, random_state = 42)
# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)

# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_df.head(20)
print (feature_importances_df)