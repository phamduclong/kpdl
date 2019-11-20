import pandas as pd

pd.set_option("display.max_colwidth", 200)
import numpy as np
import string
import gensim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from pickle import dumps, load, dump
from pyvi import ViTokenizer, ViPosTagger
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
from gensim.parsing.preprocessing import preprocess_string
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from sklearn import utils
from gensim.models import Word2Vec
from underthesea import sentiment


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = ViTokenizer
        self.pos_tagger = ViPosTagger

    def fit(self, *_):
        return self

    def transform(self, X, y=None, **fit_params):
        result = X.apply(lambda text: self.tokenizer.tokenize(text))
        return result


class Doc2VecTransformer(BaseEstimator):
    def __init__(self, vector_size=100, learning_rate=0.02, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = 4

    def fit(self, df_x, df_y=None):
        tagged_x = [TaggedDocument(doc, [i]) for i, doc in enumerate(df_x.apply(lambda x: x.split(" ")))]

        # tagged_x = [TaggedDocument(preprocess_string(row['content_clean']), [index]) for index, row in df_x.iterrows()]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers, window=2, min_count=1)
        for epoch in range(self.epochs):
            model.train(utils.shuffle([x for x in tqdm(tagged_x)]), total_examples=len(tagged_x), epochs=1)
        model.alpha -= self.learning_rate
        model.min_alpha = model.alpha
        self._model = model
        return self

    def transform(self, df_x):
        return np.asmatrix(
            np.array([self._model.infer_vector(doc) for i, doc in enumerate(df_x.apply(lambda x: x.split(" ")))]))


def cleant_data(data):
    def clean_text(text):
        # chuyển thành chữ thường
        text = text.lower()
        # chuyen thanh token
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        # xoa từ chứa số
        text = [word for word in text if not any(c.isdigit() for c in word)]
        # remove stop words
        stop = ['khách', 'sạn', 'phòng', 'có', 'và', 'sẽ', 'ăn', 'là', 'giá', 'biển']
        text = [x for x in text if x not in stop]
        right = ['tốt', 'xuất', 'sắc', 'tuyệt', 'vời', 'dễ', 'chịu', 'de', 'chiu', 'hao', 'hảo', 'ok', ]
        right_2 = ['dễ', 'tạm', 'không', 'vời', 'tàm', 'được', 'đẹp', 'sạch', 'sach']
        right_3 = ['kém', 'tuy', 'do', 'kem', 'thất', 'vọng', ]

        def len_text(x):
            l = ''
            if x in right:
                l = '4'
            if x in right_2:
                l = '2'
            if x in right_3:
                l = '-1'
            return l

        text_2 = [str(len_text(x)) for x in text]

        text_3 = len(text)

        text = " ".join(text)
        text_2 = ''.join(text_2)
        text = text + ' ' + text_2 + ' ' + str(text_3)

        return (text)

    data["content_clean"] = data["content"].apply(lambda x: clean_text(x))
    return data


from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def build_model_forest(X_train, y_train, y_test, X_test, full_test):
    print("start traning")
    text_classifier = RandomForestClassifier
    text_classifier.fit(X_train, y_train)
    predictions = text_classifier.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))
    predictions = text_classifier.predict(full_test)
    print(accuracy_score(y_test + y_train, predictions))
    save_model(TfidfVectorizer(), text_classifier)


from nltk import FreqDist
import seaborn as sns


def get_top_word(x):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=20)
    plt.figure(figsize=(20, 5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel='Count')
    plt.show()


def save_model(vectorizer, classifier_linear):
    # pickling the vectorizer
    object_file = dump(vectorizer, open('vectorizer.sav', 'wb'))
    # pickling the model
    object_file = dump(classifier_linear, open('classifier.sav', 'wb'))


def test_model(data):
    vectorizer = load(open('vectorizer.sav', 'rb'))
    classifier = load(open('classifier.sav', 'rb'))

    # X_train, X_test, y_train, y_test = train_test_split(reviews_train['content_clean'], reviews_train['label'],
    #                                                     test_size=0,
    #                                                     random_state=0)
    result = classifier.predict(data['content_clean'])
    print(len(result))
    return result


# thêm trong số
def add_colum(data):
    # add number of characters column
    data["nb_chars"] = data["content"].apply(lambda x: len(x))

    # add number of words column
    data["nb_words"] = data["content"].apply(lambda x: len(x.split(" ")))
    return data


# tính tỉ lệ trọng số
def get_weight(X_train, y_train, y_test, X_test):
    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, y_train)
    return text_classifier


def visualization():
    test_data = {}

    reviews_train = pd.read_csv("test_p.csv")
    reviews_train = cleant_data(reviews_train)
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()

    reviews_train["sentiments"] = reviews_train["content"].apply(lambda x: sid.polarity_scores(x))
    reviews_train = pd.concat(
        [reviews_train.drop(['sentiments'], axis=1), reviews_train['sentiments'].apply(pd.Series)], axis=1)
    for col in reviews_train.columns:
        print(col)

    # print(reviews_train["content_clean"])
    def comp(x, y):
        if x > 0.3:
            return str(round(x, 2)) + y
        return ""

    reviews_train['pos'] = reviews_train['pos'].apply(lambda x: comp(x, "pos"))
    reviews_train['neu'] = reviews_train['neu'].apply(lambda x: comp(x, "neu"))
    reviews_train['neg'] = reviews_train['neg'].apply(lambda x: comp(x, "neg"))
    new = reviews_train["pos"].copy()
    new1 = reviews_train["neu"].copy()
    new2 = reviews_train["neg"].copy()

    # concatenating team with name column
    # overwriting name column
    reviews_train["content_clean"] = reviews_train["content_clean"].str.cat(new, sep=" ")
    reviews_train["content_clean"] = reviews_train["content_clean"].str.cat(new1, sep=" ")
    reviews_train["content_clean"] = reviews_train["content_clean"].str.cat(new2, sep=" ")
    reviews_train['train'] = test_model(reviews_train)
    reviews_train.pop("content")
    sr = pd.Series(reviews_train['train'])
    export_csv = sr.to_csv('sentiment_analysis_team12_solution2.result.txt', index=False)


def build_model_final(X_train, X_test, y_train, y_test):
    model = Pipeline([('vectorizer', TfidfVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', LinearSVC())])  # train on TF-IDF vectors w/ Naive Bayes classifier
    parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
                  'tfidf__use_idf': (True, False)
        ,
                  }
    text_classifier = GridSearchCV(LinearSVC(), parameters, n_jobs=-1)
    text_classifier.fit(X_train, y_train)

    predictions = text_classifier.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))
    save_model(TfidfVectorizer(), text_classifier)


def build_model_svm(X_train, y_train, y_test, X_test):
    print("train")
    model_1 = Pipeline([

        ('vectorizer', TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),

        )),

    ])
    model_2 = Pipeline([

        ('vectorizer', TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 4),
        )),

    ])

    model = Pipeline([('feat_union', FeatureUnion(transformer_list=[

        ('1_pipeline', model_1),

        ('1_model_2', model_2),
        # ('1_model_3', model_3),
        # ('1_model_4', model_4),

    ])),
                      ('tfidf', TfidfTransformer()),

                      ('clf', LinearSVC())])

    text_classifier = model
    text_classifier.fit(X_train, y_train)
    save_model(TfidfVectorizer(), text_classifier)

    predictions = text_classifier.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))
    # y_pred = [x[1] for x in rf.predict_proba(X_test)]
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    #
    # roc_auc = auc(fpr, tpr)
    #
    # plt.figure(1, figsize=(15, 10))
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #           lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()


if __name__ == "__main__":
    # đọc dữ liệu tập train
    import io

    reviews_train = pd.read_csv("https://raw.githubusercontent.com/codethoisinhvien/kpdl/master/data.csv")
    # làm sạch dữ liệu train
    reviews_train = cleant_data(reviews_train)
    # đọc dữ liệu tập test
    # reviews_test = pd.read_csv("test.csv")
    # # làm sạch dữ liệu train
    # reviews_test = cleant_data(reviews_test)
    # # lấy danh sách top word
    # reviews_train = add_colum(reviews_train)
    from gensim.test.utils import common_texts
    import nltk

    #
    #  #nltk.downloader.download('vader_lexicon')
    #  from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    #
    #  from nltk.sentiment.vader import SentimentIntensityAnalyzer
    #
    #  sid = SentimentIntensityAnalyzer()
    #
    #  reviews_train["sentiments"] = reviews_train["content"].apply(lambda x: sid.polarity_scores(x))
    #  reviews_train = pd.concat([reviews_train.drop(['sentiments'], axis=1), reviews_train['sentiments'].apply(pd.Series)], axis=1)
    #  for col in reviews_train.columns:
    #      print(col)
    #
    #
    # # print(reviews_train["content_clean"])
    #  def comp(x,y):
    #       if x> 0.3:
    #           return str(round(x,2))+ y
    #       return ""
    #  reviews_train['pos']= reviews_train['pos'].apply(lambda x: comp(x,"pos"))
    #  reviews_train['neu'] = reviews_train['neu'].apply(lambda x: comp(x, "neu"))
    #  reviews_train['neg'] = reviews_train['neg'].apply(lambda x: comp(x, "neg"))
    #  new = reviews_train["pos"].copy()
    #  new1 = reviews_train["neu"].copy()
    #  new2 = reviews_train["neg"].copy()
    #
    #  # concatenating team with name column
    #  # overwriting name column
    #  reviews_train["content_clean"] = reviews_train["content_clean"].str.cat(new, sep=" ")
    #  reviews_train["content_clean"] = reviews_train["content_clean"].str.cat(new1, sep=" ")
    #  reviews_train["content_clean"] = reviews_train["content_clean"].str.cat(new2, sep=" ")
    #  # documents = [TaggedDocument(doc, [i]) for i, doc in
    #  #              enumerate(reviews_train["content_clean"].apply(lambda x: x.split(" ")))]
    #
    #
    #  #
    X_train, X_test, y_train, y_test = train_test_split(reviews_train['content_clean'], reviews_train['label'],
                                                        test_size=0.35, )
    build_model_svm(X_train, y_train, y_test, X_test)
#  #visualization()
#  # from gensim.test.utils import common_texts
#  # from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#  #
#  # documents = [TaggedDocument(doc, [i]) for i, doc in
#  #              enumerate(reviews_train["content_clean"].apply(lambda x: x.split(" ")))]
#  #
#  # # train a Doc2Vec model with our text data
#  # model = Doc2Vec(documents, vector_size=300, window=2, min_count=1, workers=4)
#
#  # # transform each document into a vector data
#  # doc2vec_df = reviews_train["content_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
#  # doc2vec_df.columns = ["content_clean_" + str(x) for x in doc2vec_df.columns]
#  # print (doc2vec_df)
#  # reviews_train = pd.concat([reviews_train, doc2vec_df], axis=1)
#  # tfidf = TfidfVectorizer(
#  #
#  #   )
#  # # word = TfidfVectorizer(
#  # #     analyzer='word',
#  # #     ngram_range=(1, 2),
#  # # )
#  # #
#  # tfidf_result = tfidf.fit_transform(reviews_train["content_clean"]).toarray()
#  # # tw2 =word.fit_transform(reviews_train["content_clean"]).toarray()
#  # # train = hstack((tfidf_result,tw2,reviews_train["pos"]))
#  # # print(tfidf_result)
#  #
#  # tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names())
#  # tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
#  # tfidf_df.index = reviews_train.index
#  # reviews_train = pd.concat([reviews_train, tfidf_df], axis=1)
#  # ignore_cols = ['label', "content", "content_clean",'lang']
#  # features = [c for c in reviews_train.columns if c not in ignore_cols]
#  # # print (len(features))
#  # X_train, X_test, y_train, y_test = train_test_split(reviews_train[features], reviews_train['label'],
#  #
#  #                                                  test_size=0.35,
#  #                                                  random_state=0)
#  # build_model_forest(X_train, y_train, y_test, X_test,reviews_train[features])
# visualization()
