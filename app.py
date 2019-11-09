import pandas as pd

pd.set_option("display.max_colwidth", 200)
import numpy as np
import string
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import pickle
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
def read_data():
    pass






def cleant_data(data):
    def clean_text(text):
        # chuyển thành chữ thường
        text = text.lower()
        # chuyen thanh token
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        # xoa từ chứa số
        text = [word for word in text if not any(c.isdigit() for c in word)]
        # remove stop words
        stop = ['khách', 'sạn', 'phòng', 'có', 'và', 'sẽ', 'ăn', 'là', 'giá','biển']
        text = [x for x in text if x not in stop]
        right=['tốt', 'xuất',  'sắc', 'tuyệt','vời','dễ','chịu','de','chiu','hao','hảo','ok',]
        right_2=['dễ', 'tạm', 'không', 'vời', 'tàm', 'được','đẹp','sạch','sach']
        right_3=[   'kém','tuy','do','kem','thất','vọng',]

        def len_text(x):
            l='0'
            if x in right:
                l='4'
            if x in right_2:
                l='2'
            if x in right_3:
                l='-1'
            return l
        text_2 = [str(len_text(x)) for x in text  ]

        text_3 = len(text)

        text = " ".join(text)
        text_2=''.join(text_2)
        text=text+' '+text_2+' '+str(text_3)



        return (text)

    data["content_clean"] = data["content"].apply(lambda x: clean_text(x))
    return data


from sklearn import svm







from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def build_model_forest(X_train, y_train, y_test, X_test):
    print ("start traning")
    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, y_train)
    predictions = text_classifier.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))
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
    pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))
    # pickling the model
    pickle.dump(classifier_linear, open('classifier.sav', 'wb'))


def test_model():
    test_reviews_df = pd.read_csv("data.csv")
    vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
    classifier = pickle.load(open('classifier.sav', 'rb'))
    test_reviews_df = cleant_data(test_reviews_df)
    # X_train, X_test, y_train, y_test = train_test_split(reviews_train['content_clean'], reviews_train['label'],
    #                                                     test_size=0,
    #                                                     random_state=0)
    result = classifier.predict(test_reviews_df['content_clean'])
    print (len(result))
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
    test_reviews_df = pd.read_csv("data.csv")
    test_reviews_df['train'] = test_model()
    test_reviews_df['True or False'] = test_reviews_df['train'] == test_reviews_df['label']
    test_reviews_df.pop("content")
    export_csv = test_reviews_df.to_csv('result.csv')
    right = 0
    for x in test_reviews_df['True or False']:
        if x == True:
            right = right + 1
    divisions = ['Đúng']
    print(right / len(test_reviews_df['True or False']))
    divisions_marks = [right / len(test_reviews_df['True or False'])]
    plt.bar(divisions, divisions_marks, color="green")
    plt.title("%")
    plt.xlabel('label')
    plt.ylabel('so luong ban gi')
    plt.show()
def build_model_final(X_train, X_test, y_train, y_test):
    model = Pipeline([('vectorizer', TfidfVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf',LinearSVC())                  ]) # train on TF-IDF vectors w/ Naive Bayes classifier
    parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
                  'tfidf__use_idf': (True, False)}
    text_classifier = GridSearchCV(LinearSVC(), parameters, n_jobs=-1)
    text_classifier.fit(X_train, y_train)


    predictions = text_classifier.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))
    save_model(TfidfVectorizer(), text_classifier)

def build_model_svm(X_train, y_train, y_test, X_test):
    print("train")
    model_1 = Pipeline([('vectorizer',TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
       ))])
    model = Pipeline([('feat_union', FeatureUnion(transformer_list=[
          ('1_pipeline', model_1),



          ])),
                       ('tfidf', TfidfTransformer()),
                       ('clf',LinearSVC())])
    parameters = {
                  'tfidf__use_idf': (True, False),
    }

    text_classifier =GridSearchCV(model,parameters,n_jobs=-1
                                          ) #GridSearchCV(LinearSVC(), parameters, n_jobs=-1)
    text_classifier.fit(X_train, y_train)

    predictions = text_classifier.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))
    save_model(TfidfVectorizer(),text_classifier)

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
    reviews_train = pd.read_csv("data.csv")
    # làm sạch dữ liệu train
    reviews_train = cleant_data(reviews_train)
    # đọc dữ liệu tập test
    reviews_test = pd.read_csv("test.csv")
    # làm sạch dữ liệu train
    reviews_test = cleant_data(reviews_test)
    # lấy danh sách top word
    reviews_train = add_colum(reviews_train)

    X_train, X_test, y_train, y_test = train_test_split(reviews_train['content_clean'], reviews_train['label'],
                                                     test_size=0.35,
                                                     random_state=0)

    build_model_svm(X_train, y_train, y_test, X_test)
    visualization()