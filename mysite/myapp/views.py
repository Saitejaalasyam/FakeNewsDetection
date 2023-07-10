from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import gc
import nltk
import re
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron,SGDClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pyautogui as pag


# Create your views here.
stop_words = stopwords.words('english')
ps = WordNetLemmatizer()
def cleaning_data(row):
    row = row.lower()
    row = re.sub('[^a-zA-Z]', ' ', row)
    tokens = row.split()
    news = [ps.lemmatize(word) for word in tokens if not word in stop_words]
    cleaned_news = ' '.join(news)
    return cleaned_news

def index(request):
    global df
    try:
        global uploaded_file_1
        global uploaded_file_2
        uploaded_file_1 = request.FILES['my_uploaded_file_1']
        uploaded_file_2 = request.FILES['my_uploaded_file_2']


    except:
        print("datasets not selected")
    if request.method == 'POST':

        if 'Load' in request.POST:

            True_news = pd.read_csv(uploaded_file_1)
            Fake_news = pd.read_csv(uploaded_file_2)
            l = [i for i in range(4000, 21417)]
            True_news.drop(index=l)
            k = [i for i in range(4000, 21417)]
            Fake_news.drop(index=k)
            True_news['label'] = 0
            Fake_news['label'] = 1
            dataset1 = True_news[['text', 'label']]
            dataset2 = Fake_news[['text', 'label']]


            df = pd.concat([dataset1, dataset2])
            del True_news
            del Fake_news
            del dataset1, dataset2
            print(df.head())
            pag.alert(text="Loading database complete", title="Alert!")
        elif 'preprocess' in request.POST:
            df = df.sample(frac=1, random_state=0)


            global dataset,dt1,dt2
            dataset = df.iloc[:1000, :]
            dt1 = df.iloc[3000:4000, :]
            dt2 = df.iloc[5000:6000, :]


            global vectorizer
            dataset['text'] = dataset['text'].apply(lambda x: cleaning_data(x))
            dt1['text'] = dt1['text'].apply(lambda x: cleaning_data(x))
            dt2['text'] = dt2['text'].apply(lambda x: cleaning_data(x))
            vectorizer = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1, 2))
            print(dt1.head())
            pag.alert(text="preprocessing database complete", title="Alert!")
        elif 'next' in request.POST:
            return redirect('Next')
        elif 'Prepare' in request.POST:

            X = dataset.iloc[:, 0]
            y = dataset.iloc[:, 1]
            X1 = dt1.iloc[:, 0]
            y1 = dt1.iloc[:, 1]
            X2 = dt2.iloc[:, 0]
            y2 = dt2.iloc[:, 1]
            global training_data,training_data_1,training_data_2,testing_data,testing_data_1,testing_data_2
            global train_label,train_label1,train_label2,test_label,test_label1,test_label2
            train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.1, random_state=0)
            train_data1, test_data1, train_label1, test_label1 = train_test_split(X1, y1, test_size=0.1, random_state=0)
            train_data2, test_data2, train_label2, test_label2 = train_test_split(X2, y2, test_size=0.1, random_state=0)
            vec_train_data = vectorizer.fit_transform(train_data)
            vec_train_data1 = vectorizer.fit_transform(train_data1)
            vec_train_data2 = vectorizer.fit_transform(train_data2)
            vec_test_data = vectorizer.transform(test_data)
            vec_test_data1 = vectorizer.transform(test_data1)
            vec_test_data2 = vectorizer.transform(test_data2)
            vec_train_data = vec_train_data.toarray()
            vec_test_data = vec_test_data.toarray()
            training_data = pd.DataFrame(vec_train_data, columns=vectorizer.get_feature_names())
            testing_data = pd.DataFrame(vec_test_data, columns=vectorizer.get_feature_names())
            vec_train_data1 = vec_train_data1.toarray()
            vec_test_data1 = vec_test_data1.toarray()
            training_data_1 = pd.DataFrame(vec_train_data1, columns=vectorizer.get_feature_names())
            testing_data_1 = pd.DataFrame(vec_test_data1, columns=vectorizer.get_feature_names())
            vec_train_data2 = vec_train_data2.toarray()
            vec_test_data2 = vec_test_data2.toarray()
            training_data_2 = pd.DataFrame(vec_train_data2, columns=vectorizer.get_feature_names())
            testing_data_2 = pd.DataFrame(vec_test_data2, columns=vectorizer.get_feature_names())

            pag.alert(text="Preparing Data Complete",title="Alert!")
    return render(request=request, template_name='index.html')


def Next(request):
    if request.method == 'POST':
        global training_data, training_data_1, training_data_2, testing_data, testing_data_1, testing_data_2
        global train_label, train_label1, train_label2, test_label, test_label1, test_label2
        global multinomial_acc,pac_acc,perceptron_acc,SGD_acc
        try:
            global input_data
            input_data=request.POST['testing']

            if 'MultinomialNB' in request.POST:

                clf = MultinomialNB()
                clf.fit(training_data, train_label)
                clf.partial_fit(training_data_1, train_label1)
                clf.partial_fit(training_data_2, train_label2)
                y_pred = clf.predict(testing_data)
                y_pred1 = clf.predict(testing_data_1)
                y_pred2 = clf.predict(testing_data_2)
                news = cleaning_data(str(input_data))
                single_prediction = clf.predict(vectorizer.transform([news]).toarray())
                if single_prediction[0] == 1:
                    msg="Fake News"
                elif single_prediction[0]==0:
                    msg="True News"
                pag.alert(text=msg,title="Output!")

                multinomial_acc=accuracy_score(test_label,y_pred)
                print(single_prediction)
                print(classification_report(test_label, y_pred))
                print(classification_report(test_label1, y_pred1))
                print(classification_report(test_label2, y_pred2))
            elif 'Pac' in request.POST:
                pac = PassiveAggressiveClassifier(max_iter=50, random_state=0)
                pac.fit(training_data, train_label)
                pac.partial_fit(training_data_1, train_label1)
                pac.partial_fit(training_data_2, train_label2)
                y_pred = pac.predict(testing_data)
                y_pred1 = pac.predict(testing_data_1)
                y_pred2 = pac.predict(testing_data_2)
                news = cleaning_data(str(input_data))
                single_prediction = pac.predict(vectorizer.transform([news]).toarray())
                if single_prediction[0] == 1:
                    msg = "Fake News"
                elif single_prediction[0] == 0:
                    msg = "True News"
                pag.alert(text=msg, title="Output!")

                pac_acc=accuracy_score(test_label,y_pred)
                print(single_prediction)
                print(classification_report(test_label,y_pred))
                print(classification_report(test_label1, y_pred1))
                print(classification_report(test_label2, y_pred2))
            elif 'Perceptron' in request.POST:
                model = Perceptron()
                model.fit(training_data, train_label)
                model.partial_fit(training_data_1, train_label1)
                model.partial_fit(training_data_2, train_label2)
                y_pred = model.predict(testing_data)
                y_pred1 = model.predict(testing_data_1)
                y_pred2 = model.predict(testing_data_2)
                news = cleaning_data(str(input_data))
                single_prediction = model.predict(vectorizer.transform([news]).toarray())
                if single_prediction[0] == 1:
                    msg = "Fake News"
                elif single_prediction[0] == 0:
                    msg = "True News"
                pag.alert(text=msg, title="Output!")

                perceptron_acc=accuracy_score(test_label,y_pred)
                print(single_prediction)
                print(classification_report(test_label, y_pred))
                print(classification_report(test_label1, y_pred1))
                print(classification_report(test_label2, y_pred2))
            elif 'SGD' in request.POST:
                sgd = SGDClassifier(random_state=0)
                sgd.fit(training_data, train_label)
                sgd.partial_fit(training_data_1, train_label1)
                sgd.partial_fit(training_data_2, train_label2)

                y_pred = sgd.predict(testing_data)
                y_pred1 = sgd.predict(testing_data_1)
                y_pred2 = sgd.predict(testing_data_2)
                news = cleaning_data(str(input_data))
                single_prediction = sgd.predict(vectorizer.transform([news]).toarray())
                if single_prediction[0] == 1:
                    msg = "Fake News"
                elif single_prediction[0] == 0:
                    msg = "True News"
                pag.alert(text=msg, title="Output!")
                print(single_prediction)
                SGD_acc=accuracy_score(test_label,y_pred)
                print(classification_report(test_label, y_pred))
                print(classification_report(test_label1, y_pred1))
                print(classification_report(test_label2, y_pred2))
            if 'next' in request.POST:
                print(multinomial_acc*100,pac_acc*100,perceptron_acc*100,SGD_acc*100)
                return render(request, "result.html", {'MN_Accuracy': (multinomial_acc*100),
                                                       'PAC_Accuracy': (pac_acc*100),
                                                       'PER_Accuracy': (perceptron_acc*100),
                                                       'SGD_Accuracy': (SGD_acc*100),
                                                       })


            if 'Show' in request.POST:
                X = [multinomial_acc * 100, pac_acc * 100, perceptron_acc * 100, SGD_acc * 100]
                bars = ['multinomial_acc', 'pac_acc', 'perceptron_acc', 'SGD_acc']
                y_pos = np.arange(len(bars))
                plt.bar(y_pos, X)
                plt.xticks(y_pos, bars)
                plt.ylabel('Accuracy(in %)')
                plt.title("Accuracy Comparison Graph")
                plt.show()
        except:
            pag.alert("Check your input")

    return render(request=request, template_name='load.html')


def result(request):

    return render(request=request,template_name='result.html')