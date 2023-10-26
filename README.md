# Fake-news-detection-using-NLP
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pickle

main = Tk()
main.title("FAKE NEWS CLASSIFICATION ON TWITTER")
main.geometry("1300x1200")

global filename
global X, Y
global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
global tfidf_vectorizer
global accuracy,accuracy1,accuracy2,accuracy3

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []
global classifier

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():    
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="TwitterNewsData")
    textdata.clear()
    labels.clear()
    dataset = pd.read_csv(filename)
    for i in range(len(dataset)):
        msg = dataset.get_value(i, 'text')
        label = dataset.get_value(i, 'target')
        msg = str(msg)
        msg = msg.strip().lower()
        labels.append(int(label))
        clean = cleanPost(msg)
        textdata.append(clean)
        text.insert(END,clean+" ==== "+str(label)+"\n")
    


def preprocess():
    text.delete('1.0', END)
    global Y
    global tfidf_vectorizer
    global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, ngram_range=(1,2),smooth_idf=False, norm=None, decode_error='replace', max_features=200)
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:200]
    Y = np.asarray(labels)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"\n\nTotal News found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train machine learning algorithms : "+str(len(tfidf_X_train))+"\n")
    text.insert(END,"Total records used to test machine learning algorithms  : "+str(len(tfidf_X_test))+"\n")


def runDecisionTree():
    text.delete('1.0', END)
    global classifier
    global accuracy
    cls = DecisionTreeClassifier()
    cls.fit(tfidf_X_train, tfidf_y_train)
    predict = cls.predict(tfidf_X_test)
    acc = accuracy_score(tfidf_y_test,predict)*100
    accuracy = acc
    
    text.insert(END,"Decision Tree Accuracy : "+str(acc)+"\n")
    classifier = cls
    with open('model.txt', 'wb') as file:
        pickle.dump(classifier, file)
    file.close()

def  runRandom():
    text.delete('1.0', END)
    global classifier
    global accuracy1
    cls = RandomForestClassifier(max_depth=2, random_state=0)
    cls.fit(tfidf_X_train, tfidf_y_train)
    predict = cls.predict(tfidf_X_test)
    acc = accuracy_score(tfidf_y_test,predict)*100
    accuracy1 = acc
    
    text.insert(END,"Random Forest Accuracy : "+str(acc)+"\n")

def runGradient():
    text.delete('1.0', END)
    global classifier
    global accuracy2
    cls = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    cls.fit(tfidf_X_train, tfidf_y_train)
    predict = cls.predict(tfidf_X_test)
    acc = accuracy_score(tfidf_y_test,predict)*100
    accuracy2 = acc
    
    text.insert(END,"Gradient Boosting Accuracy : "+str(acc)+"\n")

def runPassive():
    text.delete('1.0', END)
    global classifier
    global accuracy3
    cls = PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3)
    cls.fit(tfidf_X_train, tfidf_y_train)
    predict = cls.predict(tfidf_X_test)
    acc = accuracy_score(tfidf_y_test,predict)*100
    accuracy3 = acc
    
    text.insert(END,"Passive Aggressive Classifier Accuracy : "+str(acc)+"\n")

    

    
def graph():
    bars = ('Decision Tree','Random Forest','Gradient Boosting','Passive Aggressive Classifier')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, [accuracy,accuracy1,accuracy2,accuracy3])
    plt.xticks(y_pos, bars)
    plt.show()

def predict():
    testfile = filedialog.askopenfilename(initialdir="TwitterNewsData")
    testData = pd.read_csv(testfile)
    text.delete('1.0', END)
    testData = testData.values
    testData = testData[:,0]
    print(testData)
    for i in range(len(testData)):
        msg = testData[i]
        msg1 = testData[i]
        print(msg)
        review = msg.lower()
        review = review.strip().lower()
        review = cleanPost(review)
        testReview = tfidf_vectorizer.transform([review]).toarray()
        predict = classifier.predict(testReview)
        print(predict)
        if predict == 0:
            text.insert(END,msg1+" === Given news predicted as GENUINE\n\n")
        else:
            text.insert(END,msg1+" == Given news predicted as FAKE\n\n")
        
    
font = ('times', 15, 'bold')
title = Label(main, text='FAKE NEWS CLASSIFICATION ON TWITTER')
#title.config(bg='powder blue', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Fake News Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset & Apply NGram", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
dtButton.place(x=20,y=200)
dtButton.config(font=ff)

graphButton = Button(main, text="Run Random Forest Algorithm", command=runRandom)
graphButton.place(x=20,y=250)
graphButton.config(font=ff)

graphButton1 = Button(main, text="Run Gradient Boosting Algorithm", command=runGradient)
graphButton1.place(x=20,y=300)
graphButton1.config(font=ff)

graphButton2 = Button(main, text="Run Passive Aggressive Algorithm", command=runPassive)
graphButton2.place(x=20,y=350)
graphButton2.config(font=ff)

graphButton3 = Button(main, text="Run Accuracy comparison", command=graph)
graphButton3.place(x=20,y=400)
graphButton3.config(font=ff)

predictButton = Button(main, text="Test News Detection", command=predict)
predictButton.place(x=20,y=450)
predictButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config()
main.mainloop()
