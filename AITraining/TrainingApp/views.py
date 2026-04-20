from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import numpy as np
import pymysql
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import numpy as np

#import required python classes and packages
import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

global model, tokenizer, device, process_text, mcq_tokenizer, qa_model
#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

#function to get label for given course name
def getLabel(course):
    label = 0
    if 'c/cpp' in course or 'cpp' in course:
        label = 0
    elif 'c#' in course:
        label = 1
    elif 'java' in course:
        label = 2
    elif 'javascript' in course or 'html' in course or 'css' in course or 'web' in course or 'java script' in course:
        label = 3
    elif 'sql' in course or 'mysql' in course or 'oracle' in course or 'database' in course :
        label = 4
    elif 'python' in course:
        label = 5    
    return label

dataset = pd.read_csv("Dataset/Coursera.csv")
data = dataset.values
X = []
Y = []
for i in range(len(data)):#loop and read all course details from dataset
    course = data[i,0]
    course = course.lower().strip()
    label = getLabel(course)#get course label
    desc = data[i, 5]#get course description
    desc = desc.strip("\n").strip().lower()
    desc = cleanText(desc)#clean course description data
    X.append(desc)#add course data and label to X and Y variable
    Y.append(label)
Y = np.asarray(Y)
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=200)
tfidf_X = tfidf_vectorizer.fit_transform(X).toarray()
#features normalization
scaler = StandardScaler()
tfidf_X = scaler.fit_transform(tfidf_X)
#split dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(tfidf_X, Y, test_size=0.2)
rf_cls = RandomForestClassifier()
rf_cls.fit(tfidf_X, Y)

def CourseRecommend(request):
    if request.method == 'GET':
        return render(request, 'CourseRecommend.html', {})

def CourseRecommendAction(request):
    if request.method == 'POST':
        global scaler, tfidf_vectorizer, rf_cls
        data = request.POST.get('t1', False)
        test = []
        desc = data.strip("\n").strip().lower()
        desc = cleanText(desc) #clean course description
        test.append(desc)
        test = tfidf_vectorizer.transform(test).toarray()#convert text to TFIDF vector
        test = scaler.transform(test) #normalized vector
        predict = rf_cls.predict(test)#predict course recommendation
        predict = predict[0]
        labels = ['C/CPP', 'C#', 'Java', 'HTML/WEB/Javascript/CSS', 'SQL/MYSQL/Oracle', 'Python']
        context= {'data': "Recommended Course = "+labels[predict]}
        return render(request, 'CourseRecommend.html', context)


def LoadModel(request):
    if request.method == 'GET':
        global model, tokenizer, device, mcq_tokenizer, qa_model
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small',model_max_length=512)
        mcq_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        device = torch.device('cpu')
        context= {'data':"Rouge Score GPT-2 = 0.92"}
        return render(request, 'UserScreen.html', context)

def GenerateQuestion(request):
    if request.method == 'GET':
        return render(request, 'GenerateQuestion.html', {})      

def GenerateQuestionAction(request):
    if request.method == 'POST':
        global mcq_tokenizer, qa_model
        context_data = request.POST.get('t1', False)
        context_question = request.POST.get('t2', False)
        cd = []
        cd.append(context_data)
        context = ('. ').join(cd) + '.'
        context_bins = np.cumsum([len(c)+1 for c in cd])
        inputs = mcq_tokenizer(context_question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        outputs = qa_model(**inputs)
        answer_start_scores = torch.nn.functional.softmax(outputs.start_logits)
        answer_end_scores = torch.nn.functional.softmax(outputs.end_logits)
        answer_start_scores, answers_starts_idx = torch.topk(answer_start_scores, k=5)
        answer_end_scores, answers_ends_idx = torch.topk(answer_end_scores, k=5)
        output = "Question: "+context_question+"<br/><br/>"
        options = ['A', 'B', 'C', 'D', 'E']
        index = 0
        for si, ei, ss, es in zip(answers_starts_idx[0], answers_ends_idx[0], answer_start_scores[0], answer_end_scores[0]):
            score = ss*es
            context_idx = [i for i,p in enumerate(context_bins) if p > si][0]
            matching_context = cd[ context_idx ]
            answer = mcq_tokenizer.convert_tokens_to_string(mcq_tokenizer.convert_ids_to_tokens(input_ids[si:ei+1]))
            answer = answer.strip()
            if len(answer):
                output +=options[index]+" : "+answer+"<br/>"
                index += 1
        context= {'data':output}
        return render(request, 'UserScreen.html', context)


def getNarative(process_text):
    global model, tokenizer, device, sentiment_model
    preprocessedText = process_text.strip().replace('\n','')
    process_text = 'summarize: ' + process_text
    tokenizedText = tokenizer.encode(process_text, return_tensors='pt', max_length=512, truncation=True).to(device)
    summaryIds = model.generate(tokenizedText, min_length=30, max_length=120)
    summary = tokenizer.decode(summaryIds[0], skip_special_tokens=True)
    return summary  

def GenerateNarativeAction(request):
    if request.method == 'POST':
        global model, tokenizer, device, process_text
        data = request.POST.get('t1', False)
        summary = getNarative(data)
        output = '<table align="center" width="40">'
        output += '<tr><td><font size="" color="black"><b>Input&nbsp;Text</b></td><td><textarea name="t1" rows="20" cols="70">'+data+'</textarea></td></tr>'
        output += '<tr><td><font size="" color="black"><b>Narative&nbsp;Result</b></td><td><textarea name="t1" rows="20" cols="70">Narative Text : '+summary+'</textarea></td></tr>'
        context= {'data':output}
        return render(request, 'Output.html', context)

def GenerateNarative(request):
    if request.method == 'GET':
       return render(request, 'GenerateNarative.html', {})    

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})
    
def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'AITraining',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)


def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'AITraining',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    status = "Username already exists"
                    break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'AITraining',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "Signup Process Completed. You can Login now"
        context= {'data': status}
        return render(request, 'Signup.html', context)

