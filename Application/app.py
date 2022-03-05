from flask import Flask, render_template, request, url_for, redirect, session
import sqlite3
import pandas as pd
import time
import numpy as np
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests, uuid, json
import flask_monitoringdashboard as dashboard
import pickle
from simpletransformers.classification import ClassificationModel
import os

from config import LOCATION, SUBSCRIPTION_KEY

global request

def preprocess_str(string):
    nlp = spacy.load('fr')
    doc = nlp(string)
    result = [w.lemma_ for w in doc if w.lemma_ not in STOP_WORDS and not w.is_punct]
    return ' '.join(result)

def connect_db():
    cwd = os.getcwd()
    db_path = cwd + '/DATASTORE/appDB.db'
    db = sqlite3.connect(db_path)
    cur = db.cursor()
    return cur, db

app = Flask(__name__)
app.secret_key = "super secret key"
dashboard.bind(app)

@app.route('/')
def home():
    #lire db, passer les messages en tant que variables a un html
    cur, db = connect_db()
    exercices = [content for content in cur.execute('SELECT exercice_id, count(label) as label, count(student_id) as responses, question, type_answer \
                  from Exercices JOIN Responses on Exercices.id = responses.exercice_id GROUP BY exercice_id ORDER BY label DESC \
                  ')]

    translated = [content for content in cur.execute('SELECT * FROM Exercices WHERE englishTranslation IS NOT NULL')]
    return render_template('form.html', exercices = exercices, translated = translated)


@app.route('/add_response', methods = ['POST'])
def add_response():
    question_id = request.form['button']
    question_id = int(question_id)
    cur, db = connect_db()
    question = [q[0] for q in cur.execute('SELECT question from Exercices WHERE id = ?', (question_id,))][0]
    session['question_id'] = question_id
    return render_template('write_type_answer.html', question = question)

@app.route("/write_type_answer", methods = ['POST'])
def write_answer():
    answer = request.form['user_phrase']
    question_id = session['question_id']
    cur, db = connect_db()
    cur.execute('UPDATE Exercices SET type_answer = ? WHERE id = ?', (answer, question_id))
    db.commit()
    return redirect(url_for('home'))

@app.route('/see_results', methods = ['POST'])
def results():
    question_id = request.form['button']
    question_id = int(question_id)
    cur, db = connect_db()

    responses = pd.read_sql_query('SELECT * from Responses WHERE exercice_id = (?)', db, params = (question_id,))
    answer = pd.read_sql_query('SELECT type_answer from Exercices WHERE id = (?)', db, params = (question_id,))
    
    if len(responses[responses['similarity'].isnull()]) != 0 and len(answer[answer['type_answer'].notnull()]) == 1:
        #calcul
        responses['answer_processed'] = responses['answer'].apply(lambda x: preprocess_str(x))
        X = responses['answer_processed'].tolist()
        X.append(preprocess_str(answer['type_answer'][0]))
        from sklearn.feature_extraction.text import TfidfVectorizer
        td = TfidfVectorizer()
        X_tfidf = td.fit_transform(X).toarray()
        a = []
        for i in range(len(X_tfidf) - 1):
            a.append(cosine_similarity([X_tfidf[i], X_tfidf[-1]])[0][1])
        responses['similarity'] = a
        cur.executemany('UPDATE Responses SET similarity = ? WHERE id = ?', [tuple(i) for i in responses[['similarity', 'id']].values])
        db.commit()

    responses = [content for content in cur.execute('SELECT question, answer, name, lastname, similarity, label, Responses.id FROM Responses join Students on Students.id = Responses.student_id join Exercices on Exercices.id = Responses.exercice_id WHERE exercice_id = ? ORDER BY similarity DESC', (question_id,))]
    responses = pd.read_sql_query('SELECT question, answer, name, lastname, similarity, label, Responses.id, ai_predicted FROM Responses join Students on Students.id = Responses.student_id join Exercices on Exercices.id = Responses.exercice_id WHERE exercice_id = ? ORDER BY similarity DESC', db, params = (question_id,))
    responses["label"].replace({"1": "Vrai", "0": "Faux"}, inplace=True)
    responses["ai_predicted"].replace({"1.0": "Vrai", "0.0": "Faux"}, inplace=True)
    return render_template('results.html', responses = responses.values)


@app.route("/write_true", methods = ['POST'])
def true():
    response_id = request.form['edit']
    response_id = int(response_id)
    cur, db = connect_db()
    cur.execute('UPDATE Responses SET label = ? WHERE id = ?', (1, response_id))
    db.commit()
    return redirect(url_for('home'))

@app.route("/write_false", methods = ['POST'])
def false():
    response_id = request.form['edit']
    response_id = int(response_id)
    cur, db = connect_db()
    cur.execute('UPDATE Responses SET label = ? WHERE id = ?', (0, response_id))
    db.commit()
    return redirect(url_for('home'))

@app.route("/train", methods = ['POST'])
def train():
    
    with open('C:/Users/grego/CorrectionAutomatique/Application/ids_trained.pkl', 'rb') as f:
        exercice_id = pickle.load(f)

    model_load_test = ClassificationModel("camembert", "C:/Users/grego/CorrectionAutomatique/outputs", use_cuda = False, weight = [1.0,0.325])
    
    cur, db = connect_db()
    responses = pd.read_sql_query("SELECT Responses.id, label, ai_predicted, answer, question, exercice_id FROM Responses INNER JOIN Exercices ON Responses.exercice_id = Exercices.id", db)
    mask = responses['exercice_id'].isin(exercice_id)
    responses = responses[mask]
    responses['question + answer'] = responses['answer'] + responses['question']
    preds = model_load_test.predict(responses['question + answer'].values.tolist())
    preds = preds[0]
    ui = [str(x) for x in preds]
    responses['ai_predicted'] = ui
    #cur.executemany('UPDATE Responses SET ai_predicted = ? WHERE id = ?', [tuple(i) for i in responses[['ai_predicted', 'id']].values])
    #db.commit()

    return redirect(url_for('home'))
 


@app.route("/translate", methods = ['POST'])
def translate():
    from flask import request
    question_id = request.form['tr']
    question_id = int(question_id)
    cur, db = connect_db()

    question = [q[0] for q in cur.execute('SELECT question from Exercices WHERE id = ?', (question_id,))][0]
    subscription_key = SUBSCRIPTION_KEY
    endpoint = "https://api.cognitive.microsofttranslator.com"
    location = LOCATION

    path = '/translate'
    constructed_url = endpoint + path
    params = {
        'api-version': '3.0',
        'from': 'fr',
        'to': ['en']}
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())}
    body = [{
        'text': question
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    response = response[0]['translations'][0]['text']
    cur, db = connect_db()
    cur.execute('UPDATE Exercices SET englishTranslation = ? WHERE id = ?', (response, question_id))
    db.commit()
    return redirect(url_for('home'))


@app.route("/p_test", methods = ['POST'])
def p_test():
    
    #with open('C:/Users/grego/CorrectionAutomatique/Application/ids_trained.pkl', 'rb') as f:
        #exercice_id = pickle.load(f)

    #model_load_test = ClassificationModel("camembert", "C:/Users/grego/CorrectionAutomatique/outputs", use_cuda = False, weight = [1.0,0.325])
    
    cur, db = connect_db()
    responses = pd.read_sql_query("SELECT Responses.id, label, ai_predicted, answer, question, exercice_id FROM Responses INNER JOIN Exercices ON Responses.exercice_id = Exercices.id", db)
    #mask = responses['exercice_id'].isin(exercice_id)
    #responses = responses[mask]
    import time
    time.sleep(5)
    responses['question + answer'] = responses['answer'] + responses['question']
    #preds = model_load_test.predict(responses['question + answer'].values.tolist())
    #preds = preds[0]
    #ui = [str(x) for x in preds]
    #responses['ai_predicted'] = ui
    #cur.executemany('UPDATE Responses SET ai_predicted = ? WHERE id = ?', [tuple(i) for i in responses[['ai_predicted', 'id']].values])
    #db.commit()

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0')
 