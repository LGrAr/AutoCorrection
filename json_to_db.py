import json
import html
from bs4 import BeautifulSoup
import numpy as np

with open('DATASTORE/nownow_student_response.json') as json_file:
    data = json.load(json_file)

def clean_str(string): #MAYBE REVOIR POUR QUE CE SOIT VRAIMENT CLEAN
	string = html.unescape(string)
	string = BeautifulSoup(string, "lxml").text
	return string

import pandas as pd
df = pd.DataFrame(data)
df['answer'] = df['answer'].apply(lambda x: clean_str(x))
df['exercice'] = df['exercice'].apply(lambda x: clean_str(x))

last_names = ["MARTIN",
"BERNARD",
"THOMAS",
"PETIT",
"ROBERT",
"RICHARD",
"DURAND",
"DUBOIS",
"MOREAU",
"LAURENT",
"SIMON",
"MICHEL",
"LEFEBVRE",
"LEROY",
"ROUX",
"DAVID",
"BERTRAND",
"MOREL",
"FOURNIER",
"GIRARD",
"BONNET",
"DUPONT",
"LAMBERT",
"FONTAINE",
"ROUSSEAU",
"VINCENT",
"MULLER",
"LEFEVRE",
"FAURE",
"ANDRE",
"MERCIER",
"GUERIN",
"BOYER",
"GARNIER",
"CHEVALIER",
"FRANCOIS",
"LEGRAND",
"GAUTHIER",
"GARCIA",
"PERRIN",
"ROBIN",
"CLEMENT",
"MORIN",
"NICOLAS",
"HENRY",
"ROUSSEL"]

names = ["Thomas",
"Léa",
"Manon", 
"Alexandre",
"Quentin",
"Camille",
"Maxime",
"Nicolas",
"Lucas",
"Antoine",
"Océane",
"Clément",
"Julien",
"Hugo",
"Valentin",
"Laura",
"Alexis",
"Théo",
"Dylan",
"Romain",
"Sarah",
"Marine",
"Florian",
"Pauline",
"Julie",
"Mathilde",
"Kevin",
"Emma",
"Guillaume",
"Anthony",
"Anaïs",
"Pierre",
"Lucie",
"Benjamin",
"Justine",
"Adrien",
"Corentin",
"Louis",
"Marion",
"Morgane",
"Vincent",
"Paul",
"Mathieu",
"Baptiste", 
"Nathan",
"Jérémy"]

names = [(n, ln) for n, ln in zip(names, last_names)]
# database
import sqlite3
db = sqlite3.connect('DATASTORE/appDB.db')
cur = db.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS Students\
(id INTEGER PRIMARY KEY, name TEXT, lastname TEXT)')
db.commit()

cur.executemany("INSERT INTO Students (name, lastname) values (?, ?)",names)
db.commit()
import pandas as pd

print(pd.read_sql_query('SELECT * FROM Students', db))

cur.execute('CREATE TABLE IF NOT EXISTS Exercices\
(id INTEGER PRIMARY KEY, question TEXT, type_answer TEXT, englishTranslation TEXT)')
db.commit()

cur.executemany("INSERT INTO Exercices (question, type_answer, englishTranslation) VALUES (?, ?, ?)", [(i, None, None) for i in df['exercice'].unique()])
db.commit()
import pandas as pd
print(pd.read_sql_query('SELECT * FROM Exercices', db))

dict_df = pd.read_sql_query('SELECT * FROM Exercices', db)
m = dict_df[['id', 'question']].set_index('question').to_dict()
df['exercice_id'] = df.replace({"exercice": m['id']})['exercice']
df['student_id'] = [np.random.randint(1,47) for _ in range(len(df))]
df['similarity'] = None
df['ai_predicted'] = None

cur.execute('CREATE TABLE IF NOT EXISTS Responses\
(id INTEGER PRIMARY KEY, label TEXT, answer TEXT, exercice_id INT, student_id INT, similarity FLOAT, ai_predicted TEXT)')
db.commit()

cur.executemany("INSERT INTO Responses (label, answer, exercice_id, student_id, similarity, ai_predicted) VALUES (?, ?, ?, ?, ?)", df[['isRight', 'answer', 'exercice_id', 'student_id', 'similarity', 'ai_predicted']].values)
db.commit()
import pandas as pd
print(pd.read_sql_query('SELECT * FROM Responses', db))