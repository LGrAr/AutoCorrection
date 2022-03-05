from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import sklearn
from sklearn.model_selection import KFold
import sqlite3
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from simpletransformers.classification import ClassificationModel

def connect_db():
    db = sqlite3.connect('C:/Users/grego/CorrectionAutomatique/DATASTORE/appDB.db')
    cur = db.cursor()
    return cur, db

cur, db = connect_db()

added_negative_responses = pd.read_sql_query("SELECT label, answer, question, exercice_id FROM AddedNegativeResponses INNER JOIN Exercices ON AddedNegativeResponses.exercice_id = Exercices.id", db)
students_responses_labelised = pd.read_sql_query("SELECT label, answer, question, exercice_id FROM Responses INNER JOIN Exercices ON Responses.exercice_id = Exercices.id", db)

students_responses_labelised = students_responses_labelised[students_responses_labelised['label'].notnull()]
df = pd.concat([students_responses_labelised,added_negative_responses])
df.replace('0', 0, inplace = True)
df.replace('1', 1, inplace = True)

df_False = df[df['label'] == 0]
df_True = df[df['label'] == 1]

filtered = df_False.groupby("exercice_id").filter(lambda x: x['label'].count() >= 3)
ids_false = filtered.groupby("exercice_id").count()
ids_false = ids_false.index.tolist()

filtered = df_True.groupby("exercice_id").filter(lambda x: x['label'].count() >= 5)
ids_true = filtered.groupby("exercice_id").count()
ids_true = ids_true.index.tolist()

exercice_id = list(set(ids_true).intersection(ids_false))

train_df = pd.DataFrame(columns = ['label', 'answer', 'question', 'exercice_id'])
test_df = pd.DataFrame(columns = ['label', 'answer', 'question', 'exercice_id'])

for i in exercice_id:
    sub_train_df_true = df_True[df_True['exercice_id'] == i].sample(frac=0.75)
    sub_test_df_true = df_True[df_True['exercice_id'] == i].drop(sub_train_df_true.index)

    sub_train_df_false = df_False[df_False['exercice_id'] == i].sample(frac=0.75)
    sub_test_df_false = df_False[df_False['exercice_id'] == i].drop(sub_train_df_false.index)
    
    train_df = train_df.append(sub_train_df_true)
    train_df = train_df.append(sub_train_df_false)
    
    test_df = test_df.append(sub_test_df_true)
    test_df = test_df.append(sub_test_df_false)

train_df = shuffle(train_df)
test_df = shuffle(test_df)

train_df['question + answer'] = train_df['answer'] + train_df['question']
test_df['question + answer'] = test_df['answer'] + test_df['question']

test_df = test_df[['question + answer', 'label']]
train_df = train_df[['question + answer', 'label']]

df_copy = train_df.append(test_df)
df_copy = shuffle(df_copy)
df_copy

model_args = {
    'num_train_epochs': 4,
    'learning_rate' : 1e-5,
    'overwrite_output_dir': True,
    'output_dir': 'training'
    }

model = ClassificationModel(model_type = 'camembert', model_name ='camembert-base', use_cuda=False, weight = [1.0,0.325], args = model_args)

kf = KFold(n_splits=5, shuffle=True)

for train_index, val_index in kf.split(df_copy):
    train_df = df_copy.iloc[train_index]
    val_df = df_copy.iloc[val_index]


    model.train_model(train_df, acc = sklearn.metrics.f1_score)
    result, model_outputs, wrong_predictions = model.eval_model(val_df, acc = sklearn.metrics.f1_score)
    print(result['acc'])

with open('ids_trained.pkl', 'wb') as f:
    pickle.dump(exercice_id, f)