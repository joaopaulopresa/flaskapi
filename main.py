#!/usr/bin/env python
import pickle

import nltk as nltk
from flask import Flask, request, jsonify
from google.cloud import storage

from functions import my_prepro as my_preprocessor


def transform(x):
    # ter o metodo qeu recebe o texto e aplica o metodo remove_tokenize_remove_stopword_stemming
    nltk.download('punkt')
    nltk.download('stopwords')
    x = x.lower()
    x = nltk.word_tokenize(x)
    stopwords = nltk.corpus.stopwords.words('portuguese')
    x = [item for item in x if item not in stopwords]
    return x

# criar o x do texto transformado pelo tfidf
def prob(x, assunto):
    storage_client = storage.Client()
    bucket_classificadores = storage_client.get_bucket('classificadores')
    bucket_preprocessors = storage_client.get_bucket('vectorizer')
    if (assunto == 'dano_moral'):
        blob_dano_moral = bucket_classificadores.blob('dano_moral')
        clf = blob_dano_moral.download_as_string()
        classifiers = pickle.loads(clf, encoding='latin1')
        blob_pre_dano_moral = bucket_preprocessors.blob('dano_moral.pkl')
        prestr = blob_pre_dano_moral.download_as_string()
        pre = pickle.loads(prestr, encoding='latin1')
        pre.set_params(tokenizer=my_preprocessor,preprocessor=my_preprocessor)
        x = pre.transform([x])
    if (assunto == 'acidente_transito'):
        blob_acidente_transito = bucket_classificadores.blob('acidente_transito')
        clf = blob_acidente_transito.download_as_string()
        classifiers = pickle.loads(clf, encoding='latin1')
        blob__pre_acidente_transito = bucket_preprocessors.blob('acidente_transito.pkl')
        prestr = blob__pre_acidente_transito.download_as_string()
        pre = pickle.loads(prestr, encoding='latin1')
        pre.set_params(tokenizer=my_preprocessor,preprocessor=my_preprocessor)
        x = pre.transform([x])
    result = {'classificador': [], 'probabilidade': []}
    for clf in classifiers:
        name = clf.__class__.__name__
        proba = clf.predict_proba(x)
        result['classificador'].append(name)
        result['probabilidade'].append(str(proba))
    return result


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route('/')
def index():
    return "Hello World!"

@app.route('/predict', methods=['POST'])
def predict():
    assunto = request.args['assunto']
    peticao = request.files['peticao']
    peticao_data = peticao.read().decode()
    print(peticao_data)
    x = peticao_data
    x = transform(x)
    result = prob(x, assunto)
    print(result)
    return jsonify(result)




if __name__ == '__main__':
    app.run(debug=True)
