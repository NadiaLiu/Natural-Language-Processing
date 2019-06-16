#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import gensim
t1 = time.time()


def remove_nonalpha(string):
    """Function remove_nonalpha() is for removing all non-alphanumeric characters except spaces."""
    pattern = re.compile('[^A-Za-z0-9 ]+')
    newstring = pattern.sub(' ', string)
    return newstring


def remove_stopwords(string):
    """Function remove_stopwords() is for removing stopwords which are meaningless."""
    stopWords = set(stopwords.words('english'))
    newstring = []
    for word in string.split(' '):
        if word not in stopWords:
            newstring.append(word)
    return ' '.join(newstring)


def remove_character(string):
    """Function remove_character() is for removing words with 2 character or less."""
    pattern = re.compile('\\b[a-z]{1,2}\\b')
    newstring = pattern.sub(' ', string)
    newstring = re.sub('[ ]{2,}', ' ', newstring)
    return newstring


def remove_digit(string):
    """Function remove_digit() is for removing numbers that are fully made of digits."""
    pattern = re.compile('[0-9]+[^ ]*')
    newstring = pattern.sub(' ', string)
    newstring = re.sub('[ ]{2,}', ' ', newstring)
    return newstring


def replace_url(string):
    """Function replace_url() is for replacing URLs with URLLINK."""
    pattern = re.compile(r'http\S+')
    newstring = pattern.sub('URLLINK', string)
    return newstring


def replace_usermention(string):
    """Function replace_usermention() is for replacing @user_mentions with USERMENTION."""
    pattern = re.compile('@[A-Za-z0-9_]+')
    newstring = pattern.sub('USERMENTION', string)
    return newstring


def remove_repeat(string):
    """Function remove_repeat() is for replacing long same character with 2 character."""
    pattern = re.compile(r'(\w)(\1{2,})')
    newstring = pattern.sub(r'\1', string)
    return newstring


def lemmatize(string):
    """Function lemmatize() is for lemmatizing the strings."""
    wnl = nltk.WordNetLemmatizer()
    lemma_word = []
    for word in string.split(' '):
        lemma_word.append(wnl.lemmatize(word))
    return ' '.join(lemma_word)


def preprocess(data):
    new_text = []
    for string in data.text:
        newstring = string.lower()
        newstring = replace_usermention(newstring)
        newstring = replace_url(newstring)
        newstring = remove_nonalpha(newstring)
        newstring = remove_stopwords(newstring)
        newstring = remove_repeat(newstring)
        newstring = remove_digit(newstring)
        newstring = remove_character(newstring)
        newstring = lemmatize(newstring)
        new_text.append(newstring)
    data['new_text'] = new_text
    return data


def get_prediction(testset, y):
    prediction = {}
    for i in range(len(testset)):
        prediction.update({str(testset.id[i]):y[i]})
    return prediction


def get_mean(tweet):
    vec = np.zeros(300).reshape((1, 300))
    count = 0.
    for word in tweet.split():
        try:
            vec += model_w2v[word]
            count += 1.
        except KeyError:
            continue
    return vec / count


def get_meanvec(data):
    return np.concatenate([get_mean(tweet) for tweet in data])


def word_score(word):
    try:
        senti_word = list(swn.senti_synsets(word))[0]
        score = (-1) * senti_word.neg_score() + 0 * senti_word.obj_score() + 1 * senti_word.pos_score()
    except IndexError:
        score = 0.
    return score


def data_score(data):
    score = []
    for tweet in data:
        word_sc = 0.
        count = 0.
        for word in tweet.split():
            word_sc += word_score(word)
            count += 1.
        if count != 0:
            score.append(word_sc / count)
    return score


# Load training data
training_data = pd.read_csv('../semeval-tweets/twitter-training-data.txt', sep='\t', header=None)
training_data.columns = ['id', 'sentiment', 'text']
training_data = preprocess(training_data)

for classifier in ['Ngram + Naive Bayes', 'Word2Vec + Logistic Regression', 'Lexicon + Logistic Regression']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'Ngram + Naive Bayes':
        print('Training ' + classifier)
        # extract features for training classifier1
        # train sentiment classifier1
        clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), max_features=15000, stop_words='english')),
                          ('clf', MultinomialNB()), ])
        clf.fit(training_data.new_text, training_data.sentiment)
    elif classifier == 'Word2Vec + Logistic Regression':
        print('Training ' + classifier)
        # extract features for training classifier2
        # train sentiment classifier2
        model_w2v = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        clf = LogisticRegression(solver='newton-cg', multi_class='multinomial')
        train_x = np.nan_to_num(get_meanvec(training_data.new_text))
        clf.fit(train_x, training_data.sentiment)
    elif classifier == 'Lexicon + Logistic Regression':
        print('Training ' + classifier)
        # extract features for training classifier3
        # train sentiment classifier3
        clf = LogisticRegression(solver='newton-cg', multi_class='multinomial')
        train_x = np.array(data_score(training_data.new_text)).reshape(-1, 1)
        clf.fit(train_x, training_data.sentiment)

    for testset in testsets.testsets:
        # classify tweets in test set
        test_data = pd.read_csv(testset, sep='\t', header=None)
        test_data.columns = ['id', 'sentiment', 'text']
        test_data = preprocess(test_data)

        if classifier == 'Ngram + Naive Bayes':
            pre_test_data = clf.predict(test_data.new_text)
        elif classifier == 'Word2Vec + Logistic Regression':
            pre_test_data = clf.predict(np.nan_to_num(get_meanvec(test_data.new_text)))
        elif classifier == 'Lexicon + Logistic Regression':
            pre_test_data = clf.predict(np.array(data_score(test_data.new_text)).reshape(-1, 1))

        predictions = get_prediction(test_data, pre_test_data)

        evaluation.evaluate(predictions, testset, classifier)
        evaluation.confusion(predictions, testset, classifier)

print("This program takes ", time.time() - t1, "seconds")
