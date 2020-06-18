#!/usr/bin/env python3

import tensorflow as tf
from tqdm import tqdm
from xeger import Xeger
from sklearn.metrics import accuracy_score
import numpy as np
import models

def calculate_minus_accuracy(thresholds, probs, labels):
    preds = (probs > thresholds).astype(np.int)
    return -accuracy_score(preds, labels)

def batch_sentences(sentences, batch_size=64):
    """Simple batcher"""
    assert isinstance(sentences, list) or isinstance(sentences, np.array), \
        print("`sentences` type must be list or np.array")
    i = 0
    while i < len(sentences):
        yield sentences[i:i+batch_size]
        i+=batch_size

def generate_phrases(template_re, limit=20000):
    """
    Generate phrases from templare regexps
    """
    x = Xeger(limit=limit)
    phrases = []
    for regex in tqdm(template_re):
        try:
            phrases += list({x.xeger(regex) for _ in range(limit)})
        except Exception as e:
            print(e)
            print(regex)
            raise e
    return phrases


def cosine_similarity(a, b): # a: (data_size, dim) ; b: (batch_size, dim)
    numerator = a@tf.transpose(b) # data_size x batch_size
    denominator = tf.expand_dims(tf.reduce_sum(a*a, axis=1), axis=1)@tf.expand_dims(tf.reduce_sum(b*b, axis=1), axis=0)
    return tf.reduce_max(numerator/denominator, axis=0) # batch_size


def arccos_similarity(a, b): # a: (batch_size, dim) ; b: (batch_size, dim)
    return 1-tf.acos(cosine_similarity(a, b))/np.pi


def concatenate_encoders(encoder_a, encoder_b): # Concatenate two encoder outputs, basically - new encoder
    return lambda data: tf.compat.v1.concat([encoder_a(data), encoder_b(data)], axis=1)


def print_dataset_stat(data):
    for intent in data:
        print(f"{intent}:{len(data[intent])}")


def train_test_split(data, train_size=None, train_num=None):
    assert train_size or train_num, print("Neither `train_size` nor train_num have been provided")
    train, test = {}, {}
    for intent in data:
        if train_size: # Determine the true train_size
            size = int(train_size*len(data[intent]))
        else:
            size = train_num
        if len(data[intent]) < size: # Too little samples, everything goes to train
            train[intent] = data[intent]
            test[intent] = []
        else:
            train_idx = np.random.randint(low=0, high=len(data[intent]), size=size)
            test_idx = list(set(np.arange(start=0, stop=len(data[intent]))) - set(train_idx))
            train[intent] = list(np.array(data[intent])[train_idx])
            test[intent] = list(np.array(data[intent])[test_idx])
    return train, test

                            
def train_model(model, train, valid=None, mode='sim', batch_size=512, epochs=40): # TBD
    session = tf.compat.v1.Session()
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    if mode=='sim':
        model.fit(train, session, valid=valid, batch_size=batch_size)
    elif mode=='mlp':
        model.fit(train, session, valid=valid, epochs=epochs)
    else:
        raise Exception("Unknown mode")
    session.close()
    return model


def evaluate_model(model, test, batch_size=512):
    session = tf.compat.v1.Session()
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    preds, labels = [], []
    for i, intent in enumerate(model.intent_order):
        batched = batch_sentences(test[intent], batch_size)
        pred = model.predict(batched, session)
        label = np.zeros_like(pred)
        label[:, i] = 1.0
        preds.append(pred)
        labels.append(label)
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return accuracy_score(preds, labels)


def train_and_eval_model(model, train, valid, test, mode='sim'):
    # Similarity multilabel:
    train_model(model, train, valid, mode=mode)
    return evaluate_model(model, test)


similarity_measures = {
    "cosine": cosine_similarity, 
    "arccos": arccos_similarity
}