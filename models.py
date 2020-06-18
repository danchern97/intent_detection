#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import tqdm
import tensorflow_hub as hub
import utils
from scipy import optimize
import datetime

class Classifier:
    """
    Base class for classifiers.
    """
    def __init__(self, encoder, multilabel=True):
        self.encoder = encoder
        self.multilabel = multilabel
        self.is_fitted  = False
        self.thresholds = []
    
    def fit(self, train, session, valid=None, batch_size=512):
        raise Exception("This is a base class for classifiers. Method is not implemented!")
    
    def predict_batch_proba(self, sentences, session):
        raise Exception("This is a base class for classifiers. Method should be implemented!")
    
    def predict_proba(self, batches, session):
        assert self.is_fitted
        probs = [self.predict_batch_proba(batch, session) for batch in batches]
        return np.concatenate(probs, axis=0)
    
    def predict(self, sentences, session):
        assert self.is_fitted
        probs = self.predict_proba(sentences, session)
        if not self.multilabel:
            preds = np.zeros_like(probs)
            preds[np.arange(preds.shape[0]), np.argmax(probs, axis=1)] = 1.0
        else:
            preds = (probs > self.thresholds).astype(np.int)
        return preds
        
        
class Similarity(Classifier):
    """
    Similarity-based model.
    """
    def __init__(self, encoder, multilabel=True, similarity='cosine'):
        super().__init__(encoder, multilabel)
        self.similarity = utils.similarity_measures[similarity]
        self.input = tf.compat.v1.placeholder(dtype=tf.string)
        self.intent_sentences = tf.compat.v1.placeholder(dtype=tf.float32)
        self.encode = self.encoder(self.input)
        self.output = self.similarity(self.intent_sentences, self.encode)
        self.data = {}
    
    def fit(self, train, session, valid=None, batch_size=512, fit_thresholds=True):
        self.intent_order = [intent for intent in train]
        for intent in self.intent_order:
            self.thresholds.append(0.5)
            self.data[intent] = session.run(self.encode, feed_dict={self.input:train[intent]})
        self.is_fitted = True
        if fit_thresholds and self.multilabel:
            assert valid, print("Valid is None, nothing to be used to fit thresholds")
            self.fit_thresholds(valid, session, batch_size)
        
    def fit_thresholds(self, valid, session, batch_size):
        """Find optimal thresholds in term of accuracy"""
        probs, labels = [], []
        for i, intent in enumerate(self.intent_order):
            batched = utils.batch_sentences(valid[intent], batch_size=batch_size)
            prob = self.predict_proba(batched, session)
            probs.append(prob)
            label = np.zeros_like(prob)
            label[:, i] = 1.0
            labels.append(label)
        probs = np.concatenate(probs, axis=0)
        labels = np.concatenate(labels, axis=0)
        opt = optimize.minimize(
            utils.calculate_minus_accuracy,
            x0=[0.8 for i in range(probs.shape[1])],
            args=(probs, labels), 
            bounds = [(0.0, 1.0) for i in range(probs.shape[1])],
            method='SLSQP',
            options={'eps':0.01}
        )
        self.thresholds = opt.x
        
    def predict_batch_proba(self, sentences, session):
        assert len(self.data), print("Model is not fitted!")
        probs = []
        for intent in self.data:
            prob = session.run(self.output, feed_dict={self.input: sentences, self.intent_sentences:self.data[intent]})
            probs.append(prob)
        probs = np.stack(probs, axis=1)
        return probs
    


class MLP(Classifier):
    """
    MLP upon encoder embeddings.
    """
    def __init__(self, encoder, multilabel=True, hidden_layers=0, 
                 hidden_dim=512, num_intents=22, metrics=['accuracy'], logdir='logs/', checkpointdir='checkpoints/'):
        super().__init__(encoder, multilabel)
        
        self.input = tf.compat.v1.placeholder(dtype=tf.string)
        self.encode = self.encoder(self.input)
        
        output_activation = 'sigmoid' if self.multilabel else 'softmax'
        loss = "categorical_crossentropy" # "binary_crossentropy" if self.multilabel else "categorical_crossentropy"
        encoder_dim = int(encoder(['test']).shape[1])
        
        model = [tf.keras.layers.Dense(units=hidden_dim, activation='relu', input_dim=encoder_dim if i == 0 else hidden_dim)
             for i in range(hidden_layers)]  # Hidden dense layers
        model += [tf.keras.layers.Dense(units=num_intents, activation=output_activation,
                                    input_dim=encoder_dim if not len(model) else hidden_dim)]  # Output layer
        model = tf.keras.Sequential(model)
        
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks = []
        if logdir: # Logging
            callbacks = [tf.keras.callbacks.TensorBoard(logdir+'log_'+time+'/', histogram_freq=1)]
        if checkpointdir: # Model saving
            self.checkpointdir = checkpointdir+'check_'+time+'/'
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpointdir,
                save_weights_only=True,
                monitor='val_acc',
                mode='max',
                save_best_only=True)
            callbacks += [cp_callback]
            
            
        self.callbacks = callbacks
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss,
            metrics=metrics
        )
        self.model = model
        
    def _prepare_dataset(self, data, session):
        x, y = [], []
        for intent in self.intent_order:
            embedd = session.run(self.encode, feed_dict={self.input: data[intent]})
            x.append(embedd)
            label = [[int(t==intent) for t in data] for j in range(embedd.shape[0])]
            y.append(label)
        x, y = np.concatenate(x, axis=0), np.concatenate(y, axis=0)
        np.random.shuffle(x), np.random.shuffle(y)
        return x, y
    
    def fit(self, train, session, valid=None, epochs=80, batch_size=64):
        self.thresholds = [0.5]*len(train)
        self.intent_order = [intent for intent in train]
        x_train, y_train = self._prepare_dataset(train, session)
        if valid:
            valid = self._prepare_dataset(valid, session)
        self.model.fit(x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=valid,
            callbacks=self.callbacks)
        self.model.load_weights(self.checkpointdir)
        self.is_fitted = True
        
    def predict_batch_proba(self, sentences, session):
        embedded = session.run(self.encode, feed_dict={self.input: sentences})
        return self.model.predict_proba(embedded)
    