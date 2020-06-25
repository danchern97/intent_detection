#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import tqdm
import tensorflow_hub as hub
import utils
from scipy import optimize
import datetime

tf.compat.v1.disable_eager_execution()

class Classifier:
    """
    Base class for classifiers.
    """
    def __init__(self, encoder, multilabel=True):
        self.encoder = encoder
        self.multilabel = multilabel
        self.is_fitted  = False
        self.thresholds = []
    
    def fit(self, train, session, valid=None, batch_size=256):
        raise Exception("This is a base class for classifiers. Method is not implemented!")
    
    def predict_batch_proba(self, sentences, session):
        raise Exception("This is a base class for classifiers. Method should be implemented!")
        
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
            x0=probs.mean(axis=0),
            args=(probs, labels), 
            bounds = [(0.0, 1.0) for i in range(probs.shape[1])],
            method='SLSQP',
            options={'eps':0.01}
        )
        self.thresholds = opt.x
    
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
        
    def _reinit(self):
        self.thresholds = []
        self.data = {}
        self.is_fitted = False
    
    def fit(self, train, session, valid=None, batch_size=256, fit_thresholds=True):
        if self.is_fitted:
            self._reinit()
        self.intent_order = [intent for intent in train]
        for intent in self.intent_order:
            self.thresholds.append(0.5)
            batched = utils.batch_sentences(train[intent], batch_size=batch_size)
            encoded = []
            for batch in batched:
                encoded.append(session.run(self.encode, feed_dict={self.input:batch}))
            self.data[intent] = np.concatenate(encoded, axis=0)
        self.is_fitted = True
        if fit_thresholds and self.multilabel:
            assert valid, print("Valid is None, nothing to be used to fit thresholds")
            self.fit_thresholds(valid, session, batch_size)
        
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
                 hidden_dim=512, num_intents=22, metrics=['accuracy'], logdir='logs/', checkpointdir=True):
        super().__init__(encoder, multilabel)
        
        self.input = tf.compat.v1.placeholder(dtype=tf.string)
        self.encode = self.encoder(self.input)
        
        output_activation = 'sigmoid' if self.multilabel else 'softmax'
        loss = "binary_crossentropy" if self.multilabel else "categorical_crossentropy"
        encoder_dim = int(encoder(['test']).shape[1])
        
        model = [tf.keras.layers.Dense(units=hidden_dim, activation='relu', input_dim=encoder_dim if i == 0 else hidden_dim)
             for i in range(hidden_layers)]  # Hidden dense layers
        model += [tf.keras.layers.Dense(units=num_intents, activation=output_activation,
                                    input_dim=encoder_dim if not len(model) else hidden_dim)]  # Output layer
        model = tf.keras.Sequential(model)
        
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks = []
#         if logdir: # Logging
#             callbacks += [tf.keras.callbacks.TensorBoard(logdir+'log_'+time+'/', histogram_freq=1)]
        if checkpointdir: # Model saving
            self.checkpointdir = 'checkpoints/check_'+time+'/'
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpointdir,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
            callbacks += [cp_callback]
            
            
        self.callbacks = callbacks
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss,
            metrics=metrics
        )
        self.nn = model
        self.initial_weights = model.get_weights()

    def _reinit(self, session):
        self.thresholds = []
        self.nn.set_weights(self.initial_weights)
        if self.checkpointdir:
            time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.checkpointdir = 'checkpoints/check_'+time+'/'
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpointdir,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
            self.callbacks = [cp_callback]
        self.is_fitted = False
        
    def _prepare_dataset(self, data, session, batch_size):
        x, y = [], []
        for i, intent in enumerate(self.intent_order):
            batched = utils.batch_sentences(data[intent], batch_size=batch_size)
            encoded = []
            for batch in batched:
                encoded.append(session.run(self.encode, feed_dict={self.input:batch}))
            encoded = np.concatenate(encoded, axis=0)
            x.append(encoded)
            label = np.zeros((encoded.shape[0], len(self.intent_order)))
            label[:, i] = 1.0
            y.append(label)
        x = np.concatenate(x, axis=0), 
        y = np.concatenate(y, axis=0)
        return x, y
    
    def fit(self, train, session, valid=None, epochs=30, batch_size=256):
        if self.is_fitted:
            self._reinit(session)
            
        self.thresholds = [0.5]*len(train)
        self.intent_order = [intent for intent in train]
        x_train, y_train = self._prepare_dataset(train, session, batch_size)
        if valid:
            x_valid, y_valid = self._prepare_dataset(valid, session, batch_size)
        
        self.nn.fit(x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_valid, y_valid),
            callbacks=self.callbacks, 
            verbose=0)
        self.nn.load_weights(self.checkpointdir)
        self.is_fitted = True
        if self.multilabel and valid:
            self.fit_thresholds(valid, session, batch_size)
        
        
    def predict_batch_proba(self, sentences, session):
        embedded = session.run(self.encode, feed_dict={self.input: sentences})
        return self.nn.predict_proba(embedded)
    