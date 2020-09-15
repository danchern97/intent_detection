#!/usr/bin/env python

from os import getenv
import json
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import logging
import sentry_sdk
from flask import Flask, request, jsonify

from models import MLP
from utils import train_model

sentry_sdk.init(getenv('SENTRY_DSN'))
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers
logger.setLevel(gunicorn_logger.level)

app = Flask(__name__)

sess = tf.compat.v1.Session()

logger.info('Creating model...')
encoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
data = json.load(open("data/full_dataset.json"))
train = data['alexa_prize']['train']['full']
valid = data['alexa_prize']['valid']
model = MLP(encoder, num_intents = len(train), multilabel=False, checkpointdir=True)
model = train_model(model, train=train, valid=valid, mode='mlp')
logger.info('Creating model... finished')

logger.info('Initializing tf variables...')
sess.run(tf.compat.v1.tables_initializer())

logger.info("Tables initialized")
sess.run(tf.compat.v1.global_variables_initializer())
logger.info("Global variables initialized")

test_sents = [['yes', 'no']]
print(f"Test_sents: {test_sents}, output: {model.predict(test_sents, sess)}")

logger.info("DONE")

@app.route("/detect", methods=['POST'])
def detect():
    utterances = request.json['sentences']
    logger.info(f"Number of utterances: {len(utterances)}")
    results = model.predict([utterances], sess)
    results = model.intent_order[np.argwhere(results)[:, 1]]
    results = list(results)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8014)
    sess.close()
