#/usr/bin/env python3

import os
import json
import telebot
import logging
from utils import sent_tokenize
import requests

# Define logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

# Loading bot config
config = json.load(open("config.json"))
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 8014))
url = f"http://0.0.0.0:{SERVICE_PORT}/detect"
# Creating bot
bot = telebot.TeleBot(config['token'])
# Loading data and fitting the model


test_sents = ['yes', 'no']
response = requests.post(url, json={'sentences':test_sents}).json()
print(f"Test_sents: {test_sents}, output: {response}")
logging.info("READY!!!")

@bot.message_handler(commands=['start', 'help', 'show_classes'])
def handle_start_help(message):
    logging.info(f"Chat id: {message.chat.id} | Message: {message.text}")
    if message.text == '/start':
        bot.send_message(message.chat.id, config['start_message'])
    elif message.text == '/help':
        bot.send_message(message.chat.id, config['help_message'])
    elif message.text == '/show_classes':
        classes = "\n".join(config['classes'])
        bot.send_message(message.chat.id, classes)

@bot.message_handler(content_types=["text"])
def categorize_message(message): # Название функции не играет никакой роли
    utterances = sent_tokenize(message.text)
    response = requests.post(url, json={'sentences':utterances}).json()
    reply = "\n".join([f"\"{u}\" - {r}" for u, r in zip(utterances, response)])
    logging.info(f"Chat id: {message.chat.id} | Utterances: {utterances} | Reply: {reply}")
    bot.send_message(message.chat.id, reply)

if __name__ == '__main__':
     bot.infinity_polling()
