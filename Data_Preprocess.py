import pandas as pd
import numpy as np
import tensorflow as tf
import time
import re
import pickle
data=pd.read_excel("./data/news.xlsx")
data.drop(['Source ', 'Time ', 'Publish Date'], axis=1, inplace=True)
document=data['Short']
summary=data['Headline']
summary = summary.apply(lambda x: '<go> ' + x + ' <stop>')
filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'
document_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)
document_tokenizer.fit_on_texts(document)
summary_tokenizer.fit_on_texts(summary)
inputs = document_tokenizer.texts_to_sequences(document)
targets = summary_tokenizer.texts_to_sequences(summary)
encoder_vocab_size = len(document_tokenizer.word_index) + 1
decoder_vocab_size = len(summary_tokenizer.word_index) + 1
encoder_vocab_size = len(document_tokenizer.word_index) + 1
decoder_vocab_size = len(summary_tokenizer.word_index) + 1
document_lengths = pd.Series([len(x) for x in document])
summary_lengths = pd.Series([len(x) for x in summary])
encoder_maxlen = 400
decoder_maxlen = 75
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_maxlen, padding='post', truncating='post')
inputs = tf.cast(inputs, dtype=tf.int32)
argets = tf.cast(targets, dtype=tf.int32)
BUFFER_SIZE = 20000
BATCH_SIZE = 64
dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
