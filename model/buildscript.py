from tensorflow.keras.utils import get_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
from numpy import array
import random
import sys
import io
import requests
import re
import json

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def save_doc(lines, filename,delimiter = '\n'):
    data = delimiter.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

from pickle import load
tokenizer = load(open('model/MasterTokenizer.pkl', 'rb'))


seq_to_word = dict(map(reversed, tokenizer.word_index.items()))

def decode(seq):
    seq = list(seq)
    #print(seq)
    sentence = [seq_to_word.get(index) for index in seq]
    return sentence

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


jerrymodel = load_model('model/jerry_model.h5')
georgemodel = load_model('model/george_model.h5')
elainemodel = load_model('model/elaine_model.h5')
kramermodel = load_model('model/kramer_model.h5')


seq_length = 25


jerry_i = tokenizer.word_index["jerry:"]
george_i = tokenizer.word_index["george:"]
elaine_i = tokenizer.word_index["elaine:"]
kramer_i = tokenizer.word_index["kramer:"]
other_i = tokenizer.word_index["other:"]

def predict_text(model,tokenizer,seed_in,temperature,n_words,seq_length,char_index):

    full_sequence = seed_in

    for _k in range(n_words):

        seed_in = pad_sequences([seed_in], maxlen=seq_length, truncating='pre')

        preds = model.predict(seed_in)[0]
        next_token = sample(preds,temperature)

        if next_token == tokenizer.word_index["other:"]:
            break
        elif next_token == char_index:
            break

        full_sequence = np.append(full_sequence,next_token)
        seed_in = np.append(seed_in,next_token)
        seed_in = np.delete(seed_in,0)


    
    return full_sequence


np.seterr(divide = 'ignore')


def populate(temp:float, req_script:[str]):

    characters = ['jerry','george','elaine', 'kramer']
    characters_encoded = [jerry_i,george_i,elaine_i,kramer_i]
    character_dex = 0

    script_seq = np.empty(0)


    for name in req_script:

        character_dex = characters.index(name)
        current_char = characters[character_dex]
        script_seq = np.append(script_seq,characters_encoded[character_dex])
        
        

        if current_char == characters[0]:
            script_seq = predict_text(jerrymodel,tokenizer,script_seq,temp,500,seq_length,jerry_i)
        elif current_char == characters[1]:
            script_seq = predict_text(georgemodel,tokenizer,script_seq,temp,500,seq_length,george_i)
        elif current_char == characters[2]:
            script_seq = predict_text(elainemodel,tokenizer,script_seq,temp,500,seq_length,elaine_i)
        else:
            script_seq = predict_text(kramermodel,tokenizer,script_seq,temp,500,seq_length,kramer_i)


    finished_script = ' '.join(decode(script_seq)).split("\n")
    finished_script.pop()

    response = json.dumps({
        "script":finished_script
    })

    return response