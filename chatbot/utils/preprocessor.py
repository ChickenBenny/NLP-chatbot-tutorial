import nltk
import json
import numpy as np
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk.download('punkt')

class Preprocessor:

    def __init__(self):
        self.ps = PorterStemmer()

    def pipliine(self, file_name):
        intents = self._read_json(file_name)
        all_words = []
        tags = []
        x, y = [], []
        for intent in intents['intents']:
            tag = intent['tag']
            tags.append(tag)
            for pattern in intent['patterns']:
                tokenized_word = self._tokenize(pattern)
                stem_word = self._stem(tokenized_word)
                word = self._ignore_punc(stem_word)
                all_words.extend(word)
                x.append(word)
                y.append(tag)
        return all_words, tags, x, y

    def chatbot_msg_process(self, sentence):
        tokenized_word = self._tokenize(sentence)
        stem_word = self._stem(tokenized_word)
        word = self._ignore_punc(stem_word)   
        return word     

    def _read_json(self, file_name):
        with open(file_name, 'r') as f:
            return json.load(f)

    def _tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def _stem(self, words):
        for i in range(len(words)):
            words[i] = ps.stem(words[i].lower())
        return words

    def _ignore_punc(self, stem_words):
        punctuations = ["!", ",", ".", ":", "?"]
        sentence = []
        for word in stem_words:
            if word not in punctuations:
                sentence.append(word)
        return sentence

    def bag_of_words(self, sentence, all_words):
        bag = np.zeros(len(all_words), dtype = np.float32)
        for idx, w in enumerate(all_words):
            if w in sentence:
                bag[idx] = 1
        return bag

if __name__ == '__main__':
    prep = Preprocessor()
    all_word, tags, x, y = prep.pipliine('./assets/intents.json')
    print(all_word)
    print('------------------')
    print(tags)
    print('------------------')
    print(x)
    print('------------------')
    print(y)