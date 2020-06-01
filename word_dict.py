import pickle
import nltk
from collections import defaultdict

import pandas as pd


class WordDict:
    def __init(self):
        self.indexToWord = {}
        self.wordToIndex = {}
        self.add_word_to_dict('<pad>')
        self.add_word_to_dict('<start>')
        self.add_word_to_dict('<end>')
        self.add_word_to_dict('0')
        self.add_word_to_dict('1')
        self.add_word_to_dict('<unk>')

    def clear(self):
        self.indexToWord = {}
        self.wordToIndex = {}
        self.add_word_to_dict('<pad>')
        self.add_word_to_dict('<start>')
        self.add_word_to_dict('<end>')
        self.add_word_to_dict('0')
        self.add_word_to_dict('1')
        self.add_word_to_dict('<unk>')

    def get_index_for_word(self, word):
        if word not in self.wordToIndex:
            return self.wordToIndex["<unk>"]
        return self.wordToIndex[word]

    def get_word_for_index(self, index):
        if index not in self.indexToWord:
            return "<unk>"
        return self.indexToWord[index]

    def __call__(self, item):
        if type(item) is str:
            return self.get_index_for_word(item)
        if type(item) is int:
            return self.get_word_for_index(item)
        return None

    def add_word_to_dict(self, word):
        if word not in self.wordToIndex:
            curIdx = len(self.wordToIndex)
            self.indexToWord[curIdx] = word
            self.wordToIndex[word] = curIdx

    def build_word_dict_from_words(self, words):
        for word in words:
            self.add_word_to_dict(word)

    def build_word_dict_from_json(self, json, threshold):
        nltk.download('punkt')

        data = pd.read_json(json)

        numberOccurences = defaultdict(lambda: 0)
        print("loading dictionary..........")
        for index, row in data.iterrows():
            text = row['tweet']
            tokens = nltk.tokenize.word_tokenize(text.lower())
            for token in tokens:
                numberOccurences[token] += 1

        allWords = []
        for word in numberOccurences:
            if numberOccurences[word] >= threshold:
                allWords.append(word)

        self.clear()
        self.build_word_dict_from_words(allWords)
        print("finished loading dictionary")


def get_word_dict(json, threshold, savePath="wordDictFile"):
    savePath += "threshold" + str(threshold)
    try:
        f = open(savePath, "rb")
        ret = pickle.load(f)
        print("Loaded wordDict from file %s" % savePath)
        return ret
    except:
        print("Could not load wordDict from file %s. Loading from json" % savePath)

    ret = WordDict()
    ret.build_word_dict_from_json(json, threshold)
    f = open(savePath, "wb")
    pickle.dump(ret, f)
    return ret
