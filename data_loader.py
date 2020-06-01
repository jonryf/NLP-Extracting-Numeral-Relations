import pickle

import nltk
import torch
from torch.utils.data.dataset import Dataset

from word_dict import WordDict


class FinNumDataset(Dataset):
    def __init__(self, data, json, vocab_threshold=10):
        self.data = data
        self.vocab = get_word_dict(json, vocab_threshold)

    def __getitem__(self, index):
        """
        Fetch next item

        :param index: data index
        :return: relation, tweet and label
        """
        vocab = self.vocab

        item = self.data.iloc[index]

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(item['tweet']).lower())
        text = []
        text.append(vocab('<start>'))
        text.extend([vocab(token) for token in tokens])
        text.append(vocab('<end>'))
        text = torch.Tensor(text)

        # Convert caption (string) to word ids.
        target = []
        #        target.append(vocab('<start>'))
        target.append(vocab(str(item['target_cashtag'])))
        target.append(vocab(str(item['target_num'])))
        #        target.append(vocab(str(item['relation'])))
        # target.append(vocab('<end>'))
        target = torch.Tensor(target)
        # target = self.one_hot(item['gender'])

        return text, target, torch.tensor(item['relation'])

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (image, caption).
    """

    # Sort a data list by tweet length (descending order).
    #    data.sort(key=lambda x: len(x[1]), reverse=True)
    texts_, targets_, relations = zip(*data)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(text) for text in texts_]
    texts = torch.zeros(len(texts_), max(lengths)).long()
    for i, text in enumerate(texts_):
        end = lengths[i]
        texts[i, :end] = text[:end]

    lengths_targets = [len(text) for text in targets_]
    targets = torch.zeros(len(targets_), max(lengths_targets)).long()
    for i, text in enumerate(targets_):
        end = lengths_targets[i]
        targets[i, :end] = text[:end]
    return targets, lengths, texts, torch.tensor(relations).view(-1)


def get_loader(data, json, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    dataset = FinNumDataset(data, json)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


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
