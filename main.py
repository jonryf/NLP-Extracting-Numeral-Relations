import pandas as pd

from settings import EMBEDDING_DIM, NUM_WORKERS, SHUFFLE_DATA, BATCH_SIZE
from data_loader import get_loader
from model_runner import ModelRunner
from models import RelationEncoder, TweetEncoder, Classifier
from utils import get_device


def load_datasets():
    """
    Load the dataset

    :return: train, val and test dataset
    """

    build_vocab_from = "dataset/train.json"

    train = pd.read_json("dataset/train.json")
    val = pd.read_json("dataset/val.json")
    test = pd.read_json("dataset/test.json")

    train_dataset = get_loader(train, build_vocab_from, BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)
    val_dataset = get_loader(val, build_vocab_from, BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)
    test_dataset = get_loader(test, build_vocab_from, BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)

    return train_dataset, val_dataset, test_dataset

def run():
    train_dataset, val_dataset, test_dataset = load_datasets()
    vocabulary_size = len(train_dataset.dataset.vocab.wordToIndex)
    computing_device = get_device()

    encoder = RelationEncoder(vocabulary_size, EMBEDDING_DIM).to(computing_device)
    decoder = TweetEncoder(EMBEDDING_DIM, vocabulary_size).to(computing_device)
    classifier = Classifier().to(computing_device)

    runner = ModelRunner(encoder, decoder, classifier, train_dataset, val_dataset, test_dataset)

    runner.load_from_file()  # comment if you want to train from scratch
    # runner.train() # uncomment if you want to train from scratch
    runner.test()


if __name__ == '__main__':
    run()
