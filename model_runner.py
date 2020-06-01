import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from settings import EPOCHS, MODEL_NAME, MAX_LENGTH
from utils import plot_training_data, get_device

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score


class ModelRunner:
    def __init__(self, relation_encoder, tweet_encoder, classifier, train_dataset, val_dataset, test_dataset):
        self.relation_encoder = relation_encoder
        self.tweet_encoder = tweet_encoder
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.training_data = []
        weighted_loss = torch.tensor([0.65, 0.35])
        self.criterion = nn.CrossEntropyLoss(weight=weighted_loss)
        parameters = list(self.relation_encoder.parameters()) + \
                     list(self.tweet_encoder.parameters()) + \
                     list(self.classifier.parameters())
        self.optimizer = Adam(parameters, lr=0.00005)

    def train(self):
        """
        Train the network
        """
        for epoch in range(EPOCHS):
            print("Running epoch {}".format(epoch + 1))

            # train the model
            train_loss = self.pass_data(self.train_dataset, True)

            # run validation dataset
            val_loss, val_acc = self.val()
            self.training_data.append([train_loss, val_loss])
            print("Epoch: {}  -  training loss: {}, validation loss: {}, validation accuracy: {}".
                  format((epoch + 1), train_loss, val_loss, val_acc))

            # save model
            self.save_model(epoch)

        # plot training data when training is done
        plot_training_data(self.training_data)

    def val(self):
        """
        Run validation

        :return: loss
        """
        with torch.no_grad():
            return self.pass_data(self.val_dataset, False, True)

    def test(self):
        """
        Run test set

        :return: loss
        """
        print("Running test dataset:")
        with torch.no_grad():
            return self.pass_data(self.test_dataset, False, True)

    def save_model(self, epoch):
        """
        Saves the current state of the model to the model folder

        :param epoch: current epoch
        """
        model_path = "models/{}/".format(MODEL_NAME)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.relation_encoder, model_path + "relation encoder epoch {}".format(epoch + 1))
        torch.save(self.tweet_encoder, model_path + "tweet encoder epoch {}".format(epoch + 1))
        torch.save(self.classifier, model_path + "classifier epoch {}".format(epoch + 1))
        with open("scores {}.json".format(MODEL_NAME), 'w') as file:
            json.dump(self.training_data, file)

    def pass_data(self, dataset, backward, metrics=False):
        """
        Combined loop for training and data prediction, returns loss.

        :param dataset: dataset to do data passing from
        :param backward: if backward propagation should be used
        :param metrics: output metrics information
        :return: loss
        """
        if backward:
            self.relation_encoder.train()
            self.tweet_encoder.train()
            self.classifier.train()
        else:
            self.relation_encoder.eval()
            self.tweet_encoder.eval()
            self.classifier.eval()
        loss = 0
        acc = 0

        predicted = []
        labels = []

        for minibatch, (relation, lengths, tweets, relations) in tqdm(enumerate(dataset), total=len(dataset)):
            self.optimizer.zero_grad()

            encoder_hidden = self.relation_encoder.init_hidden()

            # move the data over to the correct device
            computing_device = get_device()
            relation = relation[0].to(computing_device)
            lengths = lengths[0]
            tweets = tweets[0].to(computing_device)

            # encode the relation data
            relation_encoder_outputs = torch.zeros(MAX_LENGTH, self.relation_encoder.hidden_size, device=get_device())
            for ei in range(2):
                encoder_output, encoder_hidden = self.relation_encoder(relation[ei], encoder_hidden)
                relation_encoder_outputs[ei] = encoder_output[0, 0]

            # encode the tweet data

            tweet_encoder_hidden = encoder_hidden
            tweet_encoder_output = torch.zeros(lengths, self.relation_encoder.hidden_size, device=get_device())

            for ei in range(lengths):
                output, tweet_encoder_hidden = self.tweet_encoder(tweets[ei], tweet_encoder_hidden,
                                                                  relation_encoder_outputs)
                tweet_encoder_output[ei] = output[0, 0]

            # classify the encoded data
            output = self.classifier(tweet_encoder_hidden, tweet_encoder_output[-1])
            batch_loss = self.criterion(output, relations)

            # backward
            if backward:
                batch_loss.backward()
                self.optimizer.step()

            loss += batch_loss.item()

            if metrics:
                values, indices = output.max(1)
                acc += ((indices[0] == relations[0]).float().sum())
                predicted.append(indices.numpy()[0])
                labels.append(relations.numpy()[0])

        loss /= (minibatch + 1)
        if metrics:
            acc /= (minibatch + 1)
            precision, recall, fscore, support = score(labels, predicted)
            f1score = f1_score(labels, predicted, average="macro")
            f1score_mico = f1_score(labels, predicted, average="micro")
            print('Precision: {}'.format(precision))
            print('Recall: {}'.format(recall))
            print('F1-score (macro): {}'.format(f1score))
            print('F1-score (micro): {}'.format(f1score_mico))
            print('support: {}'.format(support))
            print('Accuracy: {}'.format(acc))
            return loss, acc

        return loss

    def load_from_file(self):
        self.classifier = torch.load("models/{}/classifier epoch {}".format(MODEL_NAME, EPOCHS))
        self.relation_encoder = torch.load("models/{}/relation encoder epoch {}".format(MODEL_NAME, EPOCHS))
        self.tweet_encoder = torch.load("models/{}/tweet encoder epoch {}".format(MODEL_NAME, EPOCHS))
