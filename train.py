import argparse

import torch
import numpy as np
from pathlib import Path
from torch import nn

from model import CharRNN
from common import one_hot_encode


# Defining method to make mini-batches for training
def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''

    batch_size_total = batch_size * seq_length

    # total number of batches we can make
    n_batches = len(arr) // batch_size_total

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]

    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):

        # The features
        x = arr[:, n:n + seq_length]

        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


# Declaring the train method
def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 

        Arguments
        ---------

        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    '''
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if torch.cuda.is_available():
        net.cuda()

    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):

        # initialize hidden state
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length).long())
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:

                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):

                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if torch.cuda.is_available():
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length).long())

                    val_losses.append(val_loss.item())

                net.train()  # reset to train mode after iterationg through validation data

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


if __name__ == "__main__":

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('corpus_path', type=Path)
    argparser.add_argument('--n_hidden', type=int, default=512)
    argparser.add_argument('--n_layers', type=int, default=2)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--seq_length', type=int, default=100)
    argparser.add_argument('--n_epochs', type=int, default=2)
    args = argparser.parse_args()

    # read train corpus
    text = args.corpus_path.read_text(encoding="utf-8")

    # encoding the text and map each character to an integer and vice versa
    # We create two dictionaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to integers
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # encode the text
    encoded = np.array([char2int[ch] for ch in text])

    # init model
    n_hidden = args.n_hidden
    n_layers = args.n_layers
    net = CharRNN(chars, n_hidden, n_layers)

    # declaring the hyperparameters
    batch_size = args.batch_size
    seq_length = args.seq_length
    n_epochs = args.n_epochs

    # train the model
    train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=50)

    # Saving the model
    model_name = f'rnn_{n_epochs}_epoch.net'

    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'tokens': net.chars}

    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)
