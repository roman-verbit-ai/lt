import argparse

import numpy as np

import torch
import torch.nn.functional as F

from common import one_hot_encode


# Defining a method to generate the next character
from model import CharRNN


def predict(net, char, h=None, top_k=None):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state.
    '''

    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if torch.cuda.is_available():
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])

    # get the output of the model
    out, h = net(inputs, h)

    # get the character probabilities
    p = F.softmax(out, dim=1).data
    if torch.cuda.is_available():
        p = p.cpu()  # move to cpu

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p / p.sum())

    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h


# Declaring a method to generate new text
def sample(net, size, prime='The', top_k=None):
    if torch.cuda.is_available():
        net.cuda()
    else:
        net.cpu()

    net.eval()  # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


if __name__ == "__main__":

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('checkpoint', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=1000)
    args = argparser.parse_args()

    # load checkpoint
    cp = torch.load(args.checkpoint)

    # init model
    nnet = CharRNN(cp['tokens'], cp['n_hidden'], cp['n_layers'])
    nnet.load_state_dict(cp['state_dict'])

    # do sample
    s = sample(nnet, args.predict_len, prime=args.prime_str, top_k=5)

    print(s)
