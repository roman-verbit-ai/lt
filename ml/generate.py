import string
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from ml.model import CharRNN
from ml.common import one_hot_encode


# Defining a method to generate the next character
def predict(net, char, h=None, top_k=None):
    """
    Given a character, predict the next character.
    Returns the predicted character and the hidden state.
    """

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
def sample(net, size, prime, top_k=None):
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


# ============== #
# Output Formats #
# ============== #
def get_chapter(nnet, prime, n_verses=40):

    # get text
    text = sample(nnet, 2000, prime=prime, top_k=5)

    # split to verses
    verses = text.split('\n')
    verses = verses[:-1]    # drop last verse, it's usually incomplete

    # return num verses
    return '\n'.join(verses[:n_verses])


def get_acrostic(nnet, prime):

    # result text
    verses = list()
    words = set()

    # iterate num verses
    for c_idx in range(len(prime)):

        # try to get word which begins with current char
        cur_prime = next((w for w in words if w.startswith(prime[c_idx])), prime[c_idx])

        # get verse
        good_verse = False
        while not good_verse:
            verse = sample(nnet, 100, prime=cur_prime, top_k=2 + c_idx)
            good_verse = len(verse) > 20

        # index words
        words.update({w.strip(string.punctuation) for w in verse.split()})

        # slice sample
        stichon = verse[:verse.find('\n')]

        # # format sample
        # first_space_idx = stichon.find(' ')
        # if first_space_idx > 0 and stichon[first_space_idx - 1] not in string.punctuation:
        #     stichon = stichon.replace(' ', ' - ' if c_idx % 2 else ', ', 1)

        # add verse to list
        verses.append(stichon)

    return '\n'.join(verses)


def get_horizontal_acrostic(nnet, prime):

    # result text
    verses = list()

    # iterate num verses
    for c_idx in range(len(prime)):

        # try to get word which begins with current char
        cur_prime = prime[:c_idx + 1]

        # get verse
        verse = sample(nnet, 100, prime=cur_prime, top_k=2 + c_idx)

        # slice sample
        stichon = verse[:verse.find('\n')]

        # add verse to list
        verses.append(stichon)

    return '\n'.join(verses)


if __name__ == "__main__":

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model_path', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, required=True)
    argparser.add_argument('-l', '--predict_len', type=int, default=1000)

    # Output
    argparser.add_argument('-n', '--n_verses', type=int)
    argparser.add_argument('-f', '--format', type=str, default='stanza',
                           choices=['chapter', 'acrostic', 'horizontal_acrostic'])

    args = argparser.parse_args()

    # load checkpoint
    cp = torch.load(args.model_path)

    # init model
    char_rnn = CharRNN(cp['tokens'], cp['n_hidden'], cp['n_layers'])
    char_rnn.load_state_dict(cp['state_dict'])

    # generate
    out = ''
    if args.format == 'chapter':
        out = get_chapter(char_rnn, args.prime_str, args.n_verses)
    elif args.format == 'acrostic':
        out = get_acrostic(char_rnn, args.prime_str)
    elif args.format == 'horizontal_acrostic':
        out = get_horizontal_acrostic(char_rnn, args.prime_str)

    print(out)
