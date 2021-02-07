import os
import torch
from fastapi import FastAPI

from ml.model import CharRNN
from ml.generate import get_chapter, get_acrostichon, get_horizontal_acrostichon


# FastAPI app
app = FastAPI()

# load model
model_cp = torch.load(os.environ['MODEL_PATH'])
char_rnn = CharRNN(model_cp['tokens'], model_cp['n_hidden'], model_cp['n_layers'])
char_rnn.load_state_dict(model_cp['state_dict'])


@app.get("/ping")
def ping():
    return {"message": "pong!"}


@app.get("/generate/{form}/{prime}")
def predict(form: str, prime: str):

    # default value
    out = ''

    # check form
    if form == 'chapter':

        # gen chapter
        out = get_chapter(char_rnn, prime, 40)

    elif form == 'acrostic':

        # gen acrostic
        out = get_acrostichon(char_rnn, prime)

    elif form == 'horizontal_acrostic':

        # gen horizontal acrostic
        out = get_horizontal_acrostichon(char_rnn, prime)

    return {"text": str(out)}
