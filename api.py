import os
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml.model import CharRNN
from ml.generate import get_chapter, get_acrostic, get_horizontal_acrostic


# FastAPI app
app = FastAPI()


origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model_cp = torch.load(os.environ['MODEL_PATH'])
char_rnn = CharRNN(model_cp['tokens'], model_cp['n_hidden'], model_cp['n_layers'])
char_rnn.load_state_dict(model_cp['state_dict'])


@app.get("/ping")
def ping():
    return {"message": "pong!"}


@app.get("/models")
def list_models():
    return ['bible']


@app.get("/models/{model}/forms")
def list_models():
    return ['chapter', 'acrostic', 'horizontal_acrostic']


@app.get("/generate")
def generate(form: str = 'chapter', prime: str = 'א'):

    # default value
    out = ''

    # check form
    if form == 'chapter':

        # gen chapter
        out = get_chapter(char_rnn, prime, 40)

    elif form == 'acrostic':

        # gen acrostic
        out = get_acrostic(char_rnn, prime)

    elif form == 'horizontal_acrostic':

        # gen horizontal acrostic
        out = get_horizontal_acrostic(char_rnn, prime)

    return {
        "form": form,
        "verses": out.split('\n')
    }
