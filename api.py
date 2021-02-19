import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml.generate import get_chapter, get_acrostic, get_horizontal_acrostic
from ml.model_manager import ModelManager

# FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=[
                       '*',
                       'http://localhost:8080',
                       'http://localhost:3000'
                   ])

# model manager
mm = ModelManager()
mm.load_models(os.environ['MODEL_CONFIG_PATH'])


@app.get("/models")
def list_models():
    return mm.describe_models()


@app.get("/forms")
def list_models():
    return [
        {
            'id': 'chapter',
            'name': 'פרק'
        },
        {
           'id': 'acrostic',
           'name': 'אקרוסטיכון'
        },
        {
           'id': 'horizontal_acrostic',
           'name': 'אקרוסטיכון אופקי'
        }]


@app.get("/generate/{model}/{form}")
def generate(model: str, form: str):

    print(f'Generate: {model}/{form}')

    # get model
    mdl = mm.get_model(model)

    # check form
    if form == 'chapter':

        # gen chapter
        out = get_chapter(mdl, 'א', 20)

    elif form == 'acrostic':

        # gen acrostic
        out = get_acrostic(mdl, 'א')

    elif form == 'horizontal_acrostic':

        # gen horizontal acrostic
        out = get_horizontal_acrostic(mdl, 'א')

    return {
        "form": form,
        "verses": [dict(text=verse) for verse in out.split('\n')]
    }
