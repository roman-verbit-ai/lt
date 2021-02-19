import json
import torch
from pathlib import Path

from ml.model import CharRNN


class ModelManager:

    def __init__(self):

        self._models = dict()

    def load_models(self, path):

        # parse
        path = Path(path)

        # load config
        config = json.loads(path.read_text())

        # iterate models
        for mdl_conf in config:

            # get path
            model_path = path.with_name(mdl_conf['file'])

            # load model
            model = self._load_model(model_path)

            # store
            self._models[mdl_conf['id']] = {
                'name': mdl_conf['name'],
                'model': model
            }

    def _load_model(self, path):

        # load
        model_cp = torch.load(path)  # os.environ['MODEL_PATH']
        char_rnn = CharRNN(model_cp['tokens'], model_cp['n_hidden'], model_cp['n_layers'])
        char_rnn.load_state_dict(model_cp['state_dict'])

        return char_rnn

    def describe_models(self):
        return [dict(id=m, name=self._models[m]['name']) for m in self._models.keys()]

    def get_model(self, model_id):
        return self._models[model_id]['model']
