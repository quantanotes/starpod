from .api import API
from .model import Model

class Pod:
    def __init__(self, model: Model):
        self._model = model
        self._api = API(model)
 
        self._api.run()