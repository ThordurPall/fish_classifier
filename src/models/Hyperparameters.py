import json


class Hyperparameters:
    def __init__(self):
        super().__init__()
        f = open(
            "./src/hyperparameters.json",
        )
        self.config = json.load(f)

    def config(self):
        return self.config
