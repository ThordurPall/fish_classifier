class Hyperparameters:
    def __init__(
        self,
        config={
            "batch_size": 64,
            "num_classes": 9,
            "image_height": 128,
            "image_width": 128,
            "rgb": 3,
            "filter1_in": 3,
            "filter1_out": 6,
            "kernel": 2,
            "pool": 2,
            "filter2_out": 16,
            "filter3_out": 48,
            "fc_1": 120,
            "fc_2": 84,
            "pad": 0,
            "stride": 1,
            "lr": 0.001,
            "epochs": 30,
            "activation": "relu",
            "dropout_p": 0.0,
        },
    ):
        super().__init__()
        self.config = config

    def config(self):
        return self.config
