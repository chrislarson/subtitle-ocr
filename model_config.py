import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfig(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(
            "models/", datetime.strftime(datetime.now(), "%Y%m%d%H%M")
        )
        self.vocab = "!\"$%'(),-./0123456789:?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz"
        self.height = 32
        self.width = 128
        self.max_text_length = 21
        self.batch_size = 4
        self.learning_rate = 1e-4
        self.train_epochs = 5
        self.train_workers = 30
