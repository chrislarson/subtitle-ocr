import os
import random

import tensorflow as tf
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from mltu.annotations.images import CVImage
from mltu.preprocessors import ImageReader
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CWERMetric
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from tqdm import tqdm

from model import crnn
from model_config import ModelConfig

data_path = "/data_processed/words"
val_annotation_path = data_path + "/annotation_val.txt"
train_annotation_path = data_path + "/annotation_test.txt"


# Read metadata file and parse it
def read_annotation_file(annotation_path):
    dataset, vocab, max_len = [], set(), 0
    with open(annotation_path, "r") as f:
        for line in tqdm(f.readlines()):
            line = line.split("  _ _  ")
            image_file = line[0]
            image_path = data_path + "/images/" + image_file
            label = "".join(line[1:]).strip()
            dataset.append([image_path, label])
            vocab.update(list(label))
            max_len = max(max_len, len(label))
    return dataset, sorted(vocab), max_len


train_dataset, train_vocab, max_train_len = read_annotation_file(train_annotation_path)
val_dataset, val_vocab, max_val_len = read_annotation_file(val_annotation_path)
random.shuffle(train_dataset)
random.shuffle(val_dataset)


def main(batchsize: int, epochs: int):
    configs = ModelConfig()

    # Save vocab and maximum text length to configs
    configs.max_text_length = max(max_train_len, max_val_len)
    configs.batch_size = batchsize
    configs.train_epochs = epochs
    configs.save()

    # Create training data_raw provider
    train_data_provider = DataProvider(
        dataset=train_dataset,
        skip_validation=False,
        batch_size=configs.batch_size,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(
                configs.width,
                configs.height,
                padding_color=(255, 255, 255),
            ),
            LabelIndexer(configs.vocab),
            LabelPadding(
                max_word_length=configs.max_text_length,
                padding_value=len(configs.vocab),
            ),
        ],
    )

    # Create validation data_raw provider
    val_data_provider = DataProvider(
        dataset=val_dataset,
        skip_validation=False,
        batch_size=configs.batch_size,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(
                configs.width,
                configs.height,
                padding_color=(255, 255, 255),
            ),
            LabelIndexer(configs.vocab),
            LabelPadding(
                max_word_length=configs.max_text_length,
                padding_value=len(configs.vocab),
            ),
        ],
    )

    model = crnn(
        input_dim=(configs.height, configs.width, 3),
        output_dim=len(configs.vocab),
    )
    # Compile the model and print summary
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=configs.learning_rate),
        loss=CTCloss(),
        metrics=[CWERMetric(padding_token=len(configs.vocab))],
        run_eagerly=False,
    )
    model.summary(line_length=110)

    # Define path to save the model
    os.makedirs(configs.model_path, exist_ok=True)

    # Define callbacks
    earlystopper = EarlyStopping(monitor="val_CER", patience=10, verbose=1)
    checkpoint = ModelCheckpoint(
        f"{configs.model_path}/model.h5",
        monitor="val_CER",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    trainLogger = TrainLogger(configs.model_path)
    tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
    reduceLROnPlat = ReduceLROnPlateau(
        monitor="val_CER",
        factor=0.9,
        min_delta=1e-10,
        patience=5,
        verbose=1,
        mode="auto",
    )
    model2onnx = Model2onnx(f"{configs.model_path}/model.h5", save_on_epoch_end=True)

    # Save training and validation datasets as csv files
    train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
    val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

    # Train the model
    model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=configs.train_epochs,
        callbacks=[
            earlystopper,
            checkpoint,
            trainLogger,
            reduceLROnPlat,
            tb_callback,
            model2onnx,
        ],
        workers=configs.train_workers,
        verbose=1,
    )


if __name__ == "__main__":
    # 3 batch, 10 epoch
    main(3, 10)
