import typing
import os

import cv2
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer


class ImageWordPredictor(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # root_path = os.path.split(dir_path)[0]
    word_model_path = os.path.join("models", "crnn_words", "configs.yaml")
    df_path = os.path.join(dir_path, "models", "crnn_words", "val.csv")

    configs = BaseModelConfigs.load(os.path.join(dir_path, word_model_path))
    model = ImageWordPredictor(
        model_path=os.path.join(dir_path, configs.model_path), char_list=configs.vocab
    )
    df = pd.read_csv(df_path).dropna().values.tolist()

    accum_cer = []
    for image_path, label in tqdm(df[:10]):
        img_path = os.path.join(dir_path, image_path)
        image = cv2.imread(img_path)
        try:
            prediction_text = model.predict(image)
            cer = get_cer(prediction_text, label)
            print(
                f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, Character Error Rate (CER): {cer}"
            )
            image = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3))
            cv2.imshow(prediction_text, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            continue
        accum_cer.append(cer)
    print(f"Average CER per word: {np.average(accum_cer)}")
