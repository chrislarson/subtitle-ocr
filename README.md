### Subtitle OCR using a CRNN

The supplied model is a Convolution Recurrent Neural Work trained to convert image-based subtitles into text-based
subtitles. This model was written and trained on a MacBook Pro, so the included instructions are based on a unix-like
environment. The tensorflow-metal was plugin was used for GPU acceleration. If the requirement cannot be satisfied on
your machine, use the `` `requirements_alt.txt` `` file during installation.

Requirements:

* Python 3.10.12
* Unix-like environment

Setup:

1. Create a virtual environment in the root of the repository.
   sh```python3 -m venv .venv

```

2. Activate the virtual environment.
   sh```source .venv/bin/activate```

3. Install dependencies (while in the activated virtual environment):
   sh```(.venv) pip install -r requirements.txt

```

Running:

- Word-level inference model:
  sh```(.venv) python3 words_inference.py```

* Line-level inference model:
  sh`` `(.venv) python3 lines_inference.py` ``

The inference model runs will open an image preview through OpenCV. Advance through the images by pressing any key.
