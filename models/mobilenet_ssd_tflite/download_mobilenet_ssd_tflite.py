import urllib.request
import zipfile

model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
urllib.request.urlretrieve(model_url, "mobilenet_model.zip")

with zipfile.ZipFile("mobilenet_model.zip", "r") as zip_ref:
    zip_ref.extractall("models")
