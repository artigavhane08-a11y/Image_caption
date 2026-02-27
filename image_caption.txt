import os
print("current working folder:",os.getcwd())
print("Files in this folder:",os.listdir())

import warnings
warnings.filterwarnings("ignore")


from transformers import BlipProcessor,BlipForConditionalGeneration
from PIL import Image
import torch
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

image = Image.open("image.jpeg").convert("RGB")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_fast=False
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    output = model.generate(**inputs)

caption = processor.decode(output[0], skip_special_tokens=True)
print(caption)
