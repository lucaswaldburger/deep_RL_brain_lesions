import logging

from PIL import Image

from lib.ReadData import extractData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

for i, tmp in enumerate(extractData(["T1ce"], "HGG", "train", 32)):
    img = tmp[0]
    img = Image.frombytes("RGB", (img['image_width'], img['image_height']), img['image'])
    img.save('tmp.jpg')
    break
