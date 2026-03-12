import sys
from PIL import Image

if "./lib" not in sys.path:
    sys.path.append("./lib")

from lib.ReadData import extractData

for i,tmp in enumerate( extractData("T1ce", "train", 32) ):
    img = tmp[0]
    img = Image.frombytes("RGB",(img['image_width'],img['image_height']),img['image'])
    img.save('tmp.jpg')
    break