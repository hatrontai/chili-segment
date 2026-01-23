import numpy as np
import math
import cv2
from PIL import Image
from pyzbar.pyzbar import decode
from pathlib import Path

class Images(object):
    def __init__(self, path):
        self.path = path
        img = Image.open(self.path)
        w, h = img.size
        self.width = w
        self.height = h
        self.bbox = []
        self.polygon = []
        self.num_fruit = 0

    def get_item:



        