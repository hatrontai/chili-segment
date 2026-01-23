import numpy as np
import math
import cv2
from PIL import Image
from pyzbar.pyzbar import decode
from pathlib import Path


class Fruit(object):
    def __init__(self, image, bbox, polygons):
        self.image = image
        self.polygons = polygons
        self.width_bbox = 0
        self.height_bbox = 0
        self.width_fruit = 0
        self.length_fruit = 0
        self.area = 0
        self.wrinkle = 0

class Images(object):
    def __init__(self, path):
        self.path = path
        self.image = Image.open(self.path)
        w, h = img.size
        self.width_img = w
        self.height_img = h
        self.bboxs = []
        self.polygons = []
        self.num_fruit = 0

    def get_item(self, index):
        x, y, w, h = self.bboxs[index]
        img =  self.image
        fruit = Fruit(img, self.bboxs[index], self.polygons)

        return fruit



        