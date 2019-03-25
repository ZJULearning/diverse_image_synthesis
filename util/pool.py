import random
import numpy as np
import torch
from torch.autograd import Variable

class ImagePool():
    def __init__(self):
        self.poolSize = 50
        if self.poolSize > 0:
            self.numImages = 0
            self.images = []

    def query(self, images):
        if self.poolSize == 0:
            return images
        returnImages = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.numImages < self.poolSize:
                self.numImages = self.numImages + 1
                self.images.append(image)
                returnImages.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    randomId = random.randint(0, self.poolSize-1)
                    tmp = self.images[randomId].clone()
                    self.images[randomId] = image
                    returnImages.append(tmp)
                else:
                    returnImages.append(image)
        returnImages = Variable(torch.cat(returnImages, 0))
        return returnImages
