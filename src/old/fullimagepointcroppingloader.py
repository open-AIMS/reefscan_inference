from keras.utils import Sequence
import numpy as np
import time

class FullImagePointCroppingLoader(Sequence):

    def __init__(self, x_set, y_set, batch_size, load_image_func):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.load_image_func = load_image_func

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        indexes = []
        for index in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            indexes.append(index)

        #start = time.time()

        patches = [self.load_image_func(dict_item)
            for dict_item in batch_x]

        #end = time.time()
        #print("batch took", end - start)

        return np.array(patches), np.array(batch_y)#, indexes

        #return np.array([
        #    resize(imread(file_name), (200, 200))
        #    for file_name in batch_x]), np.array(batch_y)

