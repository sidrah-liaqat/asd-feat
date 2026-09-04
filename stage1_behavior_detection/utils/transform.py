# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import imageio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class RescaleLandmarks(object):
    # Only landmarks are rescaled here. This function is used for normalizing landmarks [0,1]
    """Rescale the landmark features in a sample to a given size.
    For rescaling the landmark features to an approximately [0,1] range

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, frame):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.frame = frame

    def __call__(self, sample):

        #image, landmarks, headpose = sample['image'], sample['landmarks'], sample['headpose']
        landmarks, eyelandmarks, headpose, objectpose, gazeangle, au, action, conf = \
            sample['landmarks'], sample['eyelandmarks'], sample['headpose'], sample['objectpose'], sample['gazeangle'], sample['au'], sample['action'], sample['conf']

        #h, w = image.shape[:2]
        h, w = 480, 720
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        if np.isnan(landmarks).any():
            print("point 1")
        # min and max, along x and y axis resp.
        minn = np.amin(landmarks, axis=0)
        maxx = np.amax(landmarks, axis=0)
        #if maxx.sum()== 0: maxx = np.ones_like(minn)
        maxx[np.where(maxx == 0)] = 1
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # lmk_x = lmk_x * (new_w / w)
        # lmk_y = lmk_y * (new_h / h)
        try:
            landmarks = (landmarks - minn)/(maxx-minn)
        except:
            print("pause here")
        landmarks = np.reshape(landmarks, (136, self.frame))
        landmarks = landmarks.transpose((1,0))
        if np.isnan(landmarks).any():
            print("point 2")
        #eyelandmarks = (eyelandmarks - minn) / (maxx - minn)
        #landmarks = landmarks * [new_w / w, new_h / h]
        headpose = headpose * [new_w / w, new_h / h]
        objectpose = objectpose * [new_w / w, new_h / h]
        #eyelandmarks = eyelandmarks * [new_w / w, new_h / h]

        #return {'image': image, 'landmarks': landmarks, 'headpose': headpose}
        return {'landmarks': landmarks, 'eyelandmarks':eyelandmarks, 'headpose': headpose, 'au':au,
                 'gazeangle':gazeangle, 'action': action, 'conf': conf}
class RescaleLandmarks_test(object):
    # Only landmarks are rescaled here. This function is used for normalizing landmarks [0,1]
    """Rescale the landmark features in a sample to a given size.
    For rescaling the landmark features to an approximately [0,1] range

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, frame):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.frame = frame
    def __call__(self, sample):

        landmarks, eyelandmarks, headpose, objectpose, gazeangle, au, action, conf, rgb, flow = \
            sample['landmarks'], sample['eyelandmarks'], sample['headpose'], sample['objectpose'], sample['gazeangle'], sample['au'], \
            sample['action'], sample['conf'], sample['rgb'], sample['flow']

        #h, w = image.shape[:2]
        h, w = 480, 720
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        landmarks = landmarks * np.hstack(((new_w / w)*np.ones((self.frame)), (new_h / h)*np.ones((self.frame))))
        landmarks = np.reshape(landmarks, (136, self.frame))
        landmarks = landmarks.transpose((1, 0))
        headpose = headpose * [new_w / w, new_h / h]
        headpose = np.reshape(headpose, (self.frame, 4))
        objectpose = objectpose * [new_w / w, new_h / h]
        objectpose = np.reshape(objectpose, (self.frame, 4))
        headpose = np.concatenate((headpose, objectpose), axis=1)
        headpose = np.concatenate((headpose, gazeangle), axis=1)
        eyelandmarks = eyelandmarks * np.hstack(((new_w /w)*np.ones((self.frame)), (new_h / h)*np.ones((self.frame))))
        eyelandmarks = np.reshape(eyelandmarks, (112, self.frame))
        eyelandmarks = eyelandmarks.transpose((1, 0))
        conf = conf[:, np.newaxis]
        #eyelandmarks = np.concatenate((eyelandmarks, conf), axis=1)
        #landmarks = np.concatenate((landmarks, conf), axis=1)
        #return {'image': image, 'landmarks': landmarks, 'headpose': headpose}
        return {'landmarks': landmarks, 'eyelandmarks':eyelandmarks, 'headpose': headpose, 'objectpose' : objectpose, 'au':au,
                 'gazeangle':gazeangle, 'action': action, 'conf' : conf, 'rgb' : rgb, 'flow' : flow}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        landmarks, eyelandmarks, headpose, objectpose, gazeangle, au, action, conf, rgb, flow = \
            sample['landmarks'], sample['eyelandmarks'], sample['headpose'],sample['objectpose'], sample['gazeangle'], sample['au'], \
            sample['action'], sample['conf'], sample['rgb'], sample['flow']

        # image, lmk_x, lmk_y = sample['image'], sample['lmk_x'], sample['lmk_y']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))

        # need to reshape the landmarks from 680,2 picking on this kind of path |/|/|/
        # to a final shape of 136, 10

        #return {'image': torch.from_numpy(image),
        return {
                'landmarks': torch.from_numpy(landmarks),
                'eyelandmarks': torch.from_numpy(eyelandmarks),
                'headpose' : torch.from_numpy(headpose),
                'objectpose' : torch.from_numpy(objectpose),
                'gazeangle': torch.from_numpy(gazeangle),
                'au' : torch.from_numpy(au),
                'action' : torch.from_numpy(action),
                'conf'  : torch.from_numpy(conf),
                'rgb' : torch.from_numpy(rgb),
                'flow' : torch.from_numpy(flow)
        }
