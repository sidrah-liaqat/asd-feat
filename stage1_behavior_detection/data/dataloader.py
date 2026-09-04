# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky
from __future__ import print_function, division
import os
import torch
import pandas as pd

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from random import shuffle
from pathlib import Path
import time

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_params, paths, split=None, transform=None, srt_idx=0, end_idx=16, func_frame=16, testind=0):
        """
        Args:
        data_params: A dictionary containing data-related parameters.

                     - 'fold': the data fold for cross-validation or specific dataset partitioning.
                     - 'subfold': a sub-fold within the main fold
                     - 'debug_mode': In debug mode, smaller subset of data loaded
                     - 'offset_debug': (Used in debug_mode) An integer specifying the starting index
                                       of the file list to load when in debug mode.
                     - 'example_count_debug': (Used in debug_mode) An integer specifying the number
                                              of files to load when in debug mode.
                     - 'frame_window_size': An integer representing the number of frames per window
                                            or chunk, used for processing and potentially dropping
                                            I3D features.
        paths: various file path configurations.
               - 'parquet_features_dir': location of landmark data in parquet format
               - 'i3d_features_dir': The base directory path where I3D (Inflated 3D ConvNet) features
                                     (both RGB and optical flow) are stored.
                                     Contains'rgb/' and 'flow/' subdirectories.

        split (str, optional): A string indicating the dataset split to use.
                               Common values would be 'train', 'val' (validation), or 'test'.
                               Defaults to None, but typically set.

        transform: Applies preprocessing transformation to the data

        srt_idx (int, optional): Start index for slicing features when predictions collected in sliding window mode
                                 Defaults to 0
        end_idx (int, optional): End index for slicing features when predictions collected in sliding window mode.
                                 Defaults to 16.

        func_frame (int, optional): A parameter related to the functional frame or window size,
                                    possibly used in later processing or for defining a specific
                                    temporal window. Defaults to 16.

        testind (int, optional): An integer index used specifically when `split` is 'test'.
                                 It indicates which single file from the test set list should be loaded.
                                 This allows for loading and evaluating
        """
        self._set_filename_lists(paths, data_params.fold, data_params.subfold, split)
        self._set_feature_column_names()

        # print('# of {} set files: {}'.format(split,str(self.uselist.filename.shape[0])))
        if (split == 'test'):
            starttr = testind
            endtr = testind + 1
        elif (data_params.debug_mode):
            # in debug mode you can train on any random smaller subset of files
            # by default set to start of filename list
            # offset_debug: # only useful in debug mode, set index number of file to start training from
            # example_count_debug: #only useful in debug mode, set number of files to train on
            starttr = data_params['offset_debug']
            endtr = starttr + data_params['example_count_debug']
        else:
            starttr = 0
            endtr = len(self.uselist.filename)
            print('{} examples in split : {}'.format(len(self.uselist.filename), split))
        lmk_data_frames_list = []
        rgb_chunks_list = []
        flow_chunks_list = []
        self.i3d_features_rgb = np.empty((0, 1024), dtype=np.float64)
        self.i3d_features_flow = np.empty((0, 1024), dtype=np.float64)
        start_time = time.time()
        for i in range(starttr, endtr):
            my_file = Path(paths.parquet_features_dir + self.uselist.filename[i] + '.parquet')
            if my_file.is_file() == False:
                print('{} is not in the feature directory'.format(str(self.uselist.filename[i] + '.parquet')))
                continue

            # i3d rgb file exists check
            my_file = Path(paths.i3d_features_dir + 'rgb/' + self.uselist_i3d.filename[i] + '.npy')
            if my_file.is_file() == False:
                # print('{} is not in the i3d feature directory'.format(
                #    str(self.uselist_i3d.filename[i] + '.npy')))
                continue
            # i3d flow file exists check
            my_file = Path(paths.i3d_features_dir + 'flow/' + self.uselist_i3d.filename[0] + '.npy')
            if my_file.is_file() == False:
                # print('{} is not in the i3d feature directory'.format(str(self.uselist_i3d.filename[i].replace + '.npy')))
                continue

            temp_lmk = pd.read_parquet(paths.parquet_features_dir + self.uselist.filename[i] + '.parquet',
                                  columns=self.all_lmk_cols)


            dropping_ind = temp_lmk[temp_lmk.valid == 0].index
            temp_lmk.drop(temp_lmk[temp_lmk.valid == 0].index, inplace=True)


            temp_i3d_rgb = np.load(paths.i3d_features_dir + 'rgb/' + self.uselist_i3d.filename[i] + '.npy')
            temp_i3d_flow = np.load(paths.i3d_features_dir + 'flow/' + self.uselist_i3d.filename[i] + '.npy')


            ofst = dropping_ind[0]
            ctr = 0
            for ind, val in enumerate(dropping_ind):
                # count the increasing values until 'frame'
                # for every 'frame' number of values, drop a row in self.i3d_features
                if (ctr + ofst) == val:
                    ctr += 1

                    if ctr % data_params.frame_window_size == 0:
                        temp_i3d_rgb = np.delete(temp_i3d_rgb, 0, axis=0)
                        temp_i3d_flow = np.delete(temp_i3d_flow, 0, axis=0)
                        # print('One row deleted {}'.format(ctr))
                # else:
                #    ofst = val
                #    ctr = 1
            L = temp_lmk.shape[0] // data_params.frame_window_size

            lmk_data_frames_list.append(temp_lmk.iloc[:L * data_params.frame_window_size])
            rgb_chunks_list.append(temp_i3d_rgb[:L])
            flow_chunks_list.append(temp_i3d_flow[:L])

        self.i3d_features_rgb = np.vstack(rgb_chunks_list)
        self.i3d_features_flow = np.vstack(flow_chunks_list)
        self.landmarks_frame = pd.concat(lmk_data_frames_list, ignore_index=True)
        self.landmarks_frame.reset_index(drop=True, inplace=True)
        self.testfilename = self.uselist.filename[testind]
        self.transform = transform
        self.func_frame = func_frame
        self.srt_idx = srt_idx
        self.end_idx = end_idx
        self.filesize = self.landmarks_frame.shape[0]

    def _set_filename_lists(self, paths, fold, subfold, split ):
        fname_dir = {'train': 'behaviortrain', 'val': 'test', 'asdtrain': 'asdtrain', 'test': 'test'}
        subfold_dir = {'train': '', 'val': '_subfold{}'.format(subfold),
                       'test': '_subfold{}'.format(subfold), 'asdtrain': '_subfold{}'.format(subfold)}

        basetemp = pd.read_csv(paths.data_splits_dir + 'standard_filenames/fold{}/{}_fold{}{}.txt'.format(
            fold, fname_dir[split], fold, subfold_dir[split]),
                               header=None)
        self.uselist = pd.DataFrame(columns=['filename'])
        self.uselist.filename = basetemp
        self.uselist.dropna(inplace=True)
        self.uselist.reset_index(drop=True, inplace=True)

        # In the released dataset every modality uses the same canonical filename
        # (<sub_id>_<visit>_<partner>), so the I3D file list is identical to the
        # landmark file list.
        self.uselist_i3d = self.uselist.copy()

    def _set_feature_column_names(self):
        """Defines all feature column names used for loading data."""
        self.label = ['look_face', 'look_object', 'smile', 'vocal']
        self.C_x_fc = [f'C_ x_{n}' for n in range(68)]
        self.C_y_fc = [f'C_ y_{n}' for n in range(68)]
        self.C_x_eye = [f'C_ eye_lmk_x_{n}' for n in range(56)]
        self.C_y_eye = [f'C_ eye_lmk_y_{n}' for n in range(56)]
        self.head_cols = ['tl_x', 'tl_y', 'br_x','br_y']
        # If it's 6 DOF headpose, update accordingly.
        # Your original code used 'headpose' for f1 (6DOF), but only 4 'head_cols' defined.
        # Assuming 6 columns for headpose: ['pitch', 'yaw', 'roll', 'tx', 'ty', 'tz'] or similar.
        # Please adjust self.head_cols to match your actual data columns.
        self.gaze_angle_cols = ['C_ gaze_angle_x',
                                'C_ gaze_angle_y']  # Not used in __getitem__ inputs but in original code
        self.toy_cols = ['toytlx', 'toytly', 'toybrx', 'toybry']  # Not used in __getitem__ inputs but in original code
        self.gaze_vector_cols = ['gaze0', 'gaze1', 'gaze2']  # Not used in __getitem__ inputs but in original code

        self.AU = [f'C_ AU{i:02d}_r' for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]] + \
                  [f'C_ AU{i:02d}_c' for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28,
                                               45]]  # AU28_c added as per your code's definition

        self.all_lmk_cols = ['frame', 'valid', 'C_ confidence'] + self.label + \
                            self.C_x_fc + self.C_y_fc + self.C_x_eye + self.C_y_eye + self.AU + \
                            self.head_cols + self.gaze_vector_cols + self.toy_cols

    def __len__(self):

        return int(np.ceil(self.landmarks_frame.shape[0] / self.func_frame))

    def __getitem__(self, cidx):

        if torch.is_tensor(cidx):
            cidx = cidx.tolist()
        idx = cidx * self.func_frame
        srt_idx = idx - self.srt_idx
        end_idx = idx + self.func_frame + self.srt_idx
        if srt_idx < 0:
            srt_idx += self.srt_idx
            end_idx += self.srt_idx
        if end_idx > self.landmarks_frame.shape[0]:
            diff = end_idx - self.landmarks_frame.shape[0]
            srt_idx -= diff
            end_idx -= diff

        # idx = starting index of the frame of interest
        # print('{}'.format(idx))
        # Have to pad ends for str_idx
        # in case idx is such that there aren't enough frames for this batch, shift idx back as needed
        # if (idx+self.win) > self.landmarks_frame.shape[0]: idx = self.landmarks_frame.shape[0]-self.win
        # idx: idx + self.win - 1 replaced with srt_idx:end_idx
        # there is no provision to load image in this code
        lmk_x = np.asarray(self.landmarks_frame.loc[srt_idx:end_idx - 1, self.C_x_fc], dtype=float)
        lmk_y = np.asarray(self.landmarks_frame.loc[srt_idx:end_idx - 1, self.C_y_fc], dtype=float)
        eye_lmk_x = np.asarray(self.landmarks_frame.loc[srt_idx:end_idx - 1, self.C_x_eye], dtype=float)
        eye_lmk_y = np.asarray(self.landmarks_frame.loc[srt_idx:end_idx - 1, self.C_y_eye], dtype=float)
        headpose = np.asarray(self.landmarks_frame.loc[srt_idx:end_idx - 1, self.head_cols], dtype=float)
        objectpose = np.asarray(self.landmarks_frame.loc[srt_idx:end_idx - 1, self.toy_cols], dtype=float)
        gazeangle = np.asarray(self.landmarks_frame.loc[srt_idx:end_idx - 1, self.gaze_vector_cols], dtype=float)
        conf = np.asarray(self.landmarks_frame.loc[srt_idx:end_idx - 1, 'C_ confidence'], dtype=float)

        rgb_feat = self.i3d_features_rgb[cidx]
        flow_feat = self.i3d_features_flow[cidx]

        au = np.asarray(self.landmarks_frame.loc[srt_idx:end_idx - 1, self.AU], dtype=float)
        # reshaping operations
        headpose = headpose.reshape(-1, 2)
        objectpose = objectpose.reshape(-1, 2)
        landmarks = np.vstack((lmk_x, lmk_y))
        landmarks = landmarks.transpose((1, 0))
        eyelandmarks = np.vstack((eye_lmk_x, eye_lmk_y))
        eyelandmarks = eyelandmarks.transpose((1, 0))

        # label for training
        sep_action = np.asarray(self.landmarks_frame.loc[srt_idx:end_idx - 1, self.label], dtype=int)

        sample = {
            'landmarks': landmarks,
            'eyelandmarks': eyelandmarks,
            'headpose': headpose,
            'objectpose': objectpose,
            'gazeangle': gazeangle,
            'au': au,
            'action': sep_action,
            'conf': conf,
            'rgb': rgb_feat,
            'flow': flow_feat
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
