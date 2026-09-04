# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky
from __future__ import print_function, division
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from sklearn.cluster import DBSCAN
#from src.utils.metrics_utils import *
import torch
import numpy as np
import csv
from data.dataloader import FaceLandmarksDataset
from utils.metrics_utils import PredMetricTracker, pred_to_stats
from utils.transform import ToTensor, RescaleLandmarks_test
def load_inference_data(config, testind, srt_idx, end_idx, func_frame):

    face_dataset = FaceLandmarksDataset(data_params=config.data,
                                       paths=config.paths,
                                       split=config.inference.split,
                                       transform=transforms.Compose([
                                           RescaleLandmarks_test(1, config.data.frame_window_size),
                                           ToTensor()]),
                                       srt_idx=srt_idx, end_idx=end_idx, func_frame=func_frame,
                                        testind = testind)
    loader_params_face = {'batch_size': config.data.batch_size,'shuffle': False, 'num_workers': 0, 'drop_last': False}
    face_loader = DataLoader(dataset=face_dataset, **loader_params_face)
    filename = face_dataset.testfilename
    return face_loader, face_dataset.filesize, filename

def prepare_features(sample_batch, device):
    """Moves all feature tensors to the specified device."""
    # This assumes 'action' and 'conf' are not features but labels/metadata
    features = {
        'feat_head': sample_batch['headpose'].float().to(device),
        'feat_lmk': sample_batch['landmarks'].float().to(device),
        'feat_eye': sample_batch['eyelandmarks'].float().to(device),
        'feat_au': sample_batch['au'].float().to(device),
        'rgb': sample_batch['rgb'].float().to(device),
        'flow': sample_batch['flow'].float().to(device)
    }
    return features
def predict(model, device, config, srt_idx, end_idx, func_frame):

    filename = config.paths.data_splits_dir + 'standard_filenames/fold{}/{}_fold{}_subfold{}.txt'.format(
        config.data.fold, config.inference.split, config.data.fold, config.data.subfold)
    filename_i3d = config.paths.data_splits_dir + 'nonstandard_filenames/fold{}/{}_fold{}_subfold{}.txt'.format(
        config.data.fold, config.inference.split, config.data.fold, config.data.subfold)

    checksize = pd.read_csv(filename, header=None)
    checksize.dropna(inplace=True)
    checksize.reset_index(drop=True, inplace=True)

    checksize_i3d = pd.read_csv(filename_i3d, header=None)
    checksize_i3d.dropna(inplace=True)
    checksize_i3d.reset_index(drop=True, inplace=True)

    data_rows_mse = []
    data_rows_results = []

    parquet_file_dir = config.paths['parquet_features_dir']
    i3d_file_dir = config.paths['i3d_features_dir']
    print(checksize.shape[0])

    for testidx in range(checksize.shape[0]):

        my_file = Path(parquet_file_dir + config.inference.split + '/' + checksize[0][testidx] + '.parquet')
        if my_file.is_file() == False:
            continue
            # print('File {} is not in the feature directory'.format(checksize[0][testidx] + '.csv'))
        elif Path(i3d_file_dir + 'rgb/' + checksize_i3d[0][testidx] + '.npy').is_file() == False:
            continue
            # print('RGB file not in i3d feature directory {}'.format(checksize_i3d[0][testidx] + '.npy'))
        elif Path(i3d_file_dir + 'rgb/' + checksize_i3d[0][testidx] + '.npy').is_file() == False:
            continue
            # print('Flow file not in i3d feature directory {}'.format(checksize_i3d[0][testidx] + '.npy'))

        else:
            # print(my_file)
            conf = pd.read_parquet(my_file, columns=['C_ confidence'])
            conf_score = np.sum(conf['C_ confidence'] / conf.shape[0])
            if conf_score < 0.3:
                print('Low confidence score {}, not reading file {}'.format(conf_score, checksize[0][testidx]))
            else:
                face_loader, len_data, filename = load_inference_data(config, testidx, srt_idx, end_idx, func_frame)

                # rnning inference on the model and generating performance metrics
                with (torch.no_grad()):

                    pred_tracker = PredMetricTracker(device)
                    for i, sample in enumerate(face_loader):

                        # print('Batch # {}, num of rows {} / {}'.format(i, i*BATCHSIZE*frame, face_dataset.landmarks_frame.shape[0]))
                        labels = sample['action'].to(device)
                        conf = sample['conf'].to(device)
                        features = prepare_features(sample, device)

                        for feat in features:
                            if features[feat].isnan().any():
                                print('nan found in input features')

                        outputs = model(device, **features)
                        outputs_sliced = outputs[:, srt_idx:end_idx]
                        func_labels = labels[:, srt_idx:end_idx, config.action_id[config.data.behavior]]
                        len_current_frame = pred_tracker.update(outputs_sliced, func_labels, conf[:, srt_idx:end_idx])

                        # when sliding_inference = True, the predictions can't just be stacked
                        # Have to align them for each new time step and average them.

                        # check for video file ended
                        # then print metrics for this file, save video, initialize new video

                        if (features['feat_head'].shape[0] < config.data.batch_size) or (len_current_frame > len_data):
                            if i == 0:
                                print('Size of feature file smaller than batch size')
                                continue
                            elif i > 1:
                                print('{}-EOF'.format(len_current_frame))

                                file_metrics = pred_tracker.get_metrics()
                                # Precompute repeated expressions
                                true_pos_percentage = int(100 * file_metrics['true_positives'] / (file_metrics['total_positives'] + 0.001))  # sensitivity
                                true_neg_percentage = int(100 * file_metrics['true_negatives'] / (file_metrics['total_negatives'] + 0.001))  # specificity
                                positive_predictive_value = int(
                                    100 * file_metrics['true_positives'] / (file_metrics['true_positives'] + file_metrics['false_positives'] + 0.001))  # precision
                                average_percentage = (file_metrics['true_positives'] + file_metrics['true_negatives']) / (file_metrics['total_samples'])  # accuracy

                                if config.inference.save_results:
                                    data_rows_results.append([
                                        filename,
                                        int(100 * average_percentage),
                                        true_pos_percentage,
                                        true_neg_percentage,
                                        positive_predictive_value
                                    ])

                            if config.inference.save_predictions:
                                # for saving predictions and ground truth labels
                                # Prepare the data in a list of rows
                                data_rows = []
                                for i in range(len(file_metrics.all_preds)):
                                    data_rows.append([i, file_metrics.all_pred[i].item(), file_metrics.all_labels[i].item()])

                            ## Using DBSCAN for filling in missing predictions
                            pred_posidx = np.transpose(np.asarray(np.where(file_metrics['predicted_labels'] == 1)))
                            if pred_posidx.shape[0] == 0:
                                pred_posidx = np.ones((1, 1))

                            ## Using DBSCAN for filling in missing predictions
                            pred_clustered_arr = DBSCAN(eps=35, min_samples=8, metric='euclidean', metric_params=None,
                                                        algorithm='auto', leaf_size=30, p=None, n_jobs=None).fit(
                                pred_posidx)
                            # pred_clustered_arr = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None).fit(pred_label_arr.cpu().reshape(-1,1))
                            clust_posidx = pred_clustered_arr.labels_
                            n_clusters = len(set(clust_posidx)) - (1 if -1 in clust_posidx else 0)
                            clust = np.zeros_like(file_metrics['predicted_labels'])

                            # process the cluster sets such that for each new set, make all the elements within that index range 1
                            pred_posidx = np.squeeze(pred_posidx, axis=1)
                            for i in range(n_clusters):
                                this_cluster = np.where(clust_posidx == i)
                                clust[pred_posidx[np.min(this_cluster)]:pred_posidx[np.max(this_cluster)]] = 1

                            print('GndTr: Dur: {:0.2f}, Freq: {}'.format(pred_to_stats(file_metrics['ground_truth_labels'])[0],
                                                                         pred_to_stats(file_metrics['ground_truth_labels'])[1]))
                            print('Clust: Dur: {:0.2f}, Freq: {}'.format(pred_to_stats(clust)[0], pred_to_stats(clust)[1]))

                            data_rows_mse.append([filename,
                                                  round(pred_to_stats(file_metrics['ground_truth_labels'])[0], 2),
                                                  round(pred_to_stats(file_metrics['ground_truth_labels'])[1], 2),
                                                  round(pred_to_stats(clust)[0], 2),
                                                  round(pred_to_stats(clust)[1], 2)
                                                  ])
                            if config.inference.save_clustered_predictions:
                                # for saving clustered predictions for pipeline
                                clust_data = []
                                for i in range(len(clust)):
                                    clust_data.append([str(i / 30.0), str(clust[i].item())])
                                with open(
                                        config.paths.prediction_output_dir + config.data.behavior + config.inference.split + '/' + filename + '.csv', 'a', newline='') as csvfile:
                                    writer = csv.writer(csvfile, delimiter=',',
                                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                                    writer.writerow(['time', config.data.behavior])
                                    writer.writerows(clust_data)

                            # updating the universal counters before zeroing the per video counters
                            pred_tracker.update_per_example_vid(clust)

                            pred_tracker.reset_intravid_variables(device)
                            break
                if config.inference.save_results:
                    with open(config.paths.metrics_dir + config.data.behavior + str(config.data.frame_window_size) + str(config.inference.split) + 'results.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(['filename', 'accuracy', 'sensitivity', 'specificity', 'precision'])
                        writer.writerows(data_rows_results)
                if config.inference.save_mse:
                    with open(config.paths.metrics_dir + config.data.behavior + str(config.data.frame_window_size) + str(config.inference.split) + 'mse.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(['filename', 'gt_dur', 'gt_fq', 'clust_dur', 'clust_fq'])
                        writer.writerows(data_rows_mse)

    pred_tracker.print_metrics()
