# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky
import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
# --- Helper Class for Metric Collection ---

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0.0
        self.correct_preds = 0
        self.total_samples = 0
        self.true_pos = 0
        self.total_pos = 0  # Actual positives in ground truth
        self.false_pos = 0
        self.true_neg = 0
        self.total_neg = 0  # Actual negatives in ground truth
        self.all_preds = []
        self.all_labels = []

    def update(self, loss, preds, func_labels):
        """
        Update metrics for a single batch.
        Args:
            loss (float): Batch loss.
            preds (torch.Tensor): Binary predictions (0 or 1) for the relevant window.
                                  Shape (batch_size, func_frame)
            labels (torch.Tensor): Original labels tensor (e.g., (batch_size, seq_len, num_actions))
            func_labels (torch.Tensor): Ground truth labels sliced to the relevant window.
                                        Shape (batch_size, func_frame)
        """
        self.loss += loss

        # Ensure preds and func_labels are flattened for metrics calculation
        preds_flat = preds.flatten()
        func_labels_flat = func_labels.flatten()

        self.correct_preds += (preds_flat == func_labels_flat).float().sum().item()
        self.total_samples += func_labels_flat.numel()  # Number of elements in the flattened tensor

        self.total_pos += (func_labels_flat == 1).sum().item()
        self.total_neg += (func_labels_flat == 0).sum().item()

        self.true_pos += torch.logical_and(preds_flat == 1, func_labels_flat == 1).float().sum().item()
        self.true_neg += torch.logical_and(preds_flat == 0, func_labels_flat == 0).float().sum().item()
        self.false_pos += torch.logical_and(preds_flat == 1, func_labels_flat == 0).float().sum().item()

        # Store for overall metrics if needed (e.g., for classification report)
        self.all_preds.append(preds_flat.cpu().numpy())
        self.all_labels.append(func_labels_flat.cpu().numpy())

    def get_metrics(self, num_samples, batch_size):
        """
        Calculate and return epoch-level metrics.
        Args:
            num_samples (int): Total number of samples in the dataset for this phase (e.g., len_train, len_val).
            batch_size (int): The batch size used.
        Returns:
            dict: A dictionary of calculated metrics.
        """
        avg_loss = self.loss * batch_size / num_samples  # Adjusted to be per sample

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-6

        accuracy = 100.0 * self.correct_preds / (self.total_samples + epsilon)
        recall_pos = 100.0 * self.true_pos / (self.total_pos + epsilon)  # Sensitivity
        recall_neg = 100.0 * self.true_neg / (self.total_neg + epsilon)  # Specificity
        precision = 100.0 * self.true_pos / (self.true_pos + self.false_pos + epsilon)  # Precision
        balanced_accuracy = (recall_pos + recall_neg) / 2.0
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'recall_pos': recall_pos,  # Sensitivity
            'recall_neg': recall_neg,  # Specificity
            'precision': precision,
            'balanced_accuracy': balanced_accuracy,
            'total_pos': self.total_pos,
            'total_neg': self.total_neg,
            'true_pos': self.true_pos,
            'false_pos': self.false_pos,
            'true_neg': self.true_neg,
        }
        return metrics

class PredMetricTracker:
    def __init__(self, device):
        self.valcorrect = 0
        self.val_loss = 0 # Not directly used in the provided snippet for calculation, but good to keep.
        self.total = 0
        self.true_pos = 0
        self.total_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.total_neg = 0
        self.predicted_arr = torch.Tensor(0).to(device)
        self.pred_label_arr = torch.Tensor(0).to(device)
        self.conf_arr = torch.Tensor(0).to(device)
        self.label_arr = torch.Tensor(0).to(device)
        self.lookface_arr = torch.Tensor(0).to(device) # This seems redundant if func_labels is always lookface
        self.smile_arr = torch.Tensor(0).to(device) # Placeholder, adjust if different actions are tracked
        self.socialsmile_arr = torch.Tensor(0).to(device) # Placeholder
        self.num_pred = 6 # Assuming this is a constant used elsewhere, like AL_predictions_arr
        # Universal counters for overall metrics
        self.o_valcorrect = 0
        self.o_total = 0
        self.o_total_pos = 0
        self.o_total_neg = 0
        self.o_true_pos = 0
        self.o_true_neg = 0
        self.o_false_pos = 0

        # Tensors to accumulate all predictions and true labels
        self.o_pred_label_arr = torch.tensor([], dtype=torch.float32)
        self.o_label_arr = torch.tensor([], dtype=torch.float32)
        self.o_clust_label_arr = torch.tensor([], dtype=torch.float32)
    def update(self, outputs_sliced, func_labels, conf):
        # pred_labels == 0 ---> look face, pred_labels == 1---> look object
        pred_labels = torch.where(outputs_sliced > 0.5, 1, 0)
        predicted = outputs_sliced

        self.predicted_arr = torch.cat((self.predicted_arr, torch.flatten(predicted).float()))
        self.pred_label_arr = torch.cat((self.pred_label_arr, torch.flatten(pred_labels).float()))
        self.label_arr = torch.cat((self.label_arr, torch.flatten(func_labels).float()))
        self.conf_arr = torch.cat((self.conf_arr, torch.flatten(conf > 0.7).float()))

        self.total += func_labels.flatten().numel()
        self.total_pos += (func_labels == 1).sum().item() # .item() to get scalar
        self.total_neg += (func_labels == 0).sum().item()

        self.valcorrect += ((torch.squeeze(pred_labels) == func_labels).sum().item())

        self.true_pos += torch.logical_and((torch.squeeze(pred_labels) == func_labels),
                                           (func_labels == 1)).float().sum().item()
        self.true_neg += torch.logical_and((torch.squeeze(pred_labels) == func_labels),
                                           (func_labels == 0)).float().sum().item()
        self.false_pos += torch.logical_and(torch.squeeze(pred_labels) == 1,
                                            (func_labels == 0)).float().sum().item()
        return self.label_arr.shape[0]
    def update_per_example_vid(self, clust):
        """
        Updates the universal counters and concatenates prediction/label arrays
        with new data from a single video or batch.

        Args:
            valcorrect (int): Number of correct predictions for the current batch/video.
            total (int): Total number of samples in the current batch/video.
            total_pos (int): Total positive samples in the current batch/video.
            total_neg (int): Total negative samples in the current batch/video.
            true_pos (int): True positives for the current batch/video.
            true_neg (int): True negatives for the current batch/video.
            false_pos (int): False positives for the current batch/video.
            pred_label_arr (torch.Tensor): Predicted labels for the current batch/video.
                                          Should be a 1D tensor of floats.
            label_arr (torch.Tensor): True labels for the current batch/video.
                                     Should be a 1D tensor of floats.
            clust (np.ndarray): Clustered labels for the current batch/video.
                                Should be a 1D NumPy array.
        """
        self.o_valcorrect += self.valcorrect
        self.o_total += self.total
        self.o_total_pos += self.total_pos
        self.o_total_neg += self.total_neg
        self.o_true_pos += self.true_pos
        self.o_true_neg += self.true_neg
        self.o_false_pos += self.false_pos

        # Ensure tensors are on CPU and float32 before concatenating
        self.o_pred_label_arr = torch.cat((self.o_pred_label_arr, self.pred_label_arr.float().cpu()))
        self.o_label_arr = torch.cat((self.o_label_arr, self.label_arr.float().cpu()))
        self.o_clust_label_arr = torch.cat((self.o_clust_label_arr, torch.from_numpy(clust).float().cpu()))

    def get_metrics(self):
        accuracy = self.valcorrect / self.total if self.total > 0 else 0
        precision = self.true_pos / (self.true_pos + self.false_pos) if (self.true_pos + self.false_pos) > 0 else 0
        recall = self.true_pos / self.total_pos if self.total_pos > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = self.true_neg / self.total_neg if self.total_neg > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'total_correct': self.valcorrect,
            'total_samples': self.total,
            'true_positives': self.true_pos,
            'false_positives': self.false_pos,
            'true_negatives': self.true_neg,
            'total_positives': self.total_pos,
            'total_negatives': self.total_neg,
            # You can also return the accumulated tensors if needed for further analysis
            'predicted_scores': self.predicted_arr,
            'predicted_labels': self.pred_label_arr.cpu(),
            'ground_truth_labels': self.label_arr.cpu(),
            'confidence_scores': self.conf_arr,
            # 'AL_predictions_arr': self.AL_predictions_arr # This was outside the loop and needs separate handling if it's aggregated across batches.
        }

    def reset_intravid_variables(self, device):
        """Resets all metrics to zero, useful for per-epoch or per-video tracking."""
        self.valcorrect = 0
        self.val_loss = 0
        self.total = 0
        self.true_pos = 0
        self.total_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.total_neg = 0
        self.predicted_arr = torch.Tensor(0).to(device)
        self.pred_label_arr = torch.Tensor(0).to(device)
        self.conf_arr = torch.Tensor(0).to(device)
        self.label_arr = torch.Tensor(0).to(device)

    def print_metrics(self):
        """
        Calculates and prints all the overall classification metrics,
        including accuracy, sensitivity, specificity, precision, weighted average accuracy,
        and then provides scikit-learn's classification report and ROC AUC score
        for both raw and clustered predictions.
        """
        print("\n" + "=" * 30)
        print("Overall Metrics Summary")
        print("=" * 30)

        # Basic metrics, handling potential division by zero
        accuracy = (100 * self.o_valcorrect / self.o_total) if self.o_total > 0 else 0
        sensitivity = (100 * self.o_true_pos / (self.o_total_pos + 1)) if self.o_total_pos > 0 else 0
        specificity = (100 * self.o_true_neg / (self.o_total_neg + 1)) if self.o_total_neg > 0 else 0
        precision = (100 * self.o_true_pos / (self.o_true_pos + self.o_false_pos)) if (
                                                                                                  self.o_true_pos + self.o_false_pos) > 0 else 0
        weighted_avg_accuracy = ((sensitivity + specificity) / 2) if (
                    self.o_total_pos > 0 or self.o_total_neg > 0) else 0

        print('Accuracy : %.2f %%' % accuracy)
        print('Sensitivity : %.2f %%' % sensitivity)
        print('Specificity : %.2f %%' % specificity)
        print('Precision : %.2f %%' % precision)
        print('Weighted average accuracy : %.2f %%' % weighted_avg_accuracy)
        print('Total %d, P %d, N %d' % (self.o_total, self.o_total_pos, self.o_total_neg))

        print("\n" + "-" * 30)
        print("Classification Report (Raw Predictions)")
        print("-" * 30)
        # Ensure labels are integers for classification_report if they represent classes
        # And predictions are also integer if they are hard predictions (0 or 1)
        # roc_auc_score typically expects probabilities or binary predictions
        try:
            print(classification_report(self.o_label_arr.long().numpy(), self.o_pred_label_arr.long().numpy()))
            auc = roc_auc_score(self.o_label_arr.numpy(), self.o_pred_label_arr.numpy())
            print('ROC AUC: %.4f' % auc)
        except ValueError as e:
            print(f"Could not generate classification report/ROC AUC for raw predictions: {e}")
            print("This often happens if there's only one class present in true labels or predictions.")

        print("\n" + "-" * 30)
        print("Classification Report (After Clustering)")
        print("-" * 30)
        try:
            print(classification_report(self.o_label_arr.long().numpy(), self.o_clust_label_arr.long().numpy()))
            auc_clustered = roc_auc_score(self.o_label_arr.numpy(), self.o_clust_label_arr.numpy())
            print('ROC AUC (clustered): %.4f' % auc_clustered)
        except ValueError as e:
            print(f"Could not generate classification report/ROC AUC for clustered predictions: {e}")
            print(
                "This often happens if there's only one class present in true labels or clustered predictions.")

        print("\n" + "=" * 30)
        print("Metrics Summary End")
        print("=" * 30 + "\n")

def pred_to_stats(array):
    # This function takes frame by frame ground truth or prediction
    # (clustered, ideally) and converts it to duration and frequency
    # The objective is to compare the DL model predictions with
    # groundtruth statistics (duration and frequency of events)

    # array: 1D input numpy array consisting of ones and zeros

    dur = np.sum(np.asarray(array))/30.0
    freq = np.size(np.where(np.diff(array, prepend=0, append=0) == 1))
    return dur, freq
