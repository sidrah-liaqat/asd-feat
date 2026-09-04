# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky
from __future__ import print_function, division
from torch import optim, nn
from losses.losses import FocalLoss
from utils.metrics_utils import *

def prepare_features(sample_batch, device):
    """Move all feature tensors to the specified device."""

    features = {
        'feat_head': sample_batch['headpose'].float().to(device),
        'feat_lmk': sample_batch['landmarks'].float().to(device),
        'feat_eye': sample_batch['eyelandmarks'].float().to(device),
        'feat_au': sample_batch['au'].float().to(device),
        'rgb': sample_batch['rgb'].float().to(device),
        'flow': sample_batch['flow'].float().to(device)
    }
    return features

# not using sliding window with i3d features because we don't have the slided window version of them
# it is possible to generate the i3d features with sliding window

def training(model, device, config, train_loader, val_loader, len_train, len_val, srt_idx, end_idx):
    # Initializing Loss Function ---
    if config.training.loss_function == 'focal':
        # Assuming FocalLoss takes alpha and gamma
        criterion = FocalLoss(alpha=config.training.alpha, gamma=config.training.gamma)
    elif config.training.loss_function == 'bce_with_logits':
        # BCEWithLogitsLoss is appropriate when the model outputs raw logits
        criterion = nn.BCEWithLogitsLoss(reduction='none')  # reduction='none' because you apply manual weights later
    elif config.training.loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss()  # For multi-class classification where output is logits
    else:
        raise ValueError(f"Unsupported loss function: {config.training.loss_function}")

    # Initialize Optimizer along with Weight Decay
    if config.training.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=config.training.learning_rate,
                               weight_decay=config.training.weight_decay)
    elif config.training.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=config.training.learning_rate,
                              weight_decay=config.training.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")

    # Initializing Learning Rate Scheduler ---
    if config.training.scheduler.type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=config.training.scheduler.step_size,
                                              gamma=config.training.scheduler.gamma)
    # Add other schedulers as needed (e.g., ReduceLROnPlateau)
    elif config.training.scheduler.type == 'None':  # Or handle if no scheduler is desired
        scheduler = None  # Or just don't call scheduler.step()
    else:
        raise ValueError(f"Unsupported scheduler: {config.training.scheduler.type}")

    min_val_loss = float('inf')  # Initialize with a very large number for comparison
    for epoch in range(config.training.num_epochs):
        print(f'\n--- Epoch {epoch+1}/{config.training.num_epochs} ---')

        model.train()  # Set model to training mode
        train_tracker = MetricTracker()  # Reset metrics for current epoch's training phase

        for tr_batch, tr_sample_batch in enumerate(train_loader):

            # get features and labels
            labels = tr_sample_batch['action'].to(device)
            features = prepare_features(tr_sample_batch, device)
            target_labels_full = labels[:, :, config.action_id[config.data.behavior]].float()
            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(device, **features)

            loss_unweighted = criterion(torch.squeeze(output), target_labels_full)
            weight_matrix = torch.where(labels[:, :, config.action_id[config.data.behavior]] == 0,
                                        config.training.loss_weights[0],
                                        config.training.loss_weights[1])
            weight_matrix = weight_matrix / torch.sum(weight_matrix)
            losscls = torch.sum(torch.mul(loss_unweighted, weight_matrix))

            losscls.backward()
            optimizer.step()

            # Slice model output and true labels to the functional frame window
            outputs_sliced = torch.squeeze(output)[:, srt_idx:end_idx]
            labels_sliced = labels[:, srt_idx:end_idx, config.action_id[config.data.behavior]]
            pred_labels_binary = torch.where(outputs_sliced > 0.5, 1, 0)

            train_tracker.update(losscls.item(), pred_labels_binary, labels_sliced)

            # End of training phase for epoch
        train_metrics = train_tracker.get_metrics(len_train, config.data.batch_size)
        print(f"\nTraining Metrics (Epoch {epoch + 1}):")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.2f}%")
        print(f"  Sensitivity (Recall Pos): {train_metrics['recall_pos']:.2f}%")
        print(f"  Specificity (Recall Neg): {train_metrics['recall_neg']:.2f}%")
        print(f"  Precision: {train_metrics['precision']:.2f}%")
        print(f"  Balanced Accuracy: {train_metrics['balanced_accuracy']:.2f}%")
        train_tracker.reset()
        # update LR
        scheduler.step()
        ### validation loop
        model.eval()
        val_tracker = MetricTracker()  # Reset metrics for current epoch's validation phase

        with torch.no_grad():

            for val_batch, val_sample_batch in enumerate(val_loader):
                # 1. Prepare data
                labels = val_sample_batch['action'].to(device)
                features = prepare_features(val_sample_batch, device)

                # 2. Forward pass
                outputs = model(device, **features)

                # 3. Calculate loss
                target_labels_full = labels[:, :, config.action_id[config.data.behavior]].float()
                val_loss_unweighted = criterion(torch.squeeze(outputs), target_labels_full)
                weight_for_loss = torch.where(target_labels_full == 0,
                                              config.training.loss_weights[0],
                                              config.training.loss_weights[1])
                weight_for_loss = weight_for_loss / torch.sum(weight_for_loss)  # Normalize
                val_losscls = torch.sum(torch.mul(val_loss_unweighted, weight_for_loss))

                # 4. Collect metrics for validation
                outputs_sliced = torch.squeeze(outputs)[:, srt_idx:end_idx]
                labels_sliced = labels[:, srt_idx:end_idx, config.action_id[config.data.behavior]]
                pred_labels_binary = torch.where(outputs_sliced > 0.5, 1, 0)

                val_tracker.update(val_losscls.item(), pred_labels_binary, labels_sliced)

                # End of validation phase for epoch
            val_metrics = val_tracker.get_metrics(len_val, config.data.batch_size)
            print(f"\nValidation Metrics (Epoch {epoch + 1}):")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.2f}%")
            print(f"  Sensitivity (Recall Pos): {val_metrics['recall_pos']:.2f}%")
            print(f"  Specificity (Recall Neg): {val_metrics['recall_neg']:.2f}%")
            print(f"  Precision: {val_metrics['precision']:.2f}%")
            print(f"  Balanced Accuracy: {val_metrics['balanced_accuracy']:.2f}%")
            val_tracker.reset()
        # Model checkpoint

        val_loss_now = val_metrics['loss']
        if val_loss_now < min_val_loss:
            min_val_loss = val_loss_now
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None
            }
            # Use derived paths from config.paths
            torch.save(checkpoint, config.paths.model_chkpt_dir+config.data.behavior+
                       str(config.data.frame_window_size)+'.pth')
            print(
                f"Epoch {epoch + 1}: Saved best model with val loss {val_loss_now:.4f} to {config.paths.model_chkpt_dir}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None
        }
        torch.save(checkpoint, config.paths.model_final_dir+config.data.behavior+
                   str(config.data.frame_window_size)+'_finalstate.pth')
        print(f"Epoch {epoch + 1}: Saved current model to {config.paths.model_final_dir}")

    return val_metrics['loss'] # Return the final validation loss
