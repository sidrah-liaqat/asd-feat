# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky
import sys

import yaml
import os
import torch
import argparse
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from trainer import training
from inference import predict
from data.dataloader import FaceLandmarksDataset
from utils.transform import ToTensor, RescaleLandmarks_test
from torchvision import transforms
from models.model_factory import get_model, load_model
from utils.data_processing_utils import calculate_window_indices

def load_config(config_path="config/default.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = edict(yaml.safe_load(f))
        print(f"Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Run machine learning model.")

    # General configuration argument
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to the default YAML configuration file.')

    # Arguments that can override YAML settings
    parser.add_argument('--run', type=str,
                        help='Select run mode as training or inference.')
    parser.add_argument('--data.debug_mode', type=str,
                        help='Override: Set true for quicker run for debugging.')
    parser.add_argument('--data.behavior', type=str,
                        help='Override: Select behavior as training or inference.')
    parser.add_argument('--training.num_epochs', type=int,
                        help='Override: Set number of epochs for training.')

    #parser.add_argument('--data.batch_size', type=int,
    #                    help='Override: Batch size for data loading.')
    #parser.add_argument('--logging.level', type=str,
    #                    help='Override: Logging level (e.g., DEBUG, INFO, WARNING).')

    args = parser.parse_args()
    return args

# Function to apply CLI overrides to the YAML config
def apply_overrides(config, args):
    # Iterate through the parsed arguments (args)
    # and apply them to the config dictionary-like object.
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:  # Only apply if the argument was specified on CLI
            # Split argument name by dot to navigate nested YAML structure
            keys = arg_name.split('.')

            # Skip the 'config' argument itself
            if keys[0] == 'config':
                continue

            current_level = config
            for i, key in enumerate(keys):
                if i == len(keys) - 1:  # Last key is where the value should be set
                    current_level[key] = arg_value
                else:  # Navigate to the next level
                    if key not in current_level or not isinstance(current_level[key], dict):
                        # If a path segment doesn't exist or isn't a dict, create it as EasyDict
                        current_level[key] = edict()
                    current_level = current_level[key]
    return config

def load_data(data_params, paths, srt_idx, end_idx, func_frame):
    # specifications of dataloader, initialize dataloader i.e. load data
    # return two dataloaders
    loader_params_train = {'batch_size': data_params.batch_size, 'shuffle': True, 'num_workers': 0, 'drop_last': True}
    loader_params_val = {'batch_size': data_params.batch_size, 'shuffle': False, 'num_workers': 0, 'drop_last': True}
    train_dataset = FaceLandmarksDataset(data_params=data_params,
                                         paths=paths,
                                         split='train',
                                         transform=transforms.Compose([
                                             RescaleLandmarks_test(1, data_params.frame_window_size),
                                             ToTensor()]),
                                         srt_idx = srt_idx, end_idx= end_idx, func_frame = func_frame
                                         )
    train_loader = DataLoader(dataset=train_dataset, **loader_params_train)

    val_dataset = FaceLandmarksDataset(data_params=data_params,
                                       paths=paths,
                                       split='val',
                                       transform=transforms.Compose([
                                           RescaleLandmarks_test(1, data_params.frame_window_size),
                                           ToTensor()]),
                                       srt_idx=srt_idx, end_idx=end_idx, func_frame=func_frame
                                       )
    val_loader = DataLoader(dataset=val_dataset, **loader_params_val)

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def setup_environment(config):

    project_name = config['project']['name']
    random_seed = config['project']['random_seed']
    for directory in config['paths']:
        if not os.path.exists(config['paths'][directory]):
            # os.makedirs creates all necessary intermediate directories
            # exist_ok=True prevents an error if the directory already exists
            os.makedirs(config['paths'][directory], exist_ok=True)
            print(f"Created directory: {config['paths'][directory]}")
        else:
            print(f"Directory already exists: {config['paths'][directory]}")
    print(f"\n--- Setting up environment for {project_name} ---")
    print(f"Using random seed: {random_seed}")

def run_training(config):

    print("\n--- Starting Training {} behavior model ---".format(config.data.behavior))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model(config, device)
    srt_idx, end_idx, func_frame = calculate_window_indices(config.data.frame_window_size,
                                                            config.data.overlap_windows)
    train_loader, val_loader, len_train, len_val = load_data(config.data, config.paths, srt_idx, end_idx, func_frame)

    # Pass model to training function
    training(
        model=model,
        device=device,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        len_train=len_train,
        len_val = len_val,
        srt_idx=srt_idx,
        end_idx = end_idx
    )
    print("Training completed.")

def run_inference(config):

    print("\n--- Starting Inference ---")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model(config, device)
    model = load_model(model, config, device)
    model.eval()
    srt_idx, end_idx, func_frame = calculate_window_indices(config.data.frame_window_size,
                                                            config.data.overlap_windows)
    predict(model=model, device=device, config=config,
            srt_idx=srt_idx, end_idx = end_idx, func_frame = func_frame)

    print("Inference completed.")

if __name__ == "__main__":
    # Ensure the config directory exists for the example
    os.makedirs('config', exist_ok=True)
    # You would normally manually create default.yaml or have it part of your repo clone

    # Load the configuration
    app_config = load_config()
    apply_overrides(config=app_config, args=parse_args())
    # Use the flags
    setup_environment(app_config)
    device = torch.device(app_config.training.device if torch.cuda.is_available() else "cpu")

    if app_config.run == 'train':
        run_training(app_config)
    elif app_config.run == 'inference':
        run_inference(app_config)
