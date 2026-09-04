# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky
import torch

from models.model import Transformer

def get_model(config, device):
    """
    Dynamically selects and instantiates the model based on config.
    """
    model_type = config.model.architecture

    model_classes = {
        "Transformer": Transformer,
        # Add other model names here
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type '{model_type}' specified in config. "
                         f"Available models are: {list(model_classes.keys())}")

    ModelClass = model_classes[model_type]

    # Instantiating model with appropriate parameters based on its type
    if model_type == "Transformer":
        model_params = config['model']['transformer_params']
        model = ModelClass(
            k=model_params['k'],
            heads=model_params['heads'],
            depth=model_params['depth'],
            seq_length=config['data']['frame_window_size'],
            num_tokens=model_params['num_tokens'],
            num_classes=model_params['num_classes']
        )
    else:
        raise NotImplementedError(f"Model instantiation for '{model_type}' is not yet implemented in model_factory.py.")

    model.to(device)
    print(f"Model '{model_type}' instantiated and moved to {device}.")
    return model

def load_model(model, config, device):

    checkpoint = torch.load(config.paths.model_final_dir  + config.data.behavior +
                            str(config.data.frame_window_size) + '_finalstate.pth', map_location=device, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model