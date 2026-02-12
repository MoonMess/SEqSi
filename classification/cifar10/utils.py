import torch
import torch.nn as nn
import os
from model import ResNet
import importlib.util
import random
import numpy as np

def load_model(args, device, is_training=False):
    """
    Instantiates a model based on arguments and loads a checkpoint if specified.
    
    Args:
        args (argparse.Namespace): A namespace containing the model configuration.
        device (str): The PyTorch device ('cuda' or 'cpu') to load the model onto.
        is_training (bool): If True, the model is loaded for training (e.g., with dropout).

    Returns:
        A tuple containing:
        - model (nn.Module): The instantiated model.
        - checkpoint (dict or None): The loaded checkpoint dictionary, or None if no
          checkpoint was loaded.
    """

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"==> Loading model from checkpoint '{args.checkpoint}'...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        state_dict = checkpoint['model']
        # Create a new state_dict without the 'module.' prefix
        unwrapped_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load config from checkpoint, falling back to args
        #model_type = checkpoint.get('model_type', args.model)
        model_type = checkpoint['model_type']
        num_classes = checkpoint['num_classes']
        norm_type = checkpoint.get('norm_type', None) # Always get from checkpoint, default to None if missing
        dataset = checkpoint.get('dataset', 'cifar10') # Infer dataset from checkpoint, default to cifar10
        print(f"==> Building model '{model_type}' for dataset '{dataset}' with normalization layer '{norm_type}'..")
        dropout_rate = getattr(args, 'dropout_rate', 0.0) if is_training else 0.0
        model = ResNet(
            num_classes=num_classes,
            mode=model_type,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            dataset=dataset
        ).to(device)
        model.load_state_dict(unwrapped_state_dict)
        
        return model, checkpoint
    elif args.checkpoint:
        # A path was provided, but it's not a file.
        raise FileNotFoundError(f"==> No checkpoint found at '{args.checkpoint}'. Please provide a valid path.")
    else:
        print("==> No checkpoint provided. Using a newly initialized (untrained) model.")
        print(f"==> Building model '{args.model}' with normalization layer '{args.norm_type}'..")
        model = ResNet(
            num_classes=args.num_classes,
            mode=args.model,
            norm_type=args.norm_type,
            dropout_rate=args.dropout_rate,
            dataset=args.dataset
        ).to(device)
        
        return model, None

def generate_checkpoint_file(args):
    """
    Generates a checkpoint filename based on the run configuration.
    """
    file_name = f'{args.seed}/best_model_{args.model}_{args.dataset}'
    if args.norm_type:
        file_name += f'_{args.norm_type}'
    if args.dropout_rate > 0:
        file_name += f'_dp{args.dropout_rate}'
    if args.affine_aug:
        file_name += '_aff_aug'
    if args.nonaffine_aug:
        file_name += '_nonaff_aug'
    if args.random_aug:
        if "seqsi" in args.test_config:
            file_name += '_limited_rand_aug'
        else:
            file_name += '_rand_aug'
    if args.comment:
        file_name += f'_{args.comment}'
    file_name += '.pth'
    return os.path.join('./checkpoint', file_name)

def load_config_from_py(config_path):
    """
    Dynamically loads a Python configuration file.
    The file must contain a dictionary named `CONFIG`.
    """
    try:
        # Create a module specification from the file path
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        if spec is None:
            raise ImportError(f"Could not find the config module at: {config_path}")
        
        # Create a new module based on the spec
        config_module = importlib.util.module_from_spec(spec)
        
        # Execute the module to load its content
        spec.loader.exec_module(config_module)
        
        # Return the CONFIG dictionary from the module
        return config_module.CONFIG
    except (FileNotFoundError, AttributeError, ImportError) as e:
        print(f"Error loading Python config file '{config_path}': {e}")
        return None
    
def set_seed_and_determinism(seed):
    """
    Sets the seed for all relevant random number generators and configures
    PyTorch for deterministic behavior.
    """
    print(f"==> Setting global seed to {seed} for reproducibility")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)