import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os

from transformation.photometric_extensions import TRANSFORM_REGISTRY
from utils import load_config_from_py

def get_cifar10_data_loaders(batch_size, data_path, val_split, test_config_path, affine_aug=False, non_affine_aug=False, random_aug=False):
    """
    Creates DataLoaders for the CIFAR-10 dataset.
    """
    print("==> Preparing CIFAR-10 data..")
        
    # 1. Base transformations for training (with augmentation) and for validation/testing (without).
    train_transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    config = load_config_from_py(test_config_path)
    if not config or 'transformations' not in config:
        raise ValueError(f"Could not load transformations from config file: {test_config_path}")
    trans_config = config['transformations']

    # Add optional photometric augmentations (applied on Tensors).
    if random_aug:    
        print("==> Applying all augmentation to training data...")

        config_transforms = []
        # Instantiate all transformations from the config file.
        for group_name, group_items in trans_config.items():
            for name, params in group_items.items():
                try:
                    transform_class_name = params['type']
                    transform_params = params.get('params', {})
                    is_per_image = params.get("per_image_randomness", False)
                    class_dict = TRANSFORM_REGISTRY['per_image'] if is_per_image else TRANSFORM_REGISTRY['standard']
                    transform_class = class_dict[transform_class_name]
                    config_transforms.append(transform_class(**transform_params))
                except Exception as e:
                    print(f"Warning: Could not create transform '{name}'. Skipping. Error: {e}")
        
        # Add an identity transform to have a chance of not applying any extra augmentation.
        config_transforms.append(transforms.Lambda(lambda x: x))
        print(f"==> Created {len(config_transforms)} choices for random augmentation (including identity).")
        train_transforms_list.append(transforms.RandomChoice(config_transforms))

    elif affine_aug:
        print("==> Applying affine augmentation to training data...")
        # Instantiate only affine transformations from the config file.
        config_transforms = []
        for group_name, group_items in trans_config.items():
            if group_name == "affine":
                for name, params in group_items.items():
                    try:
                        transform_class_name = params['type']
                        transform_params = params.get('params', {})
                        is_per_image = params.get("per_image_randomness", False)
                        class_dict = TRANSFORM_REGISTRY['per_image'] if is_per_image else TRANSFORM_REGISTRY['standard']
                        transform_class = class_dict[transform_class_name]
                        config_transforms.append(transform_class(**transform_params))
                    except Exception as e:
                        print(f"Warning: Could not create transform '{name}'. Skipping. Error: {e}")
        # Add an identity transform.
        config_transforms.append(transforms.Lambda(lambda x: x))
        print(f"==> Created {len(config_transforms)} choices for affine augmentation (including identity).")
        train_transforms_list.append(transforms.RandomChoice(config_transforms))

    elif non_affine_aug:
        print("==> Applying non-affine augmentation to training data...")
        # Instantiate only non-affine transformations from the config file.
        config_transforms = []
        for group_name, group_items in trans_config.items():
            if group_name == "non_affine":
                for name, params in group_items.items():
                    try:
                        transform_class_name = params['type']
                        transform_params = params.get('params', {})
                        is_per_image = params.get("per_image_randomness", False)
                        class_dict = TRANSFORM_REGISTRY['per_image'] if is_per_image else TRANSFORM_REGISTRY['standard']
                        transform_class = class_dict[transform_class_name]
                        config_transforms.append(transform_class(**transform_params))
                    except Exception as e:
                        print(f"Warning: Could not create transform '{name}'. Skipping. Error: {e}")
        # Add an identity transform.
        config_transforms.append(transforms.Lambda(lambda x: x))
        print(f"==> Created {len(config_transforms)} choices for non affine augmentation (including identity).")
        train_transforms_list.append(transforms.RandomChoice(config_transforms))

    transform_train = transforms.Compose(train_transforms_list)

    # Create two instances of the training dataset with different transforms
    # for the train/validation split.
    train_set_with_aug = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train)
    
    val_set_no_aug = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_val_test)

    # Split into training and validation sets
    num_train = len(train_set_with_aug)
    indices = list(range(num_train))
    split = int(val_split * num_train)
    
    # Shuffle indices for a random and reproducible split.
    # The seed is now set globally at the beginning of train.py.
    indices = torch.randperm(num_train).tolist()
    train_idx, val_idx = indices[split:], indices[:split]

    train_subset = Subset(train_set_with_aug, train_idx)
    val_subset = Subset(val_set_no_aug, val_idx)

    print(f"Data split: {len(train_subset)} training samples, {len(val_subset)} validation samples.")

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create test DataLoaders from the configuration file
    test_loaders = get_config_based_test_loaders(test_config_path, batch_size=batch_size, data_path=data_path)

    return trainloader, valloader, test_loaders

def get_data_loaders(dataset_name, batch_size, data_path='./data', **kwargs):
    """
    Dispatcher function that selects the correct data loader based on the dataset name.
    """
    if dataset_name == 'cifar10':
        return get_cifar10_data_loaders(
            batch_size, 
            data_path, 
            kwargs.get('val_split', 0.10), 
            kwargs.get('test_config_path', 'transformation/config.py'),
            affine_aug=kwargs.get('affine_aug', False),
            non_affine_aug=kwargs.get('nonaffine_aug', False),
            random_aug=kwargs.get('random_aug', False),
        )
    
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

def get_config_based_test_loaders(config_path: str, batch_size: int, data_path: str = './data'):
    """
    Creates a dictionary of test DataLoaders based on a Python configuration file.
    """
    print(f"==> Preparing test data from config: {config_path}")

    # Load configuration
    config = load_config_from_py(config_path)
    if not config or 'transformations' not in config:
        return {}
    trans_config = config['transformations']
        
    common_test_transforms = transforms.Compose([transforms.ToTensor()])
    
    test_loaders = {}

    # Add the original, unperturbed test set
    testset_original = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, 
        transform=transforms.Compose([common_test_transforms])
    )
    test_loaders["Original"] = DataLoader(testset_original, batch_size=batch_size, shuffle=False, num_workers=2)

    # Iterate over transformations from the config
    for group_name, group_items in trans_config.items():
        for name, params in group_items.items():
            try:
                transform_class_name = params['type']
                transform_params = params.get('params', {})
                is_per_image = params.get("per_image_randomness", False)
                
                # Select the appropriate class dictionary
                class_dict = TRANSFORM_REGISTRY['per_image'] if is_per_image else TRANSFORM_REGISTRY['standard']
                
                if transform_class_name in class_dict:
                    transform_class = class_dict[transform_class_name]
                else:
                    raise KeyError(f"Transform class '{transform_class_name}' is not defined for the selected mode.")
                
                photometric_transform = transform_class(**transform_params)
                
                full_transform = transforms.Compose([
                    common_test_transforms, photometric_transform,
                ])
                
                testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=full_transform)
                test_loaders[name] = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
                print(f"  - Loader created for transform: '{name}'")

            except KeyError:
                print(f"Warning: Transform class '{params.get('type')}' for '{name}' is not defined. Skipped.")
            except Exception as e:
                print(f"Error creating transform '{name}': {e}")

    return test_loaders