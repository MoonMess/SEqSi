import argparse
import torch
import torch.nn as nn
import os
import time
import numpy as np
import pandas as pd

# Import project-specific modules
from utils import load_model, set_seed_and_determinism
from dataloader import get_data_loaders

def test_model_on_dataset(dataset_name, model, testloader, criterion, device):
    """
    Evaluates the model on a given dataset and returns accuracy and loss.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Measure inference time
    start_time = time.perf_counter() 

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() 
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Synchronization for accurate GPU time measurement
    end_time = time.perf_counter()

    total_time = end_time - start_time
    images_per_sec = total / total_time if total_time > 0 else 0

    acc = 100. * correct / total
    avg_loss = test_loss / len(testloader)

    print(f'==> Test Results on {dataset_name}: Loss: {avg_loss:.3f} | Acc: {acc:.3f}% ({correct}/{total})')
    return acc, avg_loss, total_time, images_per_sec


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Model Evaluation')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Path to the model checkpoint file (.pth).')
    parser.add_argument('--dataset', default=None, type=str, help='Dataset to use (cifar10). Inferred from checkpoint if not set.')
    parser.add_argument('--data_path', default='./data', type=str, help='Path to the dataset.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--output_dir', default='./output/test_results', help='Directory to save evaluation results.')
    parser.add_argument('--config', type=str, default='transformation/config.py', help='Path to the test transformation configuration file.')
    args = parser.parse_args()

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, running on CPU. This will be slow.")

    # Load the model
    model, checkpoint = load_model(args, device)
    model.eval()

    # Determine the dataset to use
    dataset_to_use = args.dataset
    if dataset_to_use is None:
        dataset_to_use = checkpoint.get('dataset', 'cifar10')
    print(f"\n==> Evaluating on dataset: {dataset_to_use}")

    # Load test datasets
    _, _, test_loaders = get_data_loaders(
        dataset_name=dataset_to_use,
        batch_size=args.batch_size,
        data_path=args.data_path,
        test_config_path=args.config,
    )    
    if not test_loaders:
        print("No test datasets could be loaded. Exiting script.")
        return

    # Loss function definition (CrossEntropyLoss is standard for classification)
    criterion = nn.CrossEntropyLoss()

    # Create an output directory name based on the checkpoint file name
    checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')

    output_path = os.path.join(args.output_dir,checkpoint_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"==> Starting evaluation from checkpoint: {args.checkpoint}")
    print("-----------------------------------")

    # Set the seed for reproducibility for this single run
    print(f"\n===== RUNNING EVALUATION WITH SEED: {args.seed} =====")
    set_seed_and_determinism(args.seed)


    results = []
    # Iterate over test loaders to evaluate model on different test sets
    for dataset_name, testloader in test_loaders.items():
        print(f"\nEvaluating on dataset: {dataset_name}")

        # Evaluate the model and get the results
        acc, avg_loss, total_time, images_per_sec = test_model_on_dataset(dataset_name, model, testloader, criterion, device)
        results.append({
            'Dataset Name': dataset_name,
            'Accuracy (%)': acc,
            'Average Loss': avg_loss,
            'Total Inference Time (s)': total_time,
            'Images/sec': images_per_sec
        })
        
    # Save the results to a single CSV file
    results_df = pd.DataFrame(results)

    csv_path = os.path.join(output_path, 'evaluation_results.csv')
    results_df.to_csv(csv_path, index=False, float_format='%.3f')

    print("\n-----------------------------------")
    print(f"Evaluation completed. Results saved to {csv_path}")

if __name__ == '__main__':
    main()