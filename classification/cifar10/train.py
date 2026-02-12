import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import wandb
import time
import random
import numpy as np
import pandas as pd

# Project-specific imports
from model import ResNet
from dataloader import get_data_loaders
from utils import generate_checkpoint_file, load_model, set_seed_and_determinism, count_parameters
print(f"PyTorch Version: {torch.__version__}")


# --- Training and Testing Loop ---

def train(epoch, model, trainloader, optimizer, criterion, device, clip_grad_norm=None):
    print(f'\n--- Epoch: {epoch} ---')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # Performance metrics
    start_time = time.perf_counter()
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')

    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    epoch_duration = end_time - start_time
    images_per_sec = total / epoch_duration if epoch_duration > 0 else 0
    
    print(f'==> Epoch {epoch} Training Summary: Duration: {epoch_duration:.2f}s | Throughput: {images_per_sec:.2f} img/s')
    if device == 'cuda':
        max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f'    Peak GPU Memory: {max_mem_gb:.3f} GB')
    else:
        max_mem_gb = 0
    avg_loss = train_loss / len(trainloader)
    avg_acc = 100. * correct / total

    return epoch_duration, images_per_sec, max_mem_gb, avg_loss, avg_acc

def test(dataset_name, model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    start_time = time.perf_counter()
    if device == 'cuda':
        torch.cuda.synchronize()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_duration = end_time - start_time
    images_per_sec = total / inference_duration if inference_duration > 0 else 0

    accuracy = 100. * correct / total
    avg_loss = test_loss / len(testloader)
    print(f'==> Test Results on {dataset_name}: Loss: {avg_loss:.3f} | Acc: {accuracy:.3f}% ({correct}/{total}) | Inference Time: {inference_duration:.2f}s ({images_per_sec:.2f} img/s)')

    return accuracy, avg_loss, inference_duration, images_per_sec


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--model', default='standard', type=str, help="Model architecture type.", choices=['standard', 'seqsi_avg', 'seqsi_telescopic', 'affeq_telescopic','affeq_avg', 'se'])
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use', choices=['cifar10'])
    parser.add_argument('--num_classes', default=None, type=int, help='Number of classes in the dataset. Inferred from dataset if not set.')
    parser.add_argument('--checkpoint', '-c', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--data_path', default='./data', type=str, help='Path to dataset.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=500, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--norm_type', default=None, type=str, help='Normalization layer type (e.g., "batch", "instance", "layer").')
    parser.add_argument('--dropout_rate', default=0.0, type=float, help='Dropout rate for the model')
    parser.add_argument('--test_config', default='transformation/config.py', type=str, help='Path to test transformations config file')
    parser.add_argument('--affine_aug', action='store_true', help='Use affine data augmentation during training.')
    parser.add_argument('--nonaffine_aug', action='store_true', help='Use non-affine data augmentation during training.')
    parser.add_argument('--random_aug', action='store_true', help='Use a randomly chosen augmentation from the config file for each image during training.')
    parser.add_argument('--comment', type=str, default='', help='A comment for the run, used in file names and WandB.')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Max norm for gradient clipping.')
    
    # WandB arguments
    parser.add_argument('--wandb', action='store_true', help='Disable logging with Weights & Biases.')
    parser.add_argument('--wandb_project', type=str, default='NE_Classif', help='Weights & Biases project name.')
    args = parser.parse_args()
    
    # --- Setup ---
    
    # Centralize seed setting for full reproducibility
    set_seed_and_determinism(args.seed)

    # Infer num_classes from dataset if not specified
    if args.num_classes is None:
        if args.dataset == 'cifar10':
            args.num_classes = 10
        else:
            raise ValueError(f"Unknown dataset '{args.dataset}'. Cannot infer num_classes.")

    # Configure device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    trainloader, valloader, test_loaders = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        data_path=args.data_path,
        val_split=0.1,
        test_config_path=args.test_config,
        affine_aug=args.affine_aug,
        nonaffine_aug=args.nonaffine_aug,
        random_aug=args.random_aug
    )
    
    # --- Model, Optimizer, and Scheduler Initialization ---
    print(f"==> Building model '{args.model}'..")
    model, checkpoint = load_model(args, device, is_training=True)
    
    # Display number of parameters
    num_params = count_parameters(model)
    print(f"==> Total trainable parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0  # To track the best validation accuracy
    start_epoch = 0

    # --- Resume from Checkpoint ---
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['acc']
        print(f"Resumed from epoch {checkpoint['epoch']}, best accuracy was {best_acc:.3f}%")

    # Model saving path
    checkpoint_path = generate_checkpoint_file(args)

    # --- Weights & Biases Initialization ---
    if args.wandb:
        run_name = os.path.basename(checkpoint_path).replace('.pth', '')
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=args
        )
        wandb.watch(model, criterion, log="all", log_freq=100)

    # --- Performance Tracking ---
    warmup_epochs = 5
    train_durations = []
    train_throughputs = []
    train_mems = []
    val_durations = []
    val_throughputs = []

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        train_duration, train_throughput, train_mem, train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, device, args.clip_grad)
        val_acc, val_loss, val_duration, val_throughput = test('Validation', model, valloader, criterion, device)

        # Collect performance metrics after the warmup period
        if epoch >= warmup_epochs:
            train_durations.append(train_duration)
            train_throughputs.append(train_throughput)
            if device == 'cuda':
                train_mems.append(train_mem)
            val_durations.append(val_duration)
            val_throughputs.append(val_throughput)

        # Log metrics to WandB
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'perf/train_duration_s': train_duration,
                'perf/train_throughput_img_s': train_throughput,
                'perf/peak_gpu_mem_gb': train_mem,
                'perf/val_duration_s': val_duration,
                'perf/val_throughput_img_s': val_throughput,
                'lr': scheduler.get_last_lr()[0]
            }, step=epoch)

        # Save checkpoint if it's the best model on the validation set
        if val_acc > best_acc:
            print(f'\nNew best model found! Validation Accuracy: {val_acc:.3f}% (old: {best_acc:.3f}%)')
            print('==> Saving model..')
            state = {
                'model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'model_type': args.model,
                'norm_type': args.norm_type,
                'num_classes': args.num_classes,
                'dataset': args.dataset,
                'checkpoint_path': os.path.dirname(checkpoint_path)
            }
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(state, checkpoint_path)
            best_acc = val_acc

        scheduler.step()

    print("Finished Training")
    print(f"Best validation accuracy achieved: {best_acc:.3f}%")
    
    # --- Display average performance metrics ---
    if len(train_durations) > 0:
        avg_train_duration = sum(train_durations) / len(train_durations)
        avg_train_throughput = sum(train_throughputs) / len(train_throughputs)
        avg_val_duration = sum(val_durations) / len(val_durations)
        avg_val_throughput = sum(val_throughputs) / len(val_throughputs)

        summary_perf = {
            "avg_train_duration_s": avg_train_duration,
            "avg_train_throughput_img_s": avg_train_throughput,
            "avg_val_duration_s": avg_val_duration,
            "avg_val_throughput_img_s": avg_val_throughput,
        }

        print("\n--- Average Performance (post-warmup) ---")
        print(f"  Avg. Training Epoch Duration: {summary_perf['avg_train_duration_s']:.2f}s")
        print(f"  Avg. Training Throughput: {summary_perf['avg_train_throughput_img_s']:.2f} img/s")
        if device == 'cuda' and train_mems:
            valid_mems = [m for m in train_mems if m > 0] # Filter out potential zero values
            if valid_mems:
                summary_perf["avg_peak_gpu_mem_gb"] = sum(valid_mems) / len(valid_mems)
                print(f"  Avg. Peak Training Memory: {summary_perf['avg_peak_gpu_mem_gb']:.3f} GB")
        print(f"  Avg. Validation Inference Time: {summary_perf['avg_val_duration_s']:.2f}s")
        print(f"  Avg. Validation Throughput: {summary_perf['avg_val_throughput_img_s']:.2f} images/sec")
        print("-----------------------------------------")

        if args.wandb:
            wandb.summary.update(summary_perf)

        # Save performance summary to a JSON file
        perf_summary_path = checkpoint_path.replace('.pth', '_perf_summary.json')
        import json
        with open(perf_summary_path, 'w') as f:
            json.dump(summary_perf, f, indent=4)
        print(f"Performance summary saved to {perf_summary_path}")

    # --- Final Evaluation ---
    print(f"\n==> Starting final evaluation using the best model from {checkpoint_path}...")

    # To ensure a clean evaluation, we re-load the test dataloaders just before
    # evaluation. This avoids potential stale state issues with DataLoader workers
    # that have been idle during the entire training process.
    print("==> Re-loading test datasets for final evaluation...")
    _, _, final_test_loaders = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        data_path=args.data_path,
        val_split=0,  # No validation split needed for final testing
        test_config_path=args.test_config,
    )

    # Load the state from the best checkpoint into the model
    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint['model'])

    final_results = []
    for name, testloader in final_test_loaders.items():
        accuracy, loss, duration, throughput = test(name, model, testloader, criterion, device)
        final_results.append({
            'Dataset Name': name,
            'Accuracy (%)': accuracy,
            'Average Loss': loss,
            'Total Inference Time (s)': duration,
            'Images/sec': throughput
        })
        if args.wandb:
            wandb.summary[f'test_acc_{name}'] = accuracy
            wandb.summary[f'test_loss_{name}'] = loss
    
    # Save final results to a CSV file
    if final_results:
        results_df = pd.DataFrame(final_results)
        csv_path = checkpoint_path.replace('.pth', '_final_eval.csv')
        results_df.to_csv(csv_path, index=False, float_format='%.3f')
        print(f"\nFinal evaluation results saved to {csv_path}")

    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()