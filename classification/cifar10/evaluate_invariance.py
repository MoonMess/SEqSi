import argparse
import torch
import torch.nn.functional as F
import os
import pandas as pd

# Utility imports
from utils import load_model, set_seed_and_determinism
import torchvision
from torchvision import transforms
from tqdm import tqdm

def compute_error(logits, logits_alter):
    """
    Computes the equivariance error between two sets of logits for a batch.
    Returns a dictionary containing error metrics.
    """
    prediction_alter = torch.argmax(F.softmax(logits_alter, dim=1), dim=1)
    prediction = torch.argmax(F.softmax(logits, dim=1), dim=1)

    # Mean and max absolute error between logits
    max_error = torch.abs(logits - logits_alter).max().item()
    mae = torch.abs(logits - logits_alter).mean().item()

    # Count mismatched predictions
    mismatches = (prediction != prediction_alter)
    mismatch_count = torch.sum(mismatches).item()
    mismatch_indices = mismatches.nonzero(as_tuple=True)[0]

    error = {
        'mismatch_count': mismatch_count,
        'mxe': max_error,
        'mae': mae,
        'mismatch_indices': mismatch_indices,
        'original_preds': prediction,
        'altered_preds': prediction_alter,
    }
    return error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Photometric Equivariance Evaluation on CIFAR-10')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Path to the model checkpoint file (.pth).')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for the test.')
    parser.add_argument('--data_path', default='./data', type=str, help='Path to CIFAR-10 data.')
    parser.add_argument('--output_dir', default='./output/equiv_eval', help='Directory to save evaluation results.')
    parser.add_argument('--float64', action='store_true', help='Use float64 precision for computations.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    # --- Initial Setup ---
    all_results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, running on CPU. This will be slow.")
    
    # Use float32 by default for better performance, which is standard in deep learning.
    if args.float64:
        print("Using float64 precision.")
        torch.set_default_dtype(torch.float64)
    else:
        print("Using float32 precision.")
        torch.set_default_dtype(torch.float32)
    
    # The seed has no impact here as data and transforms are deterministic.
    # It is set once for consistency in case stochastic operations are added later.
    set_seed_and_determinism(args.seed)
    
    # Load model
    model, checkpoint = load_model(args, device)
    model.eval()
    model_name = checkpoint['model_type']

    # --- Load CIFAR-10 test data ---
    print("==> Loading CIFAR-10 test data...")
    transform_to_tensor = transforms.ToTensor()
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform_to_tensor
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    print(f"==> Data loaded successfully. Total test images: {len(test_dataset)}.")

    print(f"==> Evaluating on the full CIFAR-10 test set ({len(test_dataset)} images).")
    
    # Prepare output directory for mismatch images if requested
    checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')
    # Define a systematic set of transformations to test
    test_cases = []
    
    # Group 1: Shift Invariance (Scale is 1.0)
    shifts_to_test = [-2, 0.5, 2, 10]
    for shift in shifts_to_test:
        test_cases.append({'shift': shift, 'scale': 1.0})
        
    # Group 2: Scale Invariance (Shift is 0.0)
    scales_to_test = [0.5, 3.0, 255]
    for scale in scales_to_test:
        test_cases.append({'shift': 0.0, 'scale': scale})
        
    # Group 3: Affine Equivariance (Both shift and scale are non-trivial)
    affine_to_test = [
        (5, 0.1),
        (5, 3.0),
        (-2, 10.0),
    ]
    for shift, scale in affine_to_test:
        test_cases.append({'shift': shift, 'scale': scale})

    # Run all test cases on the entire dataset
    for case in tqdm(test_cases, desc="Testing cases"):
        intensity_shift = case['shift']
        intensity_scale = case['scale']

        # Variables to accumulate results over all batches
        total_mismatches = 0
        total_mae = 0.0
        max_mxe_in_case = 0.0
        total_samples = 0

        # Loop over all batches of the test set
        for batch_idx, (data_batch, _) in enumerate(test_loader):
            data_batch = data_batch.to(device)

            # Apply the photometric transformation
            alter_data = data_batch.clone() * intensity_scale + intensity_shift

            with torch.no_grad():
                logits = model(data_batch)
                logits_alter = model(alter_data)
                result = compute_error(logits, logits_alter)

            # Accumulate batch results
            total_mismatches += result['mismatch_count']
            # MAE is a batch average, so we weight it by the batch size
            total_mae += result['mae'] * data_batch.size(0)
            max_mxe_in_case = max(max_mxe_in_case, result['mxe'])
            total_samples += data_batch.size(0)

        # Calculate final metrics for this test case
        avg_mismatch_percent = (total_mismatches / total_samples) * 100 if total_samples > 0 else 0
        avg_mae = total_mae / total_samples if total_samples > 0 else 0

        all_results.append({
            'Shift': f'{intensity_shift}',
            'Scale': f'{intensity_scale}',
            'Prediction Mismatch (%)': avg_mismatch_percent,
            'Mean Absolute Logits Error': avg_mae,
            'Max Logits Error': max_mxe_in_case
        })

    # Aggregate results with pandas
    final_results = pd.DataFrame(all_results)

    # Prepare the output file
    if args.float64:
        output_path = os.path.join(args.output_dir, "float64")
    else:
        output_path = os.path.join(args.output_dir, "float32")
    os.makedirs(output_path, exist_ok=True)        
    csv_path = os.path.join(output_path, f"{checkpoint_name}_equinv_results.csv")
    final_results.to_csv(csv_path, index=False, float_format='%.7f')

    print("\n===== RESULTS =====")
    print(final_results.to_string())
    print(f"\nEvaluation finished. Results saved in {csv_path}")
