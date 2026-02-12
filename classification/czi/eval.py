import argparse
from pathlib import Path
import torch
import pandas as pd
import pytorch_lightning as pl

# Local project imports
from train import ClassificationSystem, CryoETClassificationDataModule
import config.config_classification as config

def evaluate_model():
    """
    Evaluates a trained classification model on multiple test datasets,
    one for each tomogram type (e.g., denoised, wbp).
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the test sets of different tomogram types.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to the model checkpoint (.ckpt).")
    parser.add_argument("--data_root", "-d", type=str, required=True, help="Root directory containing the tomogram type folders (e.g., '.../classification/').")
    parser.add_argument("--output_dir", "-o", type=str, default="./output/evaluation", help="Directory to save the evaluation report.")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE * 2, help="Batch size for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # --- 1. Preparation ---
    pl.seed_everything(args.seed, workers=True)

    base_data_root = Path(args.data_root)
    # The output directory now includes the seed to organize results
    output_dir = Path(args.output_dir) / str(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Automatically detect available tomogram types
    try:
        tomo_types = sorted([d.name for d in base_data_root.iterdir() if d.is_dir() and (d / 'test').exists()])
    except FileNotFoundError:
        print(f"ERROR: Data directory '{base_data_root}' was not found.")
        return
        
    if not tomo_types:
        print(f"ERROR: No valid datasets (containing a 'test' subfolder) were found in {base_data_root}")
        return

    print(f"Tomogram types found for evaluation: {tomo_types}")

    # --- 2. Load Model ---
    print(f"\nLoading model from: {args.checkpoint}")
    try:
        # Load the model. map_location is handled by pl.Trainer
        model = ClassificationSystem.load_from_checkpoint(args.checkpoint)
        model.eval()
    except Exception as e:
        print(f"ERROR: Could not load model from checkpoint. Error: {e}")
        return
    
    print(f"Model '{model.hparams.model_mode}' loaded successfully.")

    all_results = []

    # --- 3. Evaluation Loop ---
    for tomo_type in tomo_types:
        print(f"\n--- Evaluating on tomogram type: {tomo_type} ---")
        
        current_data_root = base_data_root / tomo_type
        
        # Skip if the test folder is empty or does not exist
        test_metadata = current_data_root / 'test' / 'metadata.csv'
        if not test_metadata.exists() or pd.read_csv(test_metadata).empty:
            print(f"  -> Test set for '{tomo_type}' is empty or not found. Skipping.")
            continue

        # Configure the DataModule for the current tomogram type
        datamodule = CryoETClassificationDataModule(
            data_root=current_data_root,
            target_patch_size=config.TARGET_PATCH_SIZE,
            batch_size=args.batch_size,
            num_workers=config.NUM_WORKERS,
            use_augmentation=False
        )
        
        # Configure the Trainer for evaluation
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            logger=False,  # No W&B logging for this evaluation
            callbacks=[],
            precision='bf16-mixed' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else '32-true',
        )

        # Run evaluation
        test_output = trainer.test(model, datamodule=datamodule, verbose=False)
        
        if test_output:
            metrics = test_output[0]
            result_summary = {
                'tomo_type': tomo_type,
                'test_accuracy': metrics.get('test/acc', float('nan')),
                'test_f1_macro': metrics.get('test/f1_macro', float('nan')),
            }
            all_results.append(result_summary)
            
            print(f"  -> Accuracy: {result_summary['test_accuracy']:.4f}")
            print(f"  -> F1-Score (Macro): {result_summary['test_f1_macro']:.4f}")

    # --- 4. Aggregate and Save Results ---
    if not all_results:
        print("\nEvaluation produced no results for any tomogram type.")
        return

    results_df = pd.DataFrame(all_results)
    print("\n--- Summary of Evaluation Results ---")
    print(results_df.to_string(index=False))
    
    # Use the parent folder name of the checkpoint to name the results file
    checkpoint_name = Path(args.checkpoint).parent.name
    output_csv = output_dir / f"evaluation_summary_{checkpoint_name}.csv"
    results_df.to_csv(output_csv, index=False, float_format='%.4f')
    print(f"\nComplete results have been saved to: {output_csv}")

if __name__ == "__main__":
    evaluate_model()
