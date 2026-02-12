import argparse
import os
import json
import numpy as np
from pathlib import Path
import importlib.util
from tqdm import tqdm
import pandas as pd

# Optional plotting libraries for visualization
try:
    import matplotlib.pyplot as plt
    PLOTTING_LIBS_AVAILABLE = True
except ImportError:
    PLOTTING_LIBS_AVAILABLE = False

# Zarr library is required for reading tomogram data
try:
    import zarr
except ImportError:
    print("Error: The 'zarr' library is not installed. Please install it with 'pip install zarr'.")
    exit(1)


def load_config_from_file(config_path):
    """Loads the class dictionary from a Python configuration file."""
    spec = importlib.util.spec_from_file_location("particle_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the configuration module from {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    classes = getattr(config_module, 'PARTICLE_CLASSES', None)
    
    if classes is None:
        raise AttributeError("The configuration file must contain a 'PARTICLE_CLASSES' dictionary.")
        
    return classes

def _parse_json_picks(file_path: str) -> list[tuple[float, float, float]]:
    """Reads a JSON picks file and returns a list of coordinates (X, Y, Z) in Angstroms."""
    coordinates = []
    if not os.path.exists(file_path):
        return coordinates
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for point in data.get('points', []):
                location = point.get('location')
                if location and 'x' in location and 'y' in location and 'z' in location:
                    coordinates.append((location['x'], location['y'], location['z']))
    except Exception as e:
        print(f"Warning: Could not process picks file {file_path}: {e}")
    return coordinates

def _create_patch_visualization(
    patch_data: np.ndarray,
    particle_name: str,
    output_path: Path
):
    """
    Creates a 3-view visualization (XY, XZ, YZ) of an extracted patch.
    """
    if not PLOTTING_LIBS_AVAILABLE:
        return

    patch_d, patch_h, patch_w = patch_data.shape
    slice_d, slice_h, slice_w = patch_d // 2, patch_h // 2, patch_w // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Patch visualization for '{particle_name}'\nSize: {patch_w}x{patch_h}x{patch_d}", fontsize=16)

    vmin, vmax = np.percentile(patch_data, [1, 99])

    axes[0].imshow(patch_data[slice_d, :, :], cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"XY View (Z-slice={slice_d})")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")

    axes[1].imshow(patch_data[:, slice_h, :], cmap='gray', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f"XZ View (Y-slice={slice_h})")
    axes[1].set_xlabel("X-axis")
    axes[1].set_ylabel("Z-axis")

    axes[2].imshow(patch_data[:, :, slice_w], cmap='gray', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
    axes[2].set_title(f"YZ View (X-slice={slice_w})")
    axes[2].set_xlabel("Y-axis")
    axes[2].set_ylabel("Z-axis")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path)
    plt.close(fig)
    tqdm.write(f"  Patch visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and split particle patches (train/val/test) for a classification task from the CZI Challenge dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the CZI dataset (containing 'train', 'test', etc.).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the data subfolders (train, val, test).")
    parser.add_argument("--config_file", type=str, default="config/particle_config.py", help="Path to the Python configuration file (e.g., particle_config.py).")
    parser.add_argument("--tomo_type", type=str, nargs='+', default=["denoised", "ctfdeconvolved", "isonetcorrected", "wbp"], help="One or more tomogram types to process (e.g., 'denoised', 'wbp').")
    parser.add_argument("--voxel_spacing", type=float, default=10.012, help="Voxel size in Angstroms for coordinate conversion.")
    parser.add_argument("--patch_size", type=int, nargs=3, required=True, help="Size of the patches to extract in pixels [D, H, W].")
    parser.add_argument("--visualize", action='store_true', help="Generate a 3-view visualization for one example of each particle type.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting.")
    
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    print(f"Loading configuration from: {args.config_file}")
    try:
        particle_classes = load_config_from_file(args.config_file)
    except (ImportError, AttributeError) as e:
        print(f"ERROR: {e}")
        return

    # --- 2. Collect All Annotations ---
    data_root_path = Path(args.data_root)
    tomo_runs_dir = data_root_path / "train/static/ExperimentRuns"
    picks_base_dir = data_root_path / "train/overlay/ExperimentRuns"

    run_ids = sorted([d.name for d in tomo_runs_dir.iterdir() if d.is_dir() and d.name.startswith("TS_")])
    if not run_ids:
        print(f"ERROR: No run folders ('TS_*') found in {tomo_runs_dir}")
        return
    
    print("Step 1/3: Collecting all annotations from picks files...")
    all_annotations = []
    for run_id in tqdm(run_ids, desc="Reading picks"):
        for class_name, class_id in particle_classes.items():
            json_path = picks_base_dir / run_id / "Picks" / f"{class_name}.json"
            coords_angstrom = _parse_json_picks(str(json_path))
            for x_a, y_a, z_a in coords_angstrom:
                all_annotations.append({
                    'run_id': run_id,
                    'class_name': class_name,
                    'class_id': class_id,
                    'x_a': x_a,
                    'y_a': y_a,
                    'z_a': z_a,
                })
    
    if not all_annotations:
        print("ERROR: No annotations found. Aborting.")
        return
    
    print(f"Found a total of {len(all_annotations)} annotations.")

    # --- 3. Split Annotations into Train/Val/Test Sets ---
    print("\nStep 2/3: Splitting annotations into train/val/test sets...")
    # To prevent data leakage, the split is performed at the tomogram (run_id) level
    # rather than at the individual annotation level. This ensures that all patches
    # from a single tomogram belong to the same split.
    unique_run_ids = sorted(list(set(anno['run_id'] for anno in all_annotations)))

    np.random.seed(args.seed)
    np.random.shuffle(unique_run_ids)
    n_runs = len(unique_run_ids)

    val_count = max(1, int(n_runs * 0.05))
    test_count = max(1, int(n_runs * 0.15))
    train_count = n_runs - val_count - test_count

    train_runs = set(unique_run_ids[:train_count])
    val_runs = set(unique_run_ids[train_count : train_count + val_count])
    test_runs = set(unique_run_ids[train_count + val_count :])

    split_definitions = {'train': [], 'val': [], 'test': []}
    for anno in all_annotations:
        run_id = anno['run_id']
        if run_id in train_runs:
            split_definitions['train'].append(anno)
        elif run_id in val_runs:
            split_definitions['val'].append(anno)
        elif run_id in test_runs:
            split_definitions['test'].append(anno)

    print(f"Split into sets (based on {n_runs} tomograms):")
    print(f"  - Train: {len(train_runs)} tomograms ({len(split_definitions['train'])} annotations)")
    print(f"  - Validation: {len(val_runs)} tomograms ({len(split_definitions['val'])} annotations)")
    print(f"  - Test: {len(test_runs)} tomograms ({len(split_definitions['test'])} annotations)")

    print(f"\nTomograms used for VALIDATION ({len(val_runs)}):")
    print(sorted(list(val_runs)))
    print(f"\nTomograms used for TEST ({len(test_runs)}):")
    print(sorted(list(test_runs)))

    # --- 4. Group Annotations for Efficient Processing ---
    # Group annotations by run_id to load each tomogram only once per tomo_type.
    annos_by_run = {run_id: [] for run_id in run_ids}
    for split_name, annos in split_definitions.items():
        for anno in annos:
            anno['split'] = split_name
            annos_by_run[anno['run_id']].append(anno)
    
    base_output_dir = Path(args.output_dir)
    patch_d, patch_h, patch_w = args.patch_size

    # --- 5. Process Each Tomogram Type ---
    print(f"\nStep 3/3: Extracting patches...")
    for current_tomo_type in args.tomo_type:
        print(f"\n--- Processing tomogram type: {current_tomo_type} ---")

        # Prepare output directories for this specific tomogram type
        output_dir_for_type = base_output_dir / current_tomo_type
        split_dirs = {'train': output_dir_for_type / 'train', 'val': output_dir_for_type / 'val', 'test': output_dir_for_type / 'test'}
        for split_dir in split_dirs.values():
            (split_dir / 'patches').mkdir(parents=True, exist_ok=True)

        if args.visualize and PLOTTING_LIBS_AVAILABLE:
            vis_output_dir = base_output_dir / "patch_visualizations" / current_tomo_type
            vis_output_dir.mkdir(parents=True, exist_ok=True)
            visualized_classes = set()
        elif args.visualize and not PLOTTING_LIBS_AVAILABLE:
            print("WARNING: --visualize requires 'matplotlib'. Visualizations will not be generated.")
        
        all_patches_metadata = {'train': [], 'val': [], 'test': []}
        patch_counter = 0

        for run_id in tqdm(annos_by_run.keys(), desc=f"Processing tomograms ({current_tomo_type})"):
            tomo_path = tomo_runs_dir / run_id / "VoxelSpacing10.000" / f"{current_tomo_type}.zarr"

            if not tomo_path.exists():
                tqdm.write(f"  WARNING: Tomogram '{tomo_path}' not found for run '{run_id}'. Skipping.")
                continue

            try:
                tomo_data = zarr.open(tomo_path, mode='r')['0'][:]
                tomo_d_full, tomo_h_full, tomo_w_full = tomo_data.shape
            except Exception as e:
                tqdm.write(f"  WARNING: Could not read Zarr tomogram {tomo_path}. Error: {e}")
                continue

            # Process all annotations for this tomogram
            for anno in annos_by_run[run_id]:
                    current_split = anno['split']
                    current_split_dir = split_dirs[current_split]
                    current_patches_dir = current_split_dir / 'patches'

                    class_name = anno['class_name']
                    class_id = anno['class_id']
                    x_a, y_a, z_a = anno['x_a'], anno['y_a'], anno['z_a']

                    # Convert center coordinates to pixels
                    center_x_px = x_a / args.voxel_spacing
                    center_y_px = y_a / args.voxel_spacing
                    center_z_px = z_a / args.voxel_spacing

                    # Calculate the ideal starting coordinates for the patch
                    z_start = int(round(center_z_px - patch_d / 2))
                    y_start = int(round(center_y_px - patch_h / 2))
                    x_start = int(round(center_x_px - patch_w / 2))

                    # Adjust coordinates to ensure the patch stays within tomogram bounds
                    z_start = max(0, min(z_start, tomo_d_full - patch_d))
                    y_start = max(0, min(y_start, tomo_h_full - patch_h))
                    x_start = max(0, min(x_start, tomo_w_full - patch_w))

                    # Recalculate end coordinates
                    z_end, y_end, x_end = z_start + patch_d, y_start + patch_h, x_start + patch_w

                    # Extract the patch
                    patch = tomo_data[z_start:z_end, y_start:y_end, x_start:x_end]

                    # Verify that the extracted patch has the correct dimensions
                    if patch.shape != (patch_d, patch_h, patch_w):
                        tqdm.write(f"  WARNING: Extracted patch has incorrect shape {patch.shape}. Expected {(patch_d, patch_h, patch_w)}. Skipping.")
                        continue

                    # Save the patch
                    patch_filename = f"{run_id}_{patch_counter:06d}.npy"
                    patch_filepath = current_patches_dir / patch_filename
                    np.save(patch_filepath, patch)
                    
                    # Add metadata
                    all_patches_metadata[anno['split']].append({
                        "patch_path": str(patch_filepath.relative_to(current_split_dir)),
                        "run_id": run_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "center_x_px": center_x_px,
                        "center_y_px": center_y_px,
                        "center_z_px": center_z_px,
                    })
                    patch_counter += 1

                    # Visualization logic
                    if args.visualize and PLOTTING_LIBS_AVAILABLE and class_id not in visualized_classes:
                        vis_path = vis_output_dir / f"patch_preview_{class_name}.png"
                        _create_patch_visualization(patch, class_name, vis_path)
                        visualized_classes.add(class_id)

        # --- 6. Save Metadata Files ---
        total_patches_for_type = 0
        for split_name, metadata_list in all_patches_metadata.items():
            if not metadata_list:
                print(f"\nNo patches generated for the '{split_name}' set.")
                continue

            metadata_df = pd.DataFrame(metadata_list)
            metadata_path = split_dirs[split_name] / "metadata.csv"
            metadata_df.to_csv(metadata_path, index=False)
            
            num_patches = len(metadata_df)
            total_patches_for_type += num_patches
            print(f"\n'{split_name}' set: {num_patches} patches were generated.")
            print(f"  - Metadata file: {metadata_path}")
            print(f"  - Patches directory: {split_dirs[split_name] / 'patches'}")

        print(f"\nExtraction for '{current_tomo_type}' complete. A total of {total_patches_for_type} patches were generated in {output_dir_for_type}.")

    print(f"\nProcessing of all tomogram types finished. Data is located in {base_output_dir}.")

if __name__ == "__main__":
    main()
