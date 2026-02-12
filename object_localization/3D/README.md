# Instructions to reproduce the localization experiments in 3D

## Data

A small example of our synthetic dataset (one image, along with its segmentation mask and the GT score map) is provided (see `example_set.tag.gz`) for vizualization but the entire set could not be provided do to memory limitations.

If one wants to reproduce these experiments, ask for it during the reviewing process.

## Training

Fill the config `train.yaml` file in the config directory with the localization of the **train_set**, **val_set** and the directory to store the models **save/path**.

Indicate the model to run in the **experiment** field, choose among those stored in `configs/experiment`.

Indicate the data augmentation to apply using the **data_aug**, choose among those stored in `configs/data_aug`.

Fix the seed using the field **exp_group**.

We used the seeds 1,2,3,4 and 5 for our experiments.

Launch `python training.py` to run the training using the configuration described by the config file `train.yaml`.

## Testing

### Experiment on data augmentation

Fill the config `test.yaml` file with:

- **input_dir**: the path to the test set dir.
- **output_dir**: the path to the directory where the results are stored.
- **models_dir**: the path to the directory where the trained models are stored (the **save/path** from the training config).
- **which_model**: the name of the model directory (containing the weights and config file) to test.
- **group_idx**: the seed used to train the model (used to find the model in the model dir because they are grouped by seed).

Launching `python testing.py` will run the inference on the test for every image corruption type on this model.

The results are `.npz` files storing the accuracy curve from which the score can be computed.

### Experiment on bright artifacts

File the `exp_arti.yaml` config file in the same manner than for the previous experiment, except you indicate the set containing images corrupted with artifacts.

This set should be organized as follow:

```bash
directory
|-arti_<artifact intensity 1>
  |-img_1.tif
  |-img_2.tif 
  ...
|-arti_<artifact intensity 2>
  |-img_1.tif
  |-img_2.tif 
  ...
...
|-arti_<artifact intensity n>
  |-img_1.tif
  |-img_2.tif 
  ...
|-masks
  |-mask_1.tif
  |-mask_2.tif
  ...
```

Launching `python artifacts_exp.py` will run the inference on the artifact set for every artifact intensity on this model.

The results are `.npz` files storing the accuracy curve from which the score can be computed.

## Generate tables

If you have run all **model+augmentation** combinations, you can generate the result table for that seed using script `generate_res_table.py -i <path to the inference ouputs dir storing .npz>`

It will generate an `.npz` at this location summarizing the information.

To merge all the seeds results, put all the `.npz` table files in a dir and run `merge_tabs.py`, it will run generate multiple `.csv` files aggregating the results for the different seeds for each model and data augmentation. The `mean_tab.csv` file corresponds to the latex tables giving the results of our paper.
