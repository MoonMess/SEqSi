# Instructions to reproduce the localization experiments in 2D

## Data

The fluorescence subset of the DSB-2018 set can be downloaded at [https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip](https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip)

We kept the test set as it is and generated a val set by extracting 25 images from the training set.

To create the GT score maps, use:
`python data/pre-process/generate_score.py -i <path to the masks dir> -o <path to the score maps dir>`

We recommand to name the score maps dir **samplings** to match with the requirements of the Pytorch datasets used in our code.

## Training

Fill the config `train.yaml` file in the config directory with the localization of the **train_set**, **val_set** and the directory to store the models **save/path**.

Indicate the model to run in the **experiment** field, choose among those stored in `configs/experiment`.

Fix the seed using the field **seed**.

We used the seed 1 in our experiments.

Launch `python training.py` to run the training using the configuration described by the config file `train.yaml`.

## Testing

### Experiment on invariance

Fill the config `inv_exp.yaml` file with:

- **input_dir**: the path to the test set dir.
- **output_dir**: the path to the directory where the results are stored.
- **models_dir**: the path to the directory where the trained models are stored (the **save/path** from the training config).
- **which_model**: the name of the model directory (containing the weights and config file) to test.

Launching `python inv_experiments.py` will run the inference on the test set for different scale, shift and affine corrpution to test the invariance by comparing with prediction with no corruption.

The results are `.npz` files storing the accuracy curve, the first point of this curve corresponds to acc(d=0) which should be equal to 1 if the model is invariant.

## Generate tables

If you have run all **model** invariance prediction, you can generate the result table using script `generate_res_table.py -i <path to the inference ouputs dir storing .npz>`

It will generate an `.csv` at this location summarizing the information.
