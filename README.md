# Uncertainty guided virtual screening of small molecules
This repository contains the scripts to perform uncertainty guided virtual screening by utilizing a deterministic pre-trained predictor model. 

![alt text](image.png)

# Install dependencies
The `basic_env.yml` file contains the required package information. Run the following command to create a conda environment for the project.

```
conda env create -f basic_env.yml
source activate vs_env
pip install -e .
```
# Train the predictor model
Following command will train a predictor model for DRD2. Currenlty the `train_surrogate.py` has implementation for $\text{DRD2}$, $\text{GSK}3\beta$ and $\text{JNK}3$. For a different property, the training/validation/test data split needs to defined under the `data` folder. Additionally minor edits are needed inside the `train_surrogate.py`.
```
python train_surrogate.py --prop_name=DRD2
```
# Enable UQ for the given predictor model
To perform UQ through AS, first we need to construct the active subspace around the pre-trained model weights, and learn the posterior distribution over the active subspace parameters by variational inference technique.

## Construct active subspace

```
python run_active_subspace_construction.py --prop_name=DRD2 --AS_dim=10
```

## Approximate active subspace posterior distribution

```
python run_vi_training.py --prop_name=DRD2 --AS_dim=10
```

# Perform prediction via Bayesian inference of the predictor enabled by the AS posterior

```
python run_screening_AS_pred.py --prop_name=DRD2 --AS_dim=10 --trial=0 --num_models=10
```

# Reproducing the experiment

`job_script.sh` has the python commands for performing the experiment. Note that, the machine needs to have GPU, otherwise it will take longer time.

```
source activate vs_env
./job_script.sh
```
`UQ_guided_virtual_screening.ipynb` has the code to perform the screening process for  $\text{DRD2}$ and $\text{GSK}3\beta$ and compute the hit rate for different screening threshold, active subspace dimension and number of model samples in Bayesian inference. It also has the code for visualizing the standard deviation of hit rate across 5 trials of each property. 