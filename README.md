# UQ_VS
UQ_guided virtual screening

# Train the predictor model

```
python train_surrogate.py --prop_name=DRD2
```

# Construct active subspace

```
python run_active_subspace_construction.py --prop_name=DRD2 --AS_dim=10
```

# Approximate active subspace posterior distribution

```
python run_vi_training.py --prop_name=DRD2 --AS_dim=10
```

# Perform prediction via Bayesian inference of the predictor enabled by the AS posterior

```
python run_screening_AS_pred.py --prop_name=DRD2 --AS_dim=10 --trial=0
```