#!/bin/bash

if [ -f "pip_env/bin/activate" ]; then
    source pip_env/bin/activate
    echo "Activated virtual environment"
    python train_surrogate.py --prop_name=DRD2 --epochs=1 --num_workers=0
    deactivate
else
    echo "env not found"
fi

