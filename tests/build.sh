#!/bin/bash

python -m venv pip_venv
source pip_venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt