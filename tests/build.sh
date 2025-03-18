#!/bin/bash

[ ! -d "UQ_VS/" ] && git clone https://github.com/RECUP-DOE/UQ_VS.git
cd UQ_VS

python -m venv pip_venv
source pip_venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt