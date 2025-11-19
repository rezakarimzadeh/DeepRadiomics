#!/bin/bash

python3 main.py --model_name transformer --use_tda False
python3 main.py --model_name deep_sets --use_tda False
python3 main.py --model_name mil --use_tda False
python3 main.py --model_name ml_models --use_tda False

python3 main.py --model_name transformer --use_tda True
python3 main.py --model_name deep_sets --use_tda True
python3 main.py --model_name mil --use_tda True
python3 main.py --model_name ml_models --use_tda True