#!/bin/bash

# Run set_transformer model with all combinations of use_coords and use_demographic
python3 main.py --model_name set_transformer --use_coords False --use_demographic False
python3 main.py --model_name set_transformer --use_coords True --use_demographic False
python3 main.py --model_name set_transformer --use_coords False --use_demographic True
python3 main.py --model_name set_transformer --use_coords True --use_demographic True

# Run Transformer model with all combinations of use_coords and use_demographic
# python3 main.py --model_name transformer --use_coords False --use_demographic False
# python3 main.py --model_name transformer --use_coords True --use_demographic False
# python3 main.py --model_name transformer --use_coords False --use_demographic True
# python3 main.py --model_name transformer --use_coords True --use_demographic True

# Run graph-based model with all combinations of use_coords and use_demographic
# python3 main.py --model_name graph --use_coords False --use_demographic False
# python3 main.py --model_name graph --use_coords True --use_demographic False
# python3 main.py --model_name graph --use_coords False --use_demographic True
# python3 main.py --model_name graph --use_coords True --use_demographic True

# Run ml_models with all combinations of use_coords and use_demographic
# python3 main.py --model_name ml_models --use_coords False --use_demographic False
# python3 main.py --model_name ml_models --use_coords True --use_demographic False
# python3 main.py --model_name ml_models --use_coords False --use_demographic True
# python3 main.py --model_name ml_models --use_coords True --use_demographic True

# python3 main.py --model_name transformer --use_coords False --use_demographic False
# python3 main.py --model_name deep_sets --use_coords False --use_demographic False
# python3 main.py --model_name mil --use_coords False --use_demographic False
# python3 main.py --model_name graph --use_coords False --use_demographic False
# python3 main.py --model_name ml_models --use_coords False --use_demographic False

# python3 main.py --model_name transformer --use_coords True --use_demographic False
# python3 main.py --model_name deep_sets --use_coords True --use_demographic False
# python3 main.py --model_name mil --use_coords True --use_demographic False
# python3 main.py --model_name graph --use_coords True --use_demographic False
# python3 main.py --model_name ml_models --use_coords True --use_demographic False

# python3 main.py --model_name transformer --use_coords False --use_demographic True
# python3 main.py --model_name deep_sets --use_coords False --use_demographic True
# python3 main.py --model_name mil --use_coords False --use_demographic True
# python3 main.py --model_name graph --use_coords False --use_demographic True
# python3 main.py --model_name ml_models --use_coords False --use_demographic True

# python3 main.py --model_name transformer --use_coords True --use_demographic True
# python3 main.py --model_name deep_sets --use_coords True --use_demographic True
# python3 main.py --model_name mil --use_coords True --use_demographic True
# python3 main.py --model_name graph --use_coords True --use_demographic True
# python3 main.py --model_name ml_models --use_coords True --use_demographic True