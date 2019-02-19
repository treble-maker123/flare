#!/bin/bash
#
#SBATCH --job-name=forecast-cae
#SBATCH -e outputs/errors/err_%j.err    # File to which STDERR will be written
#SBATCH --output=outputs/logs/%j.txt    # Output file
#SBATCH --partition=1080ti-long
#SBATCH --ntasks=34                     # Set to max_workers + 2
#SBATCH --time=02-00:00                 # Runtime in D-HH:MM
#SBATCH --mem=32000
#SBATCH --gres=gpu:4

python main.py
sleep 1
exit
