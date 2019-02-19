#!/bin/bash
#
#SBATCH --job-name=forecast-cae
#SBATCH -e outputs/errors/err_%j.err    # File to which STDERR will be written
#SBATCH --output=outputs/logs/%j.txt    # Output file
#SBATCH --partition=1080ti-short
#SBATCH --ntasks=34                     # Set to max_workers + 2
#SBATCH --time=00-04:00                 # Runtime in D-HH:MM
#SBATCH --mem=32000
#SBATCH --gres=gpu:8

python main.py
sleep 1
exit
