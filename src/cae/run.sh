#!/bin/bash
#
#SBATCH --job-name=forecast-cae
#SBATCH -e outputs/errors/err_%j.err    # File to which STDERR will be written
#SBATCH --output=outputs/logs/%j.txt    # Output file
#SBATCH --partition=1080ti-short
#SBATCH --ntasks=18                     # Set to max_workers + 1
#SBATCH --time=00-04:00                 # Runtime in D-HH:MM
#SBATCH --mem=1600
#SBATCH --gres=gpu:8

python main.py
sleep 1
exit
