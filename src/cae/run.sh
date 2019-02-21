#!/bin/bash
#
#SBATCH --job-name=forecast-cae
#SBATCH -e outputs/errors/err_%j.err    # File to which STDERR will be written
#SBATCH --output=outputs/logs/%j.txt    # Output file
#SBATCH --partition=1080ti-long
#SBATCH --ntasks=2                     # Set to max_workers + 2
#SBATCH --time=02-00:00                 # Runtime in D-HH:MM
#SBATCH --mem=24000
#SBATCH --gres=gpu:1

#python main.py
python classifier.py
sleep 1
exit
