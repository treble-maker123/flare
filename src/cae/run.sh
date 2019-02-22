#!/bin/bash
#
#SBATCH --job-name=forecast-cae
#SBATCH -e outputs/errors/%j.txt    # File to which STDERR will be written
#SBATCH --output=outputs/logs/%j.txt    # Output file
#SBATCH --partition=1080ti-long
#SBATCH --ntasks=12                      # Set to max_workers + 2
#SBATCH --time=02-00:00                 # Runtime in D-HH:MM
#SBATCH --mem=240000
#SBATCH --gres=gpu:2


python3 main.py --run_id=$SLURM_JOB_ID --config="config/classify.yaml"
sleep 1
exit
