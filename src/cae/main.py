import os
import yaml
import uuid
import torch

from time import time
from argparse import ArgumentParser
from pdb import set_trace

from engine import Engine
from utils.dir import mkdir

def main(config_path, run_id):
    # https://github.com/pytorch/pytorch/issues/1485
    torch.backends.cudnn.benchmark=True

    with open(config_path) as file:
        config = yaml.load(file)

    print("----- START ({}) -----".format(run_id))
    print("Following configurations are used for this run:")
    print("")
    print(yaml.dump(config, default_flow_style=False))
    print("")

    main_start = time()

    # Make the output directories
    mkdir("outputs/errors")
    mkdir("outputs/logs")
    mkdir("outputs/stats")
    mkdir("weights/{}".format(run_id))

    num_epochs = config["train"]["num_epochs"]
    engine = Engine(config)

    train_history = []
    valid_history = []
    lowest_losses = [float("inf")] * 5

    for epoch in range(num_epochs):
        epoch_start = time()
        print("Starting epoch {}:".format(epoch + 1))
        train_result = engine.train()
        train_history.append(train_result)
        print("\tAverage training loss: {}"
                .format(train_result["average_loss"]))

        valid_result = engine.validate()
        valid_history.append(valid_result)
        print("\tAverage validation loss: {}"
                .format(valid_result["average_loss"]))

        # Five lowest loss models are saved
        loss_idx = 0
        current_highest = lowest_losses[0]
        # Find the highest error entry to be replaced
        for idx, loss in enumerate(lowest_losses):
            if loss > current_highest:
                loss_idx = idx
                current_highest = loss

        if valid_result["average_loss"] < current_highest:
            lowest_losses[loss_idx] = valid_result["average_loss"]
            file_name = "weights/{}/{}.pt".format(run_id, loss_idx)
            engine.save_model(file_name)
            print("\tModel saved as {}.".format(file_name))

        elapsed_time = time() - epoch_start
        print("Epoch {} completed in {} seconds."
                .format(epoch, round(elapsed_time)))

    print("Experiment finished in {} seconds."
            .format(round(time() - main_start)))
    print("----- END ({}) -----".format(run_id))

if __name__ == "__main__":
    # Make sure this is run from the correct directory
    current_dir = os.getcwd().split("/")
    assert current_dir[-3:] == ['disease_forecasting', 'src', 'cae'], \
            "Running from the wrong directory. Make sure to run \"python main.py\" from \"disease_forecasting/src/cae/\"."

    parser = ArgumentParser()

    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--run_id", type=str,
                                    default=uuid.uuid4().hex.upper()[0:6])

    args = parser.parse_args()
    main(args.config, args.run_id)
