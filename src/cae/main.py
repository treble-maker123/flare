import os
import yaml
import uuid
import torch
import pickle

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
    mkdir("outputs/weights")
    mkdir("outputs/weights/{}".format(run_id))

    num_epochs = config["train"]["num_epochs"]
    engine = Engine(config)

    pretrain_history = []
    train_history = []
    valid_history = []
    lowest_losses = [float("inf")] * 5

    # PRETRAINING
    if config["pretrain"]["num_epochs"] > 0:
        for epoch in range(config["pretrain"]["num_epochs"]):
            epoch_start = time()
            print("Starting pretraining epoch {}:".format(epoch + 1))

            pretrain_result = engine.pretrain()
            pretrain_history.append(pretrain_result)

            print("\tAverage training loss: {}"
                .format(pretrain_result["average_loss"]))
    else:
        print("Skipping pretraining.")

    for epoch in range(num_epochs):
        epoch_start = time()

        # TRAINING
        print("Starting training epoch {}:".format(epoch + 1))
        train_result = engine.train()
        train_history.append(train_result)
        print("\tAverage training loss: {}"
                .format(train_result["average_loss"]))

        # VALIDATION
        valid_result = engine.validate()
        valid_history.append(valid_result)
        num_correct = valid_result["num_correct"]
        num_total = valid_result["num_total"]

        print("\tAverage validation loss: {}"
                .format(valid_result["average_loss"]))
        if num_total > 0:
            percent = round(((num_correct * 1.0) / num_total) * 100, 2)
            print("\tAccuracy: {}/{} ({}%) correct."
                    .format(num_correct, num_total, percent))

        # CHECKPOINT
        # Five lowest loss models are saved
        current_highest = max(lowest_losses)
        highest_loss_idx = lowest_losses.index(max(lowest_losses))

        if valid_result["average_loss"] < current_highest:
            lowest_losses[highest_loss_idx] = valid_result["average_loss"]
            file_name = "outputs/weights/{}/{}.pt" \
                            .format(run_id, highest_loss_idx)
            engine.save_model(file_name)
            print("\tModel saved as {}.".format(file_name))

        elapsed_time = time() - epoch_start
        print("Epoch {} completed in {} seconds."
                .format(epoch + 1, round(elapsed_time)))

    # TESTING
    print("Starting test...")
    print("Top 5 lowest losses: {}".format(lowest_losses))
    lowest_loss_idx = lowest_losses.index(min(lowest_losses))
    file_name = "outputs/weights/{}/{}.pt" \
                    .format(run_id, lowest_loss_idx)
    print("Loading model with lowest loss for testing.")
    engine.load_model()
    test_result = engine.test()
    num_correct = test_result["num_correct"]
    num_total = test_result["num_total"]
    test_percent = round(((num_correct * 1.0) / num_total) * 100, 2)
    print("Final test results: {}/{} ({}%)".format(num_correct, num_total,
                                                   test_percent))

    print("Writing statistics to file")
    statistics = {
        "pretrain_history": pretrain_history,
        "train_history": train_history,
        "valid_history": valid_history,
        "lowest_losses": lowest_losses,
        "test_accuracy": test_percent
    }
    with open("outputs/stats/{}.pickle".format(run_id), "wb") as file:
        pickle.dump(statistics, file)

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
                                    default=uuid.uuid4().hex.upper()[0:4])

    args = parser.parse_args()
    main(args.config, args.run_id)
