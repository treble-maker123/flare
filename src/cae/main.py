import os
import yaml
import uuid
import torch
import pickle

from pathlib import Path
from time import time
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from pdb import set_trace

from engine import Engine
from utils.dir import mkdir

def main(config_path, run_id, tb_writer):
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
    engine = Engine(config, tb_writer)

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
        train_result, tally = engine.train()
        tb_writer.add_scalars("training/Num AD Correct", {
            "AD Correct": tally["AD"][0],
            "AD Total": tally["AD"][1]
        }, epoch)
        tb_writer.add_scalars("training/Num CN Correct", {
            "CN Correct": tally["CN"][0],
            "CN Total": tally["CN"][1]
        }, epoch)
        tb_writer.add_scalars("training/Num MCI Correct", {
            "MCI Correct": tally["MCI"][0],
            "MCI Total": tally["MCI"][1]
        }, epoch)
        tb_writer.add_scalars("training/Num Total Correct", {
            "All Correct": tally["Total"][0],
            "All Total": tally["Total"][1]
        }, epoch)
        tb_writer.add_scalars("training/Num AD vs CN Correct", {
            "AD Correct": tally["AD"][0],
            "CN Correct": tally["CN"][0]
        }, epoch)
        tb_writer.add_scalars("training/Num CN vs MCI Correct", {
            "CN Correct": tally["CN"][0],
            "MCI Correct": tally["MCI"][0]
        }, epoch)
        tb_writer.add_scalars("training/Num AD vs MCI Correct", {
            "AD Correct": tally["AD"][0],
            "MCI Correct": tally["MCI"][0]
        }, epoch)
        train_history.append(train_result)
        print("\tAverage training loss: {}"
                .format(train_result["average_loss"]))
        percent = round((tally["Total"][0] * 100.0) / tally["Total"][1], 2)
        tb_writer.add_scalar("training/Pct Correct", percent, epoch)

        # VALIDATION
        valid_result, tally = engine.validate()
        valid_history.append(valid_result)
        tb_writer.add_scalars("validation/Num AD Correct", {
            "AD Correct": tally["AD"][0],
            "AD Total": tally["AD"][1]
        }, epoch)
        tb_writer.add_scalars("validation/Num CN Correct", {
            "CN Correct": tally["CN"][0],
            "CN Total": tally["CN"][1]
        }, epoch)
        tb_writer.add_scalars("validation/Num MCI Correct", {
            "MCI Correct": tally["MCI"][0],
            "MCI Total": tally["MCI"][1]
        }, epoch)
        tb_writer.add_scalars("validation/Num Total Correct", {
            "All Correct": tally["Total"][0],
            "All Total": tally["Total"][1]
        }, epoch)
        tb_writer.add_scalars("validation/Num AD vs CN Correct", {
            "AD Correct": tally["AD"][0],
            "CN Correct": tally["CN"][0]
        }, epoch)
        tb_writer.add_scalars("validation/Num CN vs MCI Correct", {
            "CN Correct": tally["CN"][0],
            "MCI Correct": tally["MCI"][0]
        }, epoch)
        tb_writer.add_scalars("validation/Num AD vs MCI Correct", {
            "AD Correct": tally["AD"][0],
            "MCI Correct": tally["MCI"][0]
        }, epoch)
        num_correct = valid_result["num_correct"]
        num_total = valid_result["num_total"]

        print("\tAverage validation loss: {}"
                .format(valid_result["average_loss"]))
        if num_total > 0:
            percent = round(((num_correct * 1.0) / num_total) * 100, 2)
            print("\tAccuracy: {}/{} ({}%) correct."
                    .format(num_correct, num_total, percent))
            tb_writer.add_scalar("validation/Pct Correct", percent, epoch)

        tb_writer.add_scalars("misc/loss", {
            "Training": train_result["average_loss"],
            "Validation": valid_result["average_loss"]
        }, epoch)

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
    test_result, tally = engine.test()
    tb_writer.add_scalars("testing/stats", {
            "AD Correct": tally["AD"][0],
            "AD Total": tally["AD"][1],
            "CN Correct": tally["CN"][0],
            "CN Total": tally["CN"][1],
            "MCI Correct": tally["MCI"][0],
            "MCI Total": tally["MCI"][1],
            "Correct": tally["Total"][0],
            "Total": tally["Total"][1]
        }, 0)
    num_correct = test_result["num_correct"]
    num_total = test_result["num_total"]
    test_percent = round(((num_correct * 1.0) / num_total) * 100, 2)
    print("Final test results: {}/{} ({}%)".format(num_correct, num_total,
                                                   test_percent))
    tb_writer.add_scalar("testing/Pct Correct", test_percent, 0)

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

    tb_writer.close()
    print("Experiment finished in {} seconds."
            .format(round(time() - main_start)))
    print("----- END ({}) -----".format(run_id))

if __name__ == "__main__":
    # Make sure this is run from the correct directory
    current_dir = os.getcwd().split("/")
    assert current_dir[-3:] == ['disease_forecasting', 'src', 'cae'], \
            "Running from the wrong directory. Make sure to run \"python main.py\" from \"disease_forecasting/src/cae/\"."

    parser = ArgumentParser()

    tb_logs_path = "{}/tensorboard_logs".format(Path.home())
    mkdir(tb_logs_path)

    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--run_id", type=str,
                                    default=uuid.uuid4().hex.upper()[0:4])
    args = parser.parse_args()

    tb_writer = SummaryWriter(log_dir="{}/{}.log"
                                        .format(tb_logs_path, args.run_id))

    main(args.config, args.run_id, tb_writer)
