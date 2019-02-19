import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp

from torch.utils.data import DataLoader
from pdb import set_trace

from dataset import ADNIAutoEncDataset
from models.vanilla import Vanilla

from utils.loader import invalid_collate

class Engine:
    def __init__(self, config, **kwargs):
        device = kwargs.get("device", None)

        self._config = config
        self._setup_device(device)
        self._setup_model()
        self._setup_data()

    def train(self):
        '''
        Execute the training loop.

        Args:
            **kwargs:
                print_iter (int): How often (number of iterations) to print output.
        '''
        device = self._device
        config = self._config
        model = self._model

        print_iter = config["train"]["print_iter"]

        model = model.to(device=device)
        model.train()

        optim_params = {
            "lr": config["train"]["optim"]["learn_rate"],
            "weight_decay": config["train"]["optim"]["weight_decay"]
        }
        optimizer = optim.Adam(model.parameters(), **optim_params)

        losses = []

        for num_iter, (x, y) in enumerate(self.train_loader):
            optimizer.zero_grad()

            x = x.to(device=device).float()
            y = y.to(device=device).float()

            output = model(x)

            if type(model) == torch.nn.DataParallel:
                loss = model.module.loss(output, y)
            else:
                loss = model.loss(output, y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if num_iter % print_iter == 0:
                print("\tIteration {}: {}".format(num_iter, loss.item()))

        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses)
        }

    def validate(self):
        device = self._device
        config = self._config
        model = self._model

        model = model.to(device=device)
        model.eval()

        losses = []

        with torch.no_grad():
            for num_iter, (x, y) in enumerate(self.valid_loader):
                x = x.to(device=device).float()
                y = y.to(device=device).float()

                pred = model(x)

                if type(model) == torch.nn.DataParallel:
                    loss = model.module.loss(pred, y)
                else:
                    loss = model.loss(pred, y)

                losses.append(loss.item())

        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses)
        }

    def save_model(self, path, **kwargs):
        device = kwargs.get("device", self._device)
        model = self._model.to(device=device)

        torch.save(model.state_dict(), path)

    def _setup_device(self, device):
        '''
        Set the device (CPU vs GPU) used for the experiment. Defaults to GPU if available.

        Args:
            device (torch.device, optional): Defaults to None. If supplied, this will be used instead.
        '''
        if device is not None:
            self._device = device
            return

        cuda_available = torch.cuda.is_available()
        self._use_gpu = self._config["use_gpu"]
        self._gpu_count = torch.cuda.device_count()

        if cuda_available and self._use_gpu and self._gpu_count > 0:
            print("{} GPUs detected, running in GPU mode."
                    .format(self._gpu_count))
            self._device = torch.device("cuda")
        else:
            print("Running in CPU mode.")
            self._device = torch.device("cpu")

    def _setup_model(self):
        config = self._config
        model_class = config["model"]['class']

        if model_class == "vanilla":
            print("Using vanilla model.")
            self._model = Vanilla()
        else:
            raise Exception("Unrecognized model: {}".format(model_class))

        if self._use_gpu and self._gpu_count > 1:
            self._model = nn.DataParallel(self._model)

    def _setup_data(self):
        config = self._config
        num_workers = min(mp.cpu_count() // 2, config["data"]["max_workers"])
        num_workers = max(num_workers, 1)

        train_dataset_params = {
            "mode": "train",
            "valid_split": config["data"]["valid_split"]
        }
        valid_dataset_params = {
            "mode": "valid",
            "valid_split": config["data"]["valid_split"]
        }

        train_loader_params = {
            "batch_size": config["train"]["batch_size"],
            "num_workers": num_workers,
            "collate_fn": invalid_collate,
            "shuffle": True
        }
        valid_loader_params = {
            "batch_size": config["valid"]["batch_size"],
            "num_workers": num_workers,
            "collate_fn": invalid_collate,
            "shuffle": True
        }

        self.train_dataset = ADNIAutoEncDataset(**train_dataset_params)
        self.valid_dataset = ADNIAutoEncDataset(**valid_dataset_params)
        self.train_loader = DataLoader(self.train_dataset,
                                       **train_loader_params)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       **valid_loader_params)
        print("{} training data, {} validation data"
                .format(len(self.train_dataset), len(self.valid_dataset)))
