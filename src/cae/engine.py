import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp

from torch.utils.data import DataLoader
from pdb import set_trace

from dataset import ADNIAutoEncDataset, ADNIClassDataset, ADNIAeCnnDataset
from models.vanilla_cae import VanillaCAE
from models.transform_cae import SpatialTransformConvAutoEnc
from models.classifier import Classify
<<<<<<< HEAD
from models.hosseini import Hosseini
=======
from models.ae_cnn import AE_CNN
>>>>>>> 47a4c941917428d5d919b7fd62404f5cfc43f083

from utils.loader import invalid_collate

class Engine:
    def __init__(self, config, **kwargs):
        device = kwargs.get("device", None)
        model_path = kwargs.get("model_path", None)

        self._config = config
        self._setup_device(device)
        self._setup_model(model_path)
        self._setup_data()

    def pretrain(self):
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

        for num_iter, (x, _) in enumerate(self.pretrain_loader):
            optimizer.zero_grad()

            x = x.to(device=device).float()

            if type(model) == torch.nn.DataParallel:
                output = model.module.reconstruct(x)
                loss = model.module.reconstruction_loss(output, x)
            else:
                loss = model.reconstruction_loss(output, x)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if num_iter % print_iter == 0:
                print("\tIteration {}: {}".format(num_iter, loss.item()))

        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses)
        }

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
            y = y.to(device=device).long()

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
        num_correct = 0
        num_total = 0

        with torch.no_grad():
            for num_iter, (x, y) in enumerate(self.valid_loader):
                x = x.to(device=device).float()
                y = y.to(device=device).long()

                pred = model(x)

                if type(model) == torch.nn.DataParallel:
                    loss = model.module.loss(pred, y)
                else:
                    loss = model.loss(pred, y)

                if y.dim == 1: # classification
                    pred = pred.argmax(dim=1)
                    num_correct += (y == pred).sum()
                    num_total += len(y)

                losses.append(loss.item())

        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses),
            "num_correct": num_correct,
            "num_total": num_total
        }

    def test(self):
        device = self._device
        config = self._config
        model = self._model

        model = model.to(device=device)
        model.eval()

        losses = []
        num_correct = 0
        num_total = 0

        with torch.no_grad():
            for num_iter, (x, y) in enumerate(self.test_loader):
                x = x.to(device=device).float()
                y = y.to(device=device).long()

                pred = model(x)

                if type(model) == torch.nn.DataParallel:
                    loss = model.module.loss(pred, y)
                else:
                    loss = model.loss(pred, y)

                if y.dim == 1: # classification
                    pred = pred.argmax(dim=1)
                    num_correct += (y == pred).sum()
                    num_total += len(y)

                losses.append(loss.item())

        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses),
            "num_correct": num_correct,
            "num_total": num_total
        }

    def save_model(self, path, **kwargs):
        device = kwargs.get("device", self._device)
        model = self._model.to(device=device)

        if type(model) == torch.nn.DataParallel:
            torch.save(model.module.state_dict(), path)
        else:
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

    def _setup_model(self, model_path=None):
        config = self._config
        model_class = config["model"]['class']

        if model_class == "vanilla_cae":
            print("Using vanilla_cae model.")
            self._model = VanillaCAE()
        elif model_class == "transformer":
            print("Using transformer model.")
            self._model = SpatialTransformConvAutoEnc()
        elif model_class == "classify":
            print("Using classify model.")
            self._model = Classify()
        elif model_class == "hosseini":
            print("Using Hosseini model.")
            self._model = Hosseini()
        elif model_class == "ae_cnn_patches":
            print("Using ae cnn pathces model.")
            self._model = AE_CNN()
        else:
            raise Exception("Unrecognized model: {}".format(model_class))

        if model_path is not None:
            self._model.load_state_dict(torch.load(model_path))

        if self._use_gpu and self._gpu_count > 1:
            self._model = nn.DataParallel(self._model)

    def _setup_data(self):
        config = self._config
        num_workers = min(mp.cpu_count() // 2, config["data"]["max_workers"])
        num_workers = max(num_workers, 1)

        pretrain_dataset_params = {
            "mode": "all",
            "task": "pretrain",
            "valid_split": 0.0,
            "test_split": 0.0,
            "limit": config["data"]["limit"]
        }
        train_dataset_params = {
            "mode": "train",
            "task": "classify",
            "valid_split": config["data"]["valid_split"],
            "test_split": config["data"]["test_split"],
            "limit": config["data"]["limit"]
        }
        valid_dataset_params = {
            "mode": "valid",
            "task": "classify",
            "valid_split": config["data"]["valid_split"],
            "test_split": config["data"]["test_split"],
            "limit": config["data"]["limit"]
        }
        test_dataset_params = {
            "mode": "test",
            "task": "classify",
            "valid_split": config["data"]["valid_split"],
            "test_split": config["data"]["test_split"],
            "limit": config["data"]["limit"]
        }

        pretrain_loader_params = {
            "batch_size": config["train"]["batch_size"],
            "num_workers": num_workers,
            "collate_fn": invalid_collate,
            "shuffle": True
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
        test_loader_params = {
            "batch_size": config["test"]["batch_size"],
            "num_workers": num_workers,
            "collate_fn": invalid_collate,
            "shuffle": True
        }

        if self._config["data"]['set_name'] == "autoenc":
            self.train_dataset = ADNIAutoEncDataset(**train_dataset_params)
            self.valid_dataset = ADNIAutoEncDataset(**valid_dataset_params)
            self.test_dataset = ADNIAutoEncDataset(**test_dataset_params)
        elif self._config["data"]["set_name"] == "classify":
            self.pretrain_dataset = ADNIClassDataset(**pretrain_dataset_params)
            self.train_dataset = ADNIClassDataset(**train_dataset_params)
            self.valid_dataset = ADNIClassDataset(**valid_dataset_params)
            self.test_dataset = ADNIClassDataset(**test_dataset_params)
        elif self._config["data"]["set_name"] == "ae_cnn_patches":
            self.pretrain_dataset = ADNIAeCnnDataset(**pretrain_dataset_params)
            self.train_dataset = ADNIAeCnnDataset(**train_dataset_params)
            self.valid_dataset = ADNIAeCnnDataset(**valid_dataset_params)
            self.test_dataset = ADNIAeCnnDataset(**test_dataset_params)
        self.pretrain_loader = DataLoader(self.pretrain_dataset,
                                          **pretrain_loader_params)
        self.train_loader = DataLoader(self.train_dataset,
                                       **train_loader_params)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       **valid_loader_params)
        self.test_loader = DataLoader(self.test_dataset,
                                      **test_loader_params)

        print("{} training data, {} validation data, {} test data"
                .format(len(self.train_dataset),
                        len(self.valid_dataset),
                        len(self.test_dataset)))
