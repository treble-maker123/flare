import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from random import randint

from torch.utils.data import DataLoader

from dataset import ADNIAutoEncDataset, ADNIClassDataset, ADNIAeCnnDataset
from normalized_dataset import NormalizedDataset
from models.vanilla_cae import VanillaCAE
from models.transform_cae import SpatialTransformConvAutoEnc
from models.classifier import Classify
from models.hosseini import Hosseini, HosseiniThreeLayer, HosseiniSimple, HosseiniDeep
from models.ae_cnn import AE_CNN
from models.two_d import TwoD
from models.deep_mri import DeepMRI
from models.deep_ae_mri import DeepAutoencMRI
from slice_dataset import SliceDataset

from utils.loader import invalid_collate

from pdb import set_trace

class Engine:
    def __init__(self, config, tb_writer, logger, **kwargs):
        device = kwargs.get("device", None)
        model_path = kwargs.get("model_path", None)

        self.writer = tb_writer
        self.logger = logger
        self._config = config
        self._setup_device(device)
        self.load_model(model_path)
        self._setup_data()

    def pretrain(self, epoch):
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
        # optimizer = optim.Adam(model.parameters(), **optim_params)
        optimizer = optim.SGD(model.parameters(), momentum=0.9, **optim_params)

        losses = []

        for num_iter, (x, y) in enumerate(self.pretrain_loader):
            optimizer.zero_grad()

            x = x.to(device=device).float()

            if type(model) == torch.nn.DataParallel:
                output = model.module.reconstruct(x)
                loss = model.module.reconstruction_loss(output, x)
            else:
                output = model.reconstruct(x)
                loss = model.reconstruction_loss(output, x)

            loss.backward()
            optimizer.step()

            output = output.detach()

            outputs = []
            if output.shape[1] == 3:
                outputs.append(output[:, 0, :, :, :])
                outputs.append(output[:, 1, :, :, :])
                outputs.append(output[:, 2, :, :, :])
            else:
                outputs.append(output)

            if randint(1, 100) <= 10:
                # randomly pick one image to write to tensorboard
                img_idx = randint(0, len(output) - 1)
                # pick the middle slice
                mid_idx = output.shape[-1] // 2
                for idx, o in enumerate(outputs):
                    self.writer.add_image("Channel {}; 1st idx".format(idx+1),
                                        o[img_idx, mid_idx, :, :].unsqueeze(0),
                                        global_step=epoch, dataformats="CHW")
                    self.writer.add_image("Channel {}; 2nd idx".format(idx+1),
                                        o[img_idx, :, mid_idx, :].unsqueeze(0), global_step=epoch, dataformats="CHW")
                    self.writer.add_image("Channel {}; 3rd idx".format(idx+1),
                                        o[img_idx, :, :, mid_idx].unsqueeze(0), global_step=epoch, dataformats="CHW")

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
            "weight_decay": config["train"]["optim"]["weight_decay"],
            # "momentum": config["train"]["optim"]["momentum"]
        }
        optimizer = optim.Adam(model.parameters(), **optim_params)
        # optimizer = optim.SGD(model.parameters(), **optim_params)

        losses = []
        tally = {
            # [ num correct, total ]
            "AD": [0, 0],
            "CN": [0, 0],
            "MCI": [0, 0],
            "Total": [0, 0]
        }

        for num_iter, (x, y) in enumerate(self.train_loader):
            optimizer.zero_grad()

            x = x.to(device=device).float()
            y = y.to(device=device).long()

            pred = model(x)

            if type(model) == torch.nn.DataParallel:
                loss = model.module.loss(pred, y)
            else:
                loss = model.loss(pred, y)

            if y.dim() == 1: # classification
                with torch.no_grad():
                    pred = pred.argmax(dim=1)

                    label_encoder = self.train_loader.dataset.label_encoder
                    correct = (y.cpu().numpy() == pred.cpu().numpy())
                    labels = label_encoder.inverse_transform(y.cpu().numpy())
                    tally = self._add_to_tally(labels, correct, tally)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses)
        }, tally

    def validate(self):
        device = self._device
        config = self._config
        model = self._model

        model = model.to(device=device)
        model.eval()

        losses = []
        num_correct = 0
        num_total = 0
        tally = {
            # [ num correct, total ]
            "AD": [0, 0],
            "CN": [0, 0],
            "MCI": [0, 0],
            "Total": [0, 0]
        }

        with torch.no_grad():
            for num_iter, (x, y) in enumerate(self.valid_loader):
                x = x.to(device=device).float()
                y = y.to(device=device).long()

                pred = model(x)

                if type(model) == torch.nn.DataParallel:
                    loss = model.module.loss(pred, y)
                else:
                    loss = model.loss(pred, y)

                if y.dim() == 1: # classification
                    pred = pred.argmax(dim=1)
                    num_correct += (y == pred).sum().item()
                    num_total += len(y)

                    label_encoder = self.train_loader.dataset.label_encoder
                    correct = (y.cpu().numpy() == pred.cpu().numpy())
                    labels = label_encoder.inverse_transform(y.cpu().numpy())
                    tally = self._add_to_tally(labels, correct, tally)

                losses.append(loss.item())

        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses),
            "num_correct": num_correct,
            "num_total": num_total
        }, tally

    def test(self):
        device = self._device
        config = self._config
        model = self._model

        model = model.to(device=device)
        model.eval()

        losses = []
        num_correct = 0
        num_total = 0
        tally = {
            # [ num correct, total ]
            "AD": [0, 0],
            "CN": [0, 0],
            "MCI": [0, 0],
            "Total": [0, 0]
        }

        with torch.no_grad():
            for num_iter, (x, y) in enumerate(self.test_loader):
                x = x.to(device=device).float()
                y = y.to(device=device).long()

                pred = model(x)

                if type(model) == torch.nn.DataParallel:
                    loss = model.module.loss(pred, y)
                else:
                    loss = model.loss(pred, y)

                if y.dim() == 1: # classification
                    pred = pred.argmax(dim=1)
                    num_correct += (y == pred).sum().item()
                    num_total += len(y)

                    label_encoder = self.train_loader.dataset.label_encoder
                    correct = (y.cpu().numpy() == pred.cpu().numpy())
                    labels = label_encoder.inverse_transform(y.cpu().numpy())
                    tally = self._add_to_tally(labels, correct, tally)

                losses.append(loss.item())

        pct_correct = round((tally["Total"][0] * 100.0) / tally["Total"][1], 2)
        print("\tTest correct: AD {}/{}, CN {}/{}, MCI {}/{}, total {}/{}({}%)"
                .format(tally["AD"][0], tally["AD"][1], tally["CN"][0],
                        tally["CN"][1], tally["MCI"][0], tally["MCI"][1],
                        tally["Total"][0], tally["Total"][1], pct_correct))

        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses),
            "num_correct": num_correct,
            "num_total": num_total
        }, tally

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

    def load_model(self, model_path=None):
        config = self._config
        model_class = config["model"]['class']
        n_channels = len(config["image_col"])

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
        elif model_class == "hosseini_three_layer":
            print("Using Three-Layer Hosseini model.")
            self._model = HosseiniThreeLayer(num_channels=n_channels)
        elif model_class == "hosseini_simple":
            print("Using simple Hosseini model with two-layer autoencoder.")
            self._model = HosseiniSimple()
        elif model_class == "hosseini_deep":
            print("Using deep Hosseini model.")
            self._model = HosseiniDeep(num_channels=n_channels)
        elif model_class == "deep_mri":
            print("Using deep MRI model.")
            self._model = DeepMRI(num_channels=n_channels)
        elif model_class == "deep_ae_mri":
            print("Using deep AE MRI model")
            self._model = DeepAutoencMRI(num_channels=n_channels)
        elif model_class == "2d":
            print("Using 2D deep learning model.")
            n_channels = len(config["image_col"])
            self._model = TwoD(num_channels=n_channels)
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
            "limit": config["data"]["limit"],
            "config": config
        }
        train_dataset_params = {
            "mode": "train",
            "task": "classify",
            "valid_split": config["data"]["valid_split"],
            "test_split": config["data"]["test_split"],
            "limit": config["data"]["limit"],
            "config": config
        }
        valid_dataset_params = {
            "mode": "valid",
            "task": "classify",
            "valid_split": config["data"]["valid_split"],
            "test_split": config["data"]["test_split"],
            "limit": config["data"]["limit"],
            "config": config
        }
        test_dataset_params = {
            "mode": "test",
            "task": "classify",
            "valid_split": config["data"]["valid_split"],
            "test_split": config["data"]["test_split"],
            "limit": config["data"]["limit"],
            "config": config
        }

        pretrain_loader_params = {
            "batch_size": config["pretrain"]["batch_size"],
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
        elif self._config["data"]["set_name"] == "normalized":
            self.pretrain_dataset = NormalizedDataset(**pretrain_dataset_params)
            self.train_dataset = NormalizedDataset(**train_dataset_params)
            # Ensure the label and its encoded counter part match.
            label_encoder = self.train_dataset.label_encoder
            self.valid_dataset = NormalizedDataset(label_encoer= label_encoder,
                                                    **valid_dataset_params)
            self.test_dataset = NormalizedDataset(label_encoer= label_encoder,
                                                    **test_dataset_params)
        elif self._config["data"]["set_name"] == "2d":
            self.pretrain_dataset = SliceDataset(**pretrain_dataset_params)
            self.train_dataset = SliceDataset(**train_dataset_params)
            # Ensure the label and its encoded counter part match.
            label_encoder = self.train_dataset.label_encoder
            self.valid_dataset = SliceDataset(label_encoer= label_encoder,
                                                    **valid_dataset_params)
            self.test_dataset = SliceDataset(label_encoer= label_encoder,
                                                    **test_dataset_params)

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

    def _add_to_tally(self, labels, gt, tally):
        ad = labels == "AD"
        cn = labels == "CN"
        mci = labels == "MCI"
        tally["AD"][0] += (ad * gt).sum()
        tally["AD"][1] += ad.sum()
        tally["CN"][0] += (cn * gt).sum()
        tally["CN"][1] += cn.sum()
        tally["MCI"][0] += (mci * gt).sum()
        tally["MCI"][1] += mci.sum()
        tally["Total"][0] += gt.sum()
        tally["Total"][1] += len(labels)

        return tally
