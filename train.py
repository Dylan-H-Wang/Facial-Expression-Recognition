import os
import sys
import copy

import numpy as np

import h5py

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from tqdm import trange

import dataset
import deep_model
import handcraft_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Change current working dir to find the file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def conduct_train(
    data_train, data_val, n_epoch, model, optimiser, criterion, scheduler
):
    best_acc = 0
    best_model_wts = None
    all_loss = {"train": [], "val": [], "acc": []}

    for epoch in range(n_epoch):
        print("\n" + "-" * 10)
        print("Epoch {}/{}:".format(epoch + 1, n_epoch))

        print("\nTraining:")
        # Set model to training mode
        model.train()

        train_loss = []

        for data, label in tqdm(data_train):
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            # Zero the parameter gradients
            optimiser.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                preds = model(data)
                loss = criterion(preds, label)

                # Backward + optimise
                loss.backward()
                optimiser.step()

            train_loss.append(loss.item())

        train_loss = np.mean(np.array(train_loss).astype(np.float))
        all_loss["train"].append(train_loss)

        print("Finished with training loss: {:.4f}".format(train_loss))

        # Validation
        print("\nValidating:")
        # Set model to val mode
        model.eval()

        val_loss = []
        val_acc = []

        with torch.set_grad_enabled(False):
            for data, label in tqdm(data_val):
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                preds = model(data)
                loss = criterion(preds, label)

                val_loss.append(loss.item())

                preds = torch.argmax(preds, 1).cpu().numpy()
                label = label.cpu().numpy()
                val_acc.append(accuracy_score(label, preds))

        val_loss = np.mean(np.array(val_loss).astype(np.float))
        all_loss["val"].append(val_loss)

        val_acc = np.mean(np.array(val_acc).astype(np.float))
        all_loss["acc"].append(val_acc)

        print(
            "Finished with val loss: {:.4f}, val accuracy: {:.4f}/{:.4f}".format(
                val_loss, val_acc, best_acc
            )
        )

        # Finetune learning rate
        scheduler.step(val_loss)

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Best model saved!")

            # torch.save(
            # 	{
            # 		"model_state_dict": model.state_dict(),
            # 	},
            # 	"./models/vgg_f_2_temp.pth",
            # )

    model.load_state_dict(best_model_wts)
    return all_loss["train"], all_loss["val"], all_loss["acc"]


def train_vgg_face(data_train, data_val):
    print("Begin to train vgg_face model ...")
    model = deep_model.VggFace()

    # Freeze conv layers and finetune fc layers
    set_parameter_requires_grad(model)
    model.fc_6 = nn.Linear(model.fc_6.in_features, model.fc_6.out_features)
    nn.init.normal_(model.fc_6.weight, 0, 0.01)
    nn.init.constant_(model.fc_6.bias, 0)

    model.fc_7 = nn.Linear(model.fc_7.in_features, model.fc_7.out_features)
    nn.init.normal_(model.fc_7.weight, 0, 0.01)
    nn.init.constant_(model.fc_7.bias, 0)

    model.fc_8 = nn.Linear(model.fc_8.in_features, 7)
    nn.init.normal_(model.fc_8.weight, 0, 0.01)
    nn.init.constant_(model.fc_8.bias, 0)

    # Chaneg the frist conv layer input channels
    weights = torch.mean(model.block_1[0].weight.data, 1, True)
    bias = model.block_1[0].bias.data

    model.block_1[0] = nn.Conv2d(1, 64, 3, stride=1, padding=1)
    model.block_1[0].weight.data = weights
    model.block_1[0].bias.data = bias

    model.block_1[0].weight.requires_grad_(False)
    model.block_1[0].bias.requires_grad_(False)

    model = model.to(DEVICE)

    all_loss = {"train": [], "val": [], "acc": []}
    lr = 1e-2
    criterion = nn.NLLLoss()
    optimiser = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.1, verbose=True
    )

    print("\n" + "-" * 6 + " First dense stage " + "-" * 6)
    model.make_sparse(False)
    loss_train, loss_val, acc_val = conduct_train(
        data_train, data_val, 200, model, optimiser, criterion, scheduler
    )
    all_loss["train"].extend(loss_train)
    all_loss["val"].extend(loss_val)
    all_loss["acc"].extend(acc_val)

    # torch.save(
    # 	{
    # 		"model_state_dict": model.state_dict(),
    # 		"train_loss": all_loss["train"],
    # 		"val_loss": all_loss["val"],
    # 		"val_acc": all_loss["acc"],
    # 	},
    # 	"./models/vgg_face_d1.pth"
    # )

    # print("\n" + "-" * 6 + " First sparse stage " + "-" * 6)
    # model.make_sparse(True, 0.6)
    # loss_train, loss_val, acc_val = conduct_train(
    # 	data_train, data_val, 50, model, optimiser, criterion, scheduler
    # )
    # all_loss["train"].extend(loss_train)
    # all_loss["val"].extend(loss_val)
    # all_loss["acc"].extend(acc_val)

    # torch.save(
    # 	{
    # 		"model_state_dict": model.state_dict(),
    # 		"train_loss": all_loss["train"],
    # 		"val_loss": all_loss["val"],
    # 		"val_acc": all_loss["acc"],
    # 	},
    # 	"./models/vgg_face_s1.pth"
    # )

    # print("\n" + "-" * 6 + " Second dense stage " + "-" * 6)
    # model.make_sparse(False)
    # loss_train, loss_val, acc_val = conduct_train(
    # 	data_train, data_val, 50, model, optimiser, criterion, scheduler
    # )
    # all_loss["train"].extend(loss_train)
    # all_loss["val"].extend(loss_val)
    # all_loss["acc"].extend(acc_val)

    # torch.save(
    # 	{
    # 		"model_state_dict": model.state_dict(),
    # 		"train_loss": all_loss["train"],
    # 		"val_loss": all_loss["val"],
    # 		"val_acc": all_loss["acc"],
    # 	},
    # 	"./models/vgg_face_d2.pth"
    # )

    # print("\n" + "-" * 6 + " Second sparse stage " + "-" * 6)
    # model.make_sparse(True, 0.6)
    # loss_train, loss_val, acc_val = conduct_train(
    # 	data_train, data_val, 50, model, optimiser, criterion, scheduler
    # )
    # all_loss["train"].extend(loss_train)
    # all_loss["val"].extend(loss_val)
    # all_loss["acc"].extend(acc_val)

    # torch.save(
    # 	{
    # 		"model_state_dict": model.state_dict(),
    # 		"train_loss": all_loss["train"],
    # 		"val_loss": all_loss["val"],
    # 		"val_acc": all_loss["acc"],
    # 	},
    # 	"./models/vgg_face_s2.pth"
    # )

    # print("\n" + "-" * 6 + " Third dense stage " + "-" * 6)
    # model.make_sparse(False)
    # loss_train, loss_val, acc_val = conduct_train(
    # 	data_train, data_val, 50, model, optimiser, criterion, scheduler
    # )
    # all_loss["train"].extend(loss_train)
    # all_loss["val"].extend(loss_val)
    # all_loss["acc"].extend(acc_val)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_loss": all_loss["train"],
            "val_loss": all_loss["val"],
            "val_acc": all_loss["acc"],
        },
        "./models/vgg_face_no_dsd.pth",
    )


def test_vgg_face(imgs=None):
    check = torch.load("./models/vgg_face_no_dsd.pth")
    model = deep_model.VggFace()

    model.fc_8 = nn.Linear(model.fc_8.in_features, 7)
    model.block_1[0] = nn.Conv2d(1, 64, 3, stride=1, padding=1)

    model = model.to(DEVICE)
    model.load_state_dict(check["model_state_dict"])

    if imgs is None:
        dataset_val = dataset.FER2013Dataset("test", (224, 224))
        data_loader = DataLoader(dataset_val, batch_size=128, num_workers=4)

        model.eval()
        val_acc = []
        with torch.set_grad_enabled(False):
            for data, label in tqdm(data_loader):
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                preds = model(data)

                preds = torch.argmax(preds, 1).cpu().numpy()
                label = label.cpu().numpy()
                val_acc.append(accuracy_score(label, preds))

        accuracy = np.mean(np.array(val_acc).astype(np.float))

    else:
        model.eval()
        val_acc = []
        with torch.set_grad_enabled(False):
            data = imgs[0].to(DEVICE)
            label = imgs[1].to(DEVICE)

            preds = model(data)

            preds = torch.argmax(preds, 1).cpu().numpy()
            label = label.cpu().numpy()
            val_acc.append(accuracy_score(label, preds))

        accuracy = np.mean(np.array(val_acc).astype(np.float))
        return preds

    print("Finish vgg_face model test with accuracy {:.4f}".format(accuracy))


def extract_vgg_face():
    print("Extracting vgg_face model training sample features ...")

    check = torch.load("./models/vgg_face_no_dsd.pth")
    model = deep_model.VggFace()

    model.fc_8 = nn.Linear(model.fc_8.in_features, 7)
    model.block_1[0] = nn.Conv2d(1, 64, 3, stride=1, padding=1)

    model = model.to(DEVICE)
    model.load_state_dict(check["model_state_dict"])

    dataset_ = dataset.FER2013Dataset("train", (224, 224))
    dataloader = DataLoader(dataset_, batch_size=1, num_workers=4)

    model.eval()

    with h5py.File("./data/vgg_face_features_train.hdf5", "w") as f:
        features = f.create_dataset("features", (len(dataset_), 4096))

        with torch.set_grad_enabled(False):
            counter = 0
            for data, label in tqdm(dataloader):
                data = data.to(DEVICE)
                preds = model.extract_features(data)
                preds = preds.cpu().numpy()
                preds = preprocessing.normalize(preds, norm="l2")

                features[counter] = preds
                counter += 1

    del dataset_, dataloader

    print("Extracting vgg_face model testing sample features ...")

    dataset_ = dataset.FER2013Dataset("test", (224, 224))
    dataloader = DataLoader(dataset_, batch_size=1, num_workers=4)

    with h5py.File("./data/vgg_face_features_test.hdf5", "w") as f:
        features = f.create_dataset("features", (len(dataset_), 4096))

        with torch.set_grad_enabled(False):
            counter = 0
            for data, label in tqdm(dataloader):
                data = data.to(DEVICE)
                preds = model.extract_features(data)
                preds = preds.cpu().numpy()
                preds = preprocessing.normalize(preds, norm="l2")

                features[counter] = preds
                counter += 1


def train_vgg_f(data_train, data_val):
    print("Begin to train vgg_f model ...")
    model = deep_model.Vggf(pretrain=False)

    model.fc8 = nn.Linear(model.fc8.in_features, 7)
    nn.init.normal_(model.fc8.weight, 0, 0.01)
    nn.init.constant_(model.fc8.bias, 0)

    # Chaneg the frist conv layer input channels
    weights = torch.mean(model.conv1.weight.data, 1, True)
    bias = model.conv1.bias.data

    model.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4)
    model.conv1.weight.data = weights
    model.conv1.bias.data = bias

    model = model.to(DEVICE)

    all_loss = {"train": [], "val": [], "acc": []}
    lr = 1e-2
    criterion = nn.NLLLoss()
    optimiser = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.1, verbose=True
    )

    print("\n" + "-" * 6 + " First dense stage " + "-" * 6)
    model.make_sparse(False)
    loss_train, loss_val, acc_val = conduct_train(
        data_train, data_val, 300, model, optimiser, criterion, scheduler
    )
    all_loss["train"].extend(loss_train)
    all_loss["val"].extend(loss_val)
    all_loss["acc"].extend(acc_val)

    # torch.save(
    # 	{
    # 		"model_state_dict": model.state_dict(),
    # 		"train_loss": all_loss["train"],
    # 		"val_loss": all_loss["val"],
    # 		"val_acc": all_loss["acc"],
    # 	},
    # 	"./models/vgg_f_2_d1.pth",
    # )

    # print("\n" + "-" * 6 + " First sparse stage " + "-" * 6)
    # model.make_sparse(True, 0.4)
    # loss_train, loss_val, acc_val = conduct_train(
    # 	data_train, data_val, 50, model, optimiser, criterion, scheduler
    # )
    # all_loss["train"].extend(loss_train)
    # all_loss["val"].extend(loss_val)
    # all_loss["acc"].extend(acc_val)

    # torch.save(
    # 	{
    # 		"model_state_dict": model.state_dict(),
    # 		"train_loss": all_loss["train"],
    # 		"val_loss": all_loss["val"],
    # 		"val_acc": all_loss["acc"],
    # 	},
    # 	"./models/vgg_f_2_s1.pth",
    # )

    # print("\n" + "-" * 6 + " Second dense stage " + "-" * 6)
    # model.make_sparse(False)
    # loss_train, loss_val, acc_val = conduct_train(
    # 	data_train, data_val, 50, model, optimiser, criterion, scheduler
    # )
    # all_loss["train"].extend(loss_train)
    # all_loss["val"].extend(loss_val)
    # all_loss["acc"].extend(acc_val)

    # torch.save(
    # 	{
    # 		"model_state_dict": model.state_dict(),
    # 		"train_loss": all_loss["train"],
    # 		"val_loss": all_loss["val"],
    # 		"val_acc": all_loss["acc"],
    # 	},
    # 	"./models/vgg_f_2_d2.pth",
    # )

    # print("\n" + "-" * 6 + " Second sparse stage " + "-" * 6)
    # model.make_sparse(True, 0.4)
    # loss_train, loss_val, acc_val = conduct_train(
    # 	data_train, data_val, 50, model, optimiser, criterion, scheduler
    # )
    # all_loss["train"].extend(loss_train)
    # all_loss["val"].extend(loss_val)
    # all_loss["acc"].extend(acc_val)

    # torch.save(
    # 	{
    # 		"model_state_dict": model.state_dict(),
    # 		"train_loss": all_loss["train"],
    # 		"val_loss": all_loss["val"],
    # 		"val_acc": all_loss["acc"],
    # 	},
    # 	"./models/vgg_f_2_s2.pth",
    # )

    # print("\n" + "-" * 6 + " Third dense stage " + "-" * 6)
    # model.make_sparse(False)
    # loss_train, loss_val, acc_val = conduct_train(
    # 	data_train, data_val, 50, model, optimiser, criterion, scheduler
    # )
    # all_loss["train"].extend(loss_train)
    # all_loss["val"].extend(loss_val)
    # all_loss["acc"].extend(acc_val)

    # torch.save(
    # 	{
    # 		"model_state_dict": model.state_dict(),
    # 		"train_loss": all_loss["train"],
    # 		"val_loss": all_loss["val"],
    # 		"val_acc": all_loss["acc"],
    # 	},
    # 	"./models/vgg_f_2_d3.pth",
    # )

    # print("\n" + "-" * 6 + " Third sparse stage " + "-" * 6)
    # model.make_sparse(True, 0.4)
    # loss_train, loss_val, acc_val = conduct_train(
    # 	data_train, data_val, 50, model, optimiser, criterion, scheduler
    # )
    # all_loss["train"].extend(loss_train)
    # all_loss["val"].extend(loss_val)
    # all_loss["acc"].extend(acc_val)

    # torch.save(
    # 	{
    # 		"model_state_dict": model.state_dict(),
    # 		"train_loss": all_loss["train"],
    # 		"val_loss": all_loss["val"],
    # 		"val_acc": all_loss["acc"],
    # 	},
    # 	"./models/vgg_f_2_s3.pth",
    # )

    # print("\n" + "-" * 6 + " Forth dense stage " + "-" * 6)
    # model.make_sparse(False)
    # loss_train, loss_val, acc_val = conduct_train(
    # 	data_train, data_val, 50, model, optimiser, criterion, scheduler
    # )
    # all_loss["train"].extend(loss_train)
    # all_loss["val"].extend(loss_val)
    # all_loss["acc"].extend(acc_val)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_loss": all_loss["train"],
            "val_loss": all_loss["val"],
            "val_acc": all_loss["acc"],
        },
        "./models/vgg_f_no_dsd.pth",
    )


def test_vgg_f(imgs=None):
    check = torch.load("./models/vgg_f_no_dsd.pth")
    model = deep_model.Vggf(pretrain=False)

    model.fc8 = nn.Linear(model.fc8.in_features, 7)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4)

    model = model.to(DEVICE)
    model.load_state_dict(check["model_state_dict"])

    if imgs is None:
        dataset_val = dataset.FER2013Dataset("test", (224, 224))
        data_loader = DataLoader(dataset_val, batch_size=256, num_workers=4)

        model.eval()
        val_acc = []
        with torch.set_grad_enabled(False):
            for data, label in tqdm(data_loader):
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                preds = model(data)

                preds = torch.argmax(preds, 1).cpu().numpy()
                label = label.cpu().numpy()
                val_acc.append(accuracy_score(label, preds))

        accuracy = np.mean(np.array(val_acc).astype(np.float))

    else:
        model.eval()
        val_acc = []
        with torch.set_grad_enabled(False):
            data = imgs[0].to(DEVICE)
            label = imgs[1].to(DEVICE)

            preds = model(data)

            preds = torch.argmax(preds, 1).cpu().numpy()
            label = label.cpu().numpy()
            val_acc.append(accuracy_score(label, preds))

        accuracy = np.mean(np.array(val_acc).astype(np.float))
        return preds

    print("Finish vgg_f model test with accuracy {:.4f}".format(accuracy))


def extract_vgg_f():
    print("Extracting vgg_f model training sample features ...")

    check = torch.load("./models/vgg_f_no_dsd.pth")
    model = deep_model.Vggf()

    model.fc8 = nn.Linear(model.fc8.in_features, 7)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4)

    model = model.to(DEVICE)
    model.load_state_dict(check["model_state_dict"])

    dataset_ = dataset.FER2013Dataset("train", (224, 224))
    dataloader = DataLoader(dataset_, batch_size=1, num_workers=4)

    model.eval()

    with h5py.File("./data/vgg_f_features_train.hdf5", "w") as f:
        features = f.create_dataset("features", (len(dataset_), 4096))

        with torch.set_grad_enabled(False):
            counter = 0
            for data, label in tqdm(dataloader):
                data = data.to(DEVICE)
                preds = model.extract_features(data)
                preds = preds.cpu().numpy()
                preds = preprocessing.normalize(preds, norm="l2")

                features[counter] = preds
                counter += 1

    del dataset_, dataloader

    print("Extracting vgg_f model testing sample features ...")

    dataset_ = dataset.FER2013Dataset("test", (224, 224))
    dataloader = DataLoader(dataset_, batch_size=1, num_workers=4)

    with h5py.File("./data/vgg_f_features_test.hdf5", "w") as f:
        features = f.create_dataset("features", (len(dataset_), 4096))

        with torch.set_grad_enabled(False):
            counter = 0
            for data, label in tqdm(dataloader):
                data = data.to(DEVICE)
                preds = model.extract_features(data)
                preds = preds.cpu().numpy()
                preds = preprocessing.normalize(preds, norm="l2")

                features[counter] = preds
                counter += 1


def train_vgg_13(data_train, data_val):
    print("Begin to train vgg_13 model ...")
    model = deep_model.Vgg13()

    model = model.to(DEVICE)

    all_loss = {"train": [], "val": [], "acc": []}
    lr = 1e-2
    criterion = nn.NLLLoss()
    optimiser = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.1, verbose=True
    )

    loss_train, loss_val, acc_val = conduct_train(
        data_train, data_val, 300, model, optimiser, criterion, scheduler
    )
    all_loss["train"].extend(loss_train)
    all_loss["val"].extend(loss_val)
    all_loss["acc"].extend(acc_val)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_loss": all_loss["train"],
            "val_loss": all_loss["val"],
            "val_acc": all_loss["acc"],
        },
        "./models/vgg_13_no_dsd.pth",
    )


def test_vgg_13(imgs=None):
    check = torch.load("./models/vgg_13_no_dsd.pth")
    model = deep_model.Vgg13()

    model = model.to(DEVICE)
    model.load_state_dict(check["model_state_dict"])

    if imgs is None:
        dataset_val = dataset.FER2013Dataset("test", (64, 64))
        data_loader = DataLoader(dataset_val, batch_size=512, num_workers=4)

        model.eval()
        val_acc = []
        with torch.set_grad_enabled(False):
            for data, label in tqdm(data_loader):
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                preds = model(data)

                preds = torch.argmax(preds, 1).cpu().numpy()
                label = label.cpu().numpy()
                val_acc.append(accuracy_score(label, preds))

        accuracy = np.mean(np.array(val_acc).astype(np.float))

    else:
        model.eval()
        val_acc = []
        with torch.set_grad_enabled(False):
            data = imgs[0].to(DEVICE)
            label = imgs[1].to(DEVICE)

            preds = model(data)

            preds = torch.argmax(preds, 1).cpu().numpy()
            label = label.cpu().numpy()
            val_acc.append(accuracy_score(label, preds))

        accuracy = np.mean(np.array(val_acc).astype(np.float))
        return preds

    print("Finish vgg_13 model test with accuracy {:.4f}".format(accuracy))


def extract_vgg_13():
    print("Extracting vgg_13 model training sample features ...")

    check = torch.load("./models/vgg_13_no_dsd.pth")
    model = deep_model.Vgg13()

    model = model.to(DEVICE)
    model.load_state_dict(check["model_state_dict"])

    dataset_ = dataset.FER2013Dataset("train", (64, 64))
    dataloader = DataLoader(dataset_, batch_size=1, num_workers=4)

    model.eval()

    with h5py.File("./data/vgg_13_features_train.hdf5", "w") as f:
        features = f.create_dataset("features", (len(dataset_), 1024))

        with torch.set_grad_enabled(False):
            counter = 0
            for data, label in tqdm(dataloader):
                data = data.to(DEVICE)
                preds = model.extract_features(data)
                preds = preds.cpu().numpy()
                preds = preprocessing.normalize(preds, norm="l2")

                features[counter] = preds
                counter += 1

    del dataset_, dataloader

    print("Extracting vgg_13 model testing sample features ...")

    dataset_ = dataset.FER2013Dataset("test", (64, 64))
    dataloader = DataLoader(dataset_, batch_size=1, num_workers=4)

    with h5py.File("./data/vgg_13_features_test.hdf5", "w") as f:
        features = f.create_dataset("features", (len(dataset_), 1024))

        with torch.set_grad_enabled(False):
            counter = 0
            for data, label in tqdm(dataloader):
                data = data.to(DEVICE)
                preds = model.extract_features(data)
                preds = preds.cpu().numpy()
                preds = preprocessing.normalize(preds, norm="l2")

                features[counter] = preds
                counter += 1


def train_deep_model():
    dataset_train = dataset.FER2013Dataset("train", (224, 224))
    dataset_val = dataset.FER2013Dataset("val", (224, 224))

    data_loader = {
        "train": DataLoader(dataset_train, batch_size=128, num_workers=4),
        "val": DataLoader(dataset_val, batch_size=128, num_workers=4),
    }

    train_vgg_face(data_loader["train"], data_loader["val"])
    del data_loader

    data_loader = {
        "train": DataLoader(dataset_train, batch_size=256, num_workers=4),
        "val": DataLoader(dataset_val, batch_size=256, num_workers=4),
    }

    train_vgg_f(data_loader["train"], data_loader["val"])
    del dataset_train, dataset_val, data_loader

    dataset_train = dataset.FER2013Dataset("train", (64, 64))
    dataset_val = dataset.FER2013Dataset("val", (64, 64))

    data_loader = {
        "train": DataLoader(dataset_train, batch_size=512, num_workers=4),
        "val": DataLoader(dataset_val, batch_size=512, num_workers=4),
    }

    train_vgg_13(data_loader["train"], data_loader["val"])


def test_deep_model():
    test_vgg_face()
    test_vgg_f()
    test_vgg_13()


def extract_deep_model():
    extract_vgg_face()
    extract_vgg_f()
    extract_vgg_13()

    print("Finish deep model feature extraction ...")


def extract_handcraft_model(resume=True):
    dataset_train = dataset.FER2013Dataset("train", (48, 48))
    dataset_val = dataset.FER2013Dataset("test", (48, 48))

    data_train, label_train = dataset_train.get_data()
    data_val, label_val = dataset_val.get_data()

    descriptors_train = handcraft_model.calc_densen_SIFT(data_train)
    descriptors_val = handcraft_model.calc_densen_SIFT(data_val)

    if resume:
        print("Loading kmeans models ...")
        checkpoint = torch.load("./models/kmeans.pth")
        kmeans_train = checkpoint["kmeans_train"]
        kmeans_val = checkpoint["kmeans_val"]
    else:
        kmeans_train = handcraft_model.kmeans([17000, 14000, 8000], descriptors_train)
        kmeans_val = handcraft_model.kmeans([17000, 14000, 8000], descriptors_val)

        torch.save(
            {"kmeans_train": kmeans_train, "kmeans_val": kmeans_val,},
            "./models/kmeans.pth",
        )

    handcraft_model.get_histogram_spatial_pyramid(
        data_train, kmeans_train, "./data/histogram_train.hdf5"
    )
    print("histogram train file saved")

    handcraft_model.get_histogram_spatial_pyramid(
        data_val, kmeans_val, "./data/histogram_test.hdf5"
    )
    print("histogram val file saved")


def combine_features():
    print("Combining training samples ...")

    vgg_face_train = h5py.File("./data/vgg_face_features_train.hdf5", "r")
    vgg_f_train = h5py.File("./data/vgg_f_features_train.hdf5", "r")
    vgg_13_train = h5py.File("./data/vgg_13_features_train.hdf5", "r")
    histogram_train = h5py.File("./data/histogram_train.hdf5", "r")

    shape = [
        vgg_face_train["features"].shape[0],
        vgg_face_train["features"].shape[1]
        + vgg_f_train["features"].shape[1]
        + vgg_13_train["features"].shape[1]
        + histogram_train["histogram"].shape[1],
    ]

    with h5py.File("./data/combined_features_train.hdf5", "w") as f:
        features = f.create_dataset("features", (shape[0], shape[1]), compression="lzf")

        for idx in trange(shape[0]):
            feats = [
                vgg_face_train["features"][idx],
                vgg_f_train["features"][idx],
                vgg_13_train["features"][idx],
                histogram_train["histogram"][idx],
            ]
            feats = np.concatenate(feats)
            features[idx] = feats

    vgg_face_train.close()
    vgg_f_train.close()
    vgg_13_train.close()
    histogram_train.close()

    print("Combining testing samples ...")

    vgg_face_test = h5py.File("./data/vgg_face_features_test.hdf5", "r")
    vgg_f_test = h5py.File("./data/vgg_f_features_test.hdf5", "r")
    vgg_13_test = h5py.File("./data/vgg_13_features_test.hdf5", "r")
    histogram_test = h5py.File("./data/histogram_test.hdf5", "r")

    shape = [
        vgg_face_test["features"].shape[0],
        vgg_face_test["features"].shape[1]
        + vgg_f_test["features"].shape[1]
        + vgg_13_test["features"].shape[1]
        + histogram_test["histogram"].shape[1],
    ]

    with h5py.File("./data/combined_features_test.hdf5", "w") as f:
        features = f.create_dataset("features", (shape[0], shape[1]), compression="lzf")

        for idx in trange(shape[0]):
            feats = [
                vgg_face_test["features"][idx],
                vgg_f_test["features"][idx],
                vgg_13_test["features"][idx],
                histogram_test["histogram"][idx],
            ]
            feats = np.concatenate(feats)
            features[idx] = feats

    vgg_face_test.close()
    vgg_f_test.close()
    vgg_13_test.close()
    histogram_test.close()


def cal_similarity():
    print("Calculating cosine similarity ...")

    sample_train_f = h5py.File("./data/combined_features_train.hdf5", "r")
    sample_test_f = h5py.File("./data/combined_features_test.hdf5", "r")

    sample_train = sample_train_f["features"]
    sample_test = sample_test_f["features"]

    train_idx = list(range(0, sample_train.shape[0], 1500))
    test_idx = list(range(0, sample_test.shape[0], 500))

    train_idx.append(sample_train.shape[0])
    test_idx.append(sample_test.shape[0])

    score = np.zeros((sample_test.shape[0], sample_train.shape[0]))
    for i in trange(len(test_idx) - 1):
        test = sample_test[test_idx[i] : test_idx[i + 1]]
        res = []

        for j in trange(len(train_idx) - 1):
            train = sample_train[train_idx[j] : train_idx[j + 1]]
            res.append(cosine_similarity(test, train))

        score[test_idx[i] : test_idx[i + 1]] = np.hstack(res)

    sample_train_f.close()
    sample_test_f.close()

    best_idx = np.argpartition(score, -200, axis=1)[:, -200:]
    del score, test, train
    print(best_idx.shape)

    with h5py.File("./data/cosine_similarity.hdf5", "w") as f:
        dst = f.create_dataset("similarity", data=best_idx)


def train_combine():
    sample_train_f = h5py.File("./data/combined_features_train.hdf5", "r")
    sample_test_f = h5py.File("./data/combined_features_test.hdf5", "r")
    similarity_f = h5py.File("./data/cosine_similarity.hdf5", "r")

    sample_train = sample_train_f["features"]
    sample_test = sample_test_f["features"]
    similarity = similarity_f["similarity"]

    dataset_train = dataset.FER2013Dataset("train", (48, 48))
    dataset_test = dataset.FER2013Dataset("test", (48, 48))
    data_train, label_train = dataset_train.get_data()
    data_test, label_test = dataset_test.get_data()
    del dataset_train, dataset_test, data_train, data_test

    preds = []
    for i in trange(similarity.shape[0]):
        best_idx = similarity[i]

        near_train = np.zeros((200, sample_train.shape[1]))
        near_train_label = np.zeros((200,))
        for idx, value in enumerate(best_idx):
            near_train[idx] = sample_train[value]
            near_train_label[idx] = label_train[value]

        test = sample_test[i]

        local_svm = LinearSVC(C=100, verbose=1, max_iter=100, class_weight="balanced")
        local_svm.fit(near_train, near_train_label)
        preds.append(local_svm.predict(test.reshape(1, -1)).astype(np.int64)[0])

        accuracy = accuracy_score(label_test[: i + 1], preds)
        print("Local SVM validation accuracy: {:.4f}".format(accuracy))

    accuracy = accuracy_score(label_test, preds)
    print("Local SVM validation accuracy: {:.4f}".format(accuracy))

    sample_train_f.close()
    sample_test_f.close()
    similarity_f.close()


def train_combine_global():
    dataset_train = dataset.FER2013Dataset("train", (48, 48))
    dataset_test = dataset.FER2013Dataset("test", (48, 48))

    data_train, label_train = dataset_train.get_data()
    data_test, label_test = dataset_test.get_data()

    del data_train, data_test

    sample_train_f = h5py.File("./data/combined_features_train.hdf5", "r")
    sample_test_f = h5py.File("./data/combined_features_test.hdf5", "r")

    sample_train = sample_train_f["features"]
    sample_test = sample_test_f["features"]

    # Train global SVM
    print("Training global SVM ...")

    classes = list(range(7))
    max_iter = 30
    alphas = [3.5714285714285716e-07, 10 ** -4]
    acc1 = []

    global_svm = SGDClassifier(alpha=alphas[1], n_jobs=-1)
    for i in trange(max_iter):
        minibatchor = handcraft_model.iter_minibatches(sample_train, label_train, 1000)

        for x, y in tqdm(minibatchor):
            global_svm.partial_fit(x, y, classes)

        idx = [0, 1000, 2000, 3000, 3589]
        acc = []
        for j in range(len(idx) - 1):
            acc.append(
                global_svm.score(
                    sample_test[idx[j] : idx[j + 1]], label_test[idx[j] : idx[j + 1]]
                )
            )
        print("Global SVM validation accuracy: {:.4f}".format(np.mean(acc)))

    torch.save({"svm": global_svm,}, "./models/global_svm_2.pth")

    idx = [0, 1000, 2000, 3000, 3589]
    acc = []
    for j in range(len(idx) - 1):
        acc.append(
            global_svm.score(
                sample_test[idx[j] : idx[j + 1]], label_test[idx[j] : idx[j + 1]]
            )
        )
    print("Global SVM validation accuracy: {:.4f}".format(np.mean(acc)))

    sample_train_f.close()
    sample_test_f.close()

