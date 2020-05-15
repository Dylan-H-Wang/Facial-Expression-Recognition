import os
import sys
import copy

import numpy as np

import h5py

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import dataset
import deep_model
import handcaft_model

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
		"./models/vgg_face_no_dsd.pth"
	)

def test_vgg_face():
	check = torch.load("./models/vgg_face_no_dsd.pth")
	model = deep_model.VggFace()

	model.fc_6 = nn.Linear(model.fc_6.in_features, model.fc_6.out_features)
	model.fc_7 = nn.Linear(model.fc_7.in_features, model.fc_7.out_features)
	model.fc_8 = nn.Linear(model.fc_8.in_features, 7)
	model.block_1[0] = nn.Conv2d(1, 64, 3, stride=1, padding=1)

	model = model.to(DEVICE)
	model.load_state_dict(check['model_state_dict'])

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

	print("Finish vgg_face model test with accuracy {:.4f}".format(accuracy))

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

def test_vgg_f():
	check = torch.load("./models/vgg_f_no_dsd.pth")
	model = deep_model.Vggf(pretrain=False)

	model.fc8 = nn.Linear(model.fc8.in_features, 7)
	model.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4)

	model = model.to(DEVICE)
	model.load_state_dict(check['model_state_dict'])

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

	print("Finish vgg_f model test with accuracy {:.4f}".format(accuracy))

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
		"./models/vgg_13_2.pth",
	)

def test_vgg_13():
	check = torch.load("./models/vgg_13_2.pth")
	model = deep_model.Vgg13()

	model = model.to(DEVICE)
	model.load_state_dict(check['model_state_dict'])

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

	print("Finish vgg_13 model test with accuracy {:.4f}".format(accuracy))

def train_deep_model():
# 	print(DEVICE)
	# dataset_train = dataset.FER2013Dataset("train", (224, 224))
	# dataset_val = dataset.FER2013Dataset("val", (224, 224))

	# data_loader = {
	# 	"train": DataLoader(dataset_train, batch_size=128, num_workers=4),
	# 	"val": DataLoader(dataset_val, batch_size=128, num_workers=4),
	# }

	# train_vgg_face(data_loader["train"], data_loader["val"])
	# del data_loader

	# data_loader = {
	# 	"train": DataLoader(dataset_train, batch_size=256, num_workers=4),
	# 	"val": DataLoader(dataset_val, batch_size=256, num_workers=4),
	# }

	# train_vgg_f(data_loader["train"], data_loader["val"])
	# del dataset_train, dataset_val, data_loader

	dataset_train = dataset.FER2013Dataset("train", (64, 64))
	dataset_val = dataset.FER2013Dataset("val", (64, 64))

	data_loader = {
		"train": DataLoader(dataset_train, batch_size=512, num_workers=4),
		"val": DataLoader(dataset_val, batch_size=512, num_workers=4),
	}

	train_vgg_13(data_loader["train"], data_loader["val"])

def train_handcraft_model():
	flag = True

	dataset_train = dataset.FER2013Dataset("train", (48, 48))
	dataset_val = dataset.FER2013Dataset("val", (48, 48))

	data_train, label_train = dataset_train.get_data()
	data_val, label_val = dataset_val.get_data()

	# descriptors_train = handcaft_model.calc_densen_SIFT(data_train)
	# descriptors_val = handcaft_model.calc_densen_SIFT(data_val)

	if flag:
		print("Loading kmeans models ...")
		checkpoint = torch.load("./models/kmeans.pth")
		kmeans_train = checkpoint["kmeans_train"]
		kmeans_val = checkpoint["kmeans_val"]
	else:
		kmeans_train = handcaft_model.kmeans([17000, 14000, 8000], descriptors_train)
		kmeans_val = handcaft_model.kmeans([17000, 14000, 8000], descriptors_val)

		torch.save(
			{
				"kmeans_train": kmeans_train,
				"kmeans_val": kmeans_val,
			},
			"./models/kmeans.pth",
		) 

	if flag:
		handcaft_model.get_histogram_spatial_pyramid(data_train, kmeans_train, "./data/histogram_train.hdf5")
		print("histogram train file saved")

		handcaft_model.get_histogram_spatial_pyramid(data_val, kmeans_val, "./data/histogram_val.hdf5")
		print("histogram val file saved")

	print("Loading histogram files ...")
	file_train = h5py.File("./data/histogram_train.hdf5", 'r')
	file_val = h5py.File("./data/histogram_val.hdf5", 'r')

	histogram_train = file_train["histogram"]
	histogram_val = file_val["histogram"]

	# histogram_train_np = np.array(histogram_train, dtype=object) 
	# histogram_val_np = np.array(histogram_val, dtype=object) 

	# Train global SVM
	print("Training global SVM ...")

	if not flag:
		print("Loading global SVM model ...")
		global_svm = torch.load("./models/global_svm.pth")["global_svm"]
	else:
		parameters = {'alpha': 10.0**-np.arange(1,7,0.5)}
		# C = 1
		# alpha = 1.0 / (C * histogram_train.shape[0])
		# global_svm = SGDClassifier(class_weight="balanced", n_jobs=-1)
		# clf = GridSearchCV(global_svm, parameters, n_jobs=-1)
		# grid_result = clf.fit(histogram_train, label_train)

		# best_params = grid_result.best_params_

		global_svm = SGDClassifier(class_weight="balanced", n_jobs=-1)
		global_svm.fit(histogram_train, label_train)
		# torch.save(
		# 	{
		# 		"global_svm": global_svm,
		# 	},
		# 	"./models/global_svm.pth",
		# )
		# print("Global SVM model saved")

	# preds = global_svm.predict(histogram_val)
	# accuracy = accuracy_score(label_val, preds)
	accuracy = global_svm.score(histogram_val, label_val)
	print("Global SVM validation accuracy: {:.4f}".format(accuracy))
	
	# for k, v in best_params.items():
	# 	print("Parameter {}: {}".format(k, v))

	file_train.close()
	file_val.close()

def train_combine():
	vgg_face = None
	vgg_f = None
	vgg_13 = None
	global_svm = None

	samples_train = None
	samples_test = None

	distance = cosine_similarity(samples_test, samples_train)
	neighbours = np.argmax(distance, axis=1)

	rows = np.arange(distance.shape[0])[:, np.newaxis]
	neighbours = distance[rows, neighbours]

	local_svm = BaggingClassifier(
		LinearSVC(C=100, class_weight="balanced"),
		max_samples=0.1,
		n_estimators=10,
		bootstrap=False,
		n_jobs=-1,
		verbose=1,
	)
	local_svm.fit()

	torch.save(
		{
			"local_svm": local_svm,
		},
		"./models/local_svm.pth",
	)

def train_handcraft_model_2():
	dataset_train = dataset.FER2013Dataset("train", (48, 48))
	dataset_val = dataset.FER2013Dataset("val", (48, 48))

	data_train, label_train = dataset_train.get_data()
	data_val, label_val = dataset_val.get_data()

	descriptors_train = handcaft_model.calc_densen_SIFT(data_train)
	descriptors_val = handcaft_model.calc_densen_SIFT(data_val)

	print("Loading kmeans models ...")
	checkpoint = torch.load("./models/kmeans.pth")
	kmeans_train = checkpoint["kmeans_train"]
	kmeans_val = checkpoint["kmeans_val"]

	# with h5py.File('hist_tmp.hdf5', 'w') as f:
	# 	hist_train = f.create_dataset("hist_train", (len(data_train), 17000))
	# 	hist_val = f.create_dataset("hist_val", (len(data_val), 17000))

	# 	handcaft_model.get_histogram(data_train, kmeans_train[0], hist_train)
	# 	handcaft_model.get_histogram(data_val, kmeans_val[0], hist_val)

	with h5py.File('hist_tmp.hdf5', 'r') as f:
		hist_train = f["hist_train"]
		hist_val = f["hist_val"]

		# Train global SVM
		print("Training global SVM ...")

		C = 1
		# alpha = 1.0 / (C * hist_train.shape[0])
		# global_svm = SGDClassifier(alpha=alpha, class_weight="balanced", verbose=1, n_jobs=-1)
		# global_svm.fit(hist_train, label_train)

		global_svm = LinearSVC(class_weight="balanced", verbose=1, max_iter=20000)
		global_svm.fit(hist_train, label_train)

		torch.save(
			{
				"global_svm": global_svm,
			},
			"./models/global_svm_linear.pth",
		)

		preds = global_svm.predict(hist_val)
		accuracy = accuracy_score(label_val, preds)
		# accuracy = global_svm.score(hist_val, label_val)
		print("Global SVM validation accuracy: {:.4f}".format(accuracy))


# train_deep_model()
train_handcraft_model()

# with h5py.File("./data/histogram_train.hdf5", 'r') as f:
# 	histogram_train = f["histogram"]
# 	a = histogram_train[0]
# 	b = np.array(histogram_train[0])
# 	print(histogram_train, axis=1)

# test_vgg_face()
# test_vgg_f()
test_vgg_13()
