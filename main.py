import os
import sys
import argparse

import train
import dataset

import torch
from torchvision import transforms

from PIL import Image

import matplotlib.pyplot as plt

# Change current working dir to find the file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description="Face expression prediction")

parser.add_argument(
	"-m", "--mode", default="demo", help="mode of the program (default: demo)"
)

args = parser.parse_args()

if args.mode == "demo":
	print("-" * 6 + " demo mode " + "-" * 6)
	print("Display testing accuracy of each model:")
	train.test_deep_model()

	print("\nSaving deep model training process ...")
	checkpoint = torch.load("./models/vgg_face_no_dsd.pth")
	train_loss = checkpoint["train_loss"]
	val_loss = checkpoint["val_loss"]
	val_acc = checkpoint["val_acc"]

	fig = plt.figure()
	line1, = plt.plot(train_loss, label='train_loss')
	line2, = plt.plot(val_loss, label='val_loss')
	line3, = plt.plot(val_acc, label='val_acc')
	plt.legend(handles=[line1, line2, line3], loc='best')
	plt.xlabel('epoch')
	plt.ylabel('training loss')
	plt.grid()
	fig.savefig("./vgg_face_process.png")

	checkpoint = torch.load("./models/vgg_f_no_dsd.pth")
	train_loss = checkpoint["train_loss"]
	val_loss = checkpoint["val_loss"]
	val_acc = checkpoint["val_acc"]

	fig = plt.figure()
	line1, = plt.plot(train_loss, label='train_loss')
	line2, = plt.plot(val_loss, label='val_loss')
	line3, = plt.plot(val_acc, label='val_acc')
	plt.legend(handles=[line1, line2, line3], loc='best')
	plt.xlabel('epoch')
	plt.ylabel('training loss')
	plt.grid()
	fig.savefig("./vgg_f_process.png")

	checkpoint = torch.load("./models/vgg_13_no_dsd.pth")
	train_loss = checkpoint["train_loss"]
	val_loss = checkpoint["val_loss"]
	val_acc = checkpoint["val_acc"]

	fig = plt.figure()
	line1, = plt.plot(train_loss, label='train_loss')
	line2, = plt.plot(val_loss, label='val_loss')
	line3, = plt.plot(val_acc, label='val_acc')
	plt.legend(handles=[line1, line2, line3], loc='best')
	plt.xlabel('epoch')
	plt.ylabel('training loss')
	plt.grid()
	fig.savefig("./vgg_13_process.png")

	print("\nSelecting 3 images from testing set ...")
	test_dataset_224 = dataset.FER2013Dataset(data_type="test", scale_size=(224, 224))
	test_dataset_64 = dataset.FER2013Dataset(data_type="test", scale_size=(64, 64))

	# Make testing samples
	idx = [0, 3, 4]
	imgs_tensor_224 = []
	labels_224 = []
	imgs_tensor_64 = []
	labels_64 = []
	for i in range(len(idx)):
		data, label = test_dataset_224[idx[i]]

		# Save test sample image
		img = transforms.ToPILImage()(data)
		img.save("test_sample_{}.png".format(i))

		imgs_tensor_224.append(torch.unsqueeze(data, dim=0))
		labels_224.append(torch.unsqueeze(label, dim=0))

	for i in range(len(idx)):
		data, label = test_dataset_64[idx[i]]

		imgs_tensor_64.append(torch.unsqueeze(data, dim=0))
		labels_64.append(torch.unsqueeze(label, dim=0))

	imgs_tensor_224 = torch.cat(imgs_tensor_224, dim=0)
	labels_224 = torch.cat(labels_224, dim=0)
	imgs_tensor_64 = torch.cat(imgs_tensor_64, dim=0)
	labels_64 = torch.cat(labels_64, dim=0)

	imgs_224 = [imgs_tensor_224, labels_224]
	img_64 = [imgs_tensor_64, labels_64]

	# Make prediction
	print("\nMaking prediction ...")
	vgg_face_pred = train.test_vgg_face(imgs_224)
	vgg_f_pred = train.test_vgg_f(imgs_224)
	vgg_13_pred = train.test_vgg_13(img_64)

	# Show results
	print(
		"\n Labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral"
	)
	for i in range(3):
		print("\nObject: test_sample_{}.jpg".format(i))
		print("Ground truth: {}".format(labels_224[i]))
		print(
			"vgg_face: {}; vgg_f: {}; vgg_13: {}".format(
				vgg_face_pred[i], vgg_f_pred[i], vgg_13_pred[i]
			)
		)
		print("-" * 6)

elif args.mode == "train":
	print("-" * 6 + " train mode " + "-" * 6)

	train.train_deep_model()
	train.extract_deep_model()
	train.extract_handcraft_model(False)
	train.combine_features()
	train.cal_similarity()
	train.train_combine()
	# train.train_combine_global()
