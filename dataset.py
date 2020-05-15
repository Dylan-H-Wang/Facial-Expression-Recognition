import os
from random import random

import numpy as np

import pandas as pd

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class FER2013Dataset(Dataset):
	def __init__(self, data_type="train", scale_size=(48, 48)):
		self.data_type = data_type
		self.transform = transforms.Compose(
			[
				transforms.Resize(scale_size, Image.NEAREST),
				transforms.RandomHorizontalFlip(0.5),
			]
		)
		self._load_data()

	def __len__(self):
		return self.data_len

	def __getitem__(self, idx):
		data_ = self.data_dict["data"][idx]
		label_ = self.data_dict["label"][idx]

		img = Image.new("P", (48, 48))
		img.putdata(data_)
		img = self.transform(img)
		data_ = transforms.functional.to_tensor(img)

		return (data_, label_)

	def _load_data(self):
		print("Preparing {} dataset ...".format(self.data_type))

		# Change current working dir to find the file
		abspath = os.path.abspath(__file__)
		dname = os.path.dirname(abspath)
		os.chdir(dname)

		if self.data_type == "train":
			data = pd.read_csv("./data/train.csv")
		elif self.data_type == "val":
			data = pd.read_csv("./data/validation.csv")
		elif self.data_type == "test":
			data = pd.read_csv("./data/test.csv")

		# self.data_group = [[] for i in range(7)]

		# Iterate each row and group images with the same emotion
		# list(
		# 	map(
		# 		lambda x: self.data_group[x[0]].append(
		# 			np.array([int(i) for i in x[1].strip("][").split(", ")], dtype=np.uint8).reshape(48, 48)
		# 		),
		# 		data.itertuples(index=False),
		# 	)
		# )

		# Data for deep model training
		self.data_dict = {"data": [], "label": []}

		for row in data.itertuples(index=False):
			self.data_dict["data"].append(
				[int(i) for i in row[1].strip("][").split(", ")]
			)
			self.data_dict["label"].append(row[0])

		# Calculate the length
		self.data_len = len(self.data_dict["label"])

		print("{} dataset contains {} samples".format(self.data_type, self.data_len))

	def get_data(self):
		return np.array(self.data_dict["data"], dtype=np.uint8).reshape(-1, 48, 48), np.array(self.data_dict["label"])