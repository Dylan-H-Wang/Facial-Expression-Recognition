import os
from collections import OrderedDict

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchfile

import cv2


# Change current working dir to find the file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

class VggFace(nn.Module):
	def __init__(self, pretrain=True, path="./models/vgg_face.t7", feature_extract=False):
		super(VggFace, self).__init__()

		self.sparse = False
		self.feature_extract = feature_extract
		self.block_size = [2, 2, 3, 3, 3]

		self.block_1 = self._make_block(2, [3, 64, 64])
		self.block_2 = self._make_block(2, [64, 128, 128])
		self.block_3 = self._make_block(3, [128, 256, 256, 256])
		self.block_4 = self._make_block(3, [256, 512, 512, 512])
		self.block_5 = self._make_block(3, [512, 512, 512, 512])

		self.fc_6 = nn.Linear(512 * 7 * 7, 4096)
		self.fc_7 = nn.Linear(4096, 4096)
		self.fc_8 = nn.Linear(4096, 2622)

		self.dropout_1 = nn.Dropout(0.7)
		self.dropout_2 = nn.Dropout(0.5)

		self.log_soft = nn.LogSoftmax(dim=1)

		if pretrain:
			self._load_weights(path)

	def _make_block(self, n_layer, features):
		assert n_layer == len(features) - 1
		layers = []

		for idx in range(n_layer):
			layers.append(
				nn.Conv2d(features[idx], features[idx + 1], 3, stride=1, padding=1)
			)
			layers.append(nn.ReLU())

		layers.append(nn.MaxPool2d(2, 2))

		return nn.Sequential(*layers)

	def make_sparse(self, sparse, sparsity=None):
		if not sparse:
			self.sparse = False

		else:
			assert sparsity is not None
			self.sparse = True
			
			# Make masks
			layers = [self.fc_6, self.fc_7, self.fc_8]
			self.masks = []

			for layer in layers:
				weights_flat = torch.flatten(layer.weight.data)
				
				k = int(weights_flat.shape[0] * (1-sparsity))

				# get the kth smallest element
				kth_value = torch.abs(weights_flat).kthvalue(k)[0]

				# Create mask
				mask = layer.weight.data.lt(kth_value)
				# weights_flat = mask(weights_flat) + mask(-weights_flat)
				self.masks.append(mask)

	def forward(self, x):
		# input image (224x224)
		with torch.no_grad():
			x = self.block_1(x)
			x = self.block_2(x)
			x = self.block_3(x)
			x = self.block_4(x)
			x = self.block_5(x)

		# Flatten
		x = torch.flatten(x, 1)

		if self.sparse:
			self.fc_6.weight.data.masked_scatter_(self.masks[0], torch.zeros_like(self.fc_6.weight.data))
		x = F.relu(self.fc_6(x))
		x = self.dropout_1(x)

		if self.sparse:
			self.fc_7.weight.data.masked_scatter_(self.masks[1], torch.zeros_like(self.fc_7.weight.data))
		x = F.relu(self.fc_7(x))
		x = self.dropout_2(x)

		if self.sparse:
			self.fc_8.weight.data.masked_scatter_(self.masks[2], torch.zeros_like(self.fc_8.weight.data))
		x = self.fc_8(x)

		if not self.feature_extract:
			x = self.log_soft(x)

		return x

	# Function to load Lua torch pretrained model weights
	def _load_weights(self, path):
		model = torchfile.load(path)
		block_idx = 1
		layer_idx = 0
		state_dict = OrderedDict({})

		for idx, layer in enumerate(model.modules):
			if layer.weight is not None:
				if block_idx <= 5:
					key_name = "block_{}.{}".format(block_idx, layer_idx * 2)
					self_block = getattr(self, "block_{}".format(block_idx))
					self_layer = self_block[layer_idx * 2]

					# Update index
					layer_idx += 1
					if layer_idx >= self.block_size[block_idx - 1]:
						layer_idx = 0
						block_idx += 1

					state_dict[key_name + ".weight"] = torch.tensor(
						layer.weight
					).view_as(self_layer.weight)
					state_dict[key_name + ".bias"] = torch.tensor(layer.bias).view_as(
						self_layer.bias
					)

				else:
					key_name = "fc_{}".format(block_idx)
					self_block = getattr(self, "fc_{}".format(block_idx))

					block_idx += 1

					state_dict[key_name + ".weight"] = torch.tensor(
						layer.weight
					).view_as(self_block.weight)
					state_dict[key_name + ".bias"] = torch.tensor(layer.bias).view_as(
						self_block.bias
					)

		self.load_state_dict(state_dict)


# Used in Vgg-f model
class LRN(nn.Module):
	""" Local Response Normalisatio(LRN) by https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py
	"""

	def __init__(self, local_size=1, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=False):
		super(LRN, self).__init__()

		self.ACROSS_CHANNELS = ACROSS_CHANNELS
		self.alpha = alpha
		self.beta = beta

		if self.ACROSS_CHANNELS:
			self.average = nn.AvgPool3d(
				kernel_size=(local_size, 1, 1),
				stride=1,
				padding=(int((local_size - 1.0) / 2), 0, 0),
			)
		else:
			self.average = nn.AvgPool2d(
				kernel_size=local_size, stride=1, padding=int((local_size - 1.0) / 2),
			)

	def forward(self, x):
		if self.ACROSS_CHANNELS:
			div = x.pow(2).unsqueeze(1)
			div = self.average(div).squeeze(1)
			div = div.mul(self.alpha).add(2.0).pow(self.beta)
		else:
			div = x.pow(2)
			div = self.average(div)
			div = div.mul(self.alpha).add(2.0).pow(self.beta)
		x = x.div(div)
		return x


class Vggf(nn.Module):
	def __init__(self, pretrain=True, path="./models/vgg_f.pth", feature_extract=False):
		super(Vggf, self).__init__()

		self.sparse = False
		self.feature_extract = feature_extract

		self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4)
		self.conv2 = nn.Conv2d(64, 256, kernel_size=5, padding=2)
		self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

		self.lrn1 = LRN()
		self.lrn2 = LRN()

		self.fc6 = nn.Linear(256 * 6 * 6, 4096)
		self.fc7 = nn.Linear(4096, 4096)
		self.fc8 = nn.Linear(4096, 1000)

		self.dropout1 = nn.Dropout(0.35)
		self.dropout2 = nn.Dropout(0.35)
		self.dropout3 = nn.Dropout(0.5)
		self.dropout4 = nn.Dropout(0.5)

		self.log_soft = nn.LogSoftmax(dim=1)
		self._initialize_weights()
		
		if pretrain:
			self._load_weights(path)

	def make_sparse(self, sparse, sparsity=None):
		if not sparse:
			self.sparse = False

		else:
			assert sparsity is not None
			self.sparse = True
			
			# Make masks
			layers = [self.conv3, self.conv4, self.conv5, self.fc6, self.fc7, self.fc8]
			self.masks = []

			for layer in layers:
				weights_flat = torch.flatten(layer.weight.data)
				
				k = int(weights_flat.shape[0] * (1-sparsity))

				# get the kth smallest element
				kth_value = torch.abs(weights_flat).kthvalue(k)[0]

				# Create mask
				mask = layer.weight.data.lt(kth_value)
				# weights_flat = mask(weights_flat) + mask(-weights_flat)
				self.masks.append(mask)

	def forward(self, x):
		# input image (224x224)
		x = F.relu(self.conv1(x))
		x = self.lrn1(x)
		x = F.max_pool2d(x, (2, 2))

		x = F.relu(self.conv2(x))
		x = self.lrn2(x)
		x = F.max_pool2d(x, (2, 2))

		if self.sparse:
			self.conv3.weight.data.masked_scatter_(self.masks[0], torch.zeros_like(self.conv3.weight.data))
		x = F.relu(self.conv3(x))

		if self.sparse:
			self.conv4.weight.data.masked_scatter_(self.masks[1], torch.zeros_like(self.conv4.weight.data))
		x = F.relu(self.conv4(x))
		x = self.dropout1(x)

		if self.sparse:
			self.conv5.weight.data.masked_scatter_(self.masks[2], torch.zeros_like(self.conv5.weight.data))
		x = F.relu(self.conv5(x))
		x = self.dropout2(x)
		x = F.max_pool2d(x, (2, 2))

		# Flatten
		x = torch.flatten(x, 1)

		if self.sparse:
			self.fc6.weight.data.masked_scatter_(self.masks[3], torch.zeros_like(self.fc6.weight.data))
		x = F.relu(self.fc6(x))
		x = self.dropout3(x)

		if self.sparse:
			self.fc7.weight.data.masked_scatter_(self.masks[4], torch.zeros_like(self.fc7.weight.data))
		x = F.relu(self.fc7(x))
		x = self.dropout4(x)

		if self.sparse:
			self.fc8.weight.data.masked_scatter_(self.masks[5], torch.zeros_like(self.fc8.weight.data))
		x = F.relu(self.fc8(x))

		if not self.feature_extract:
			x = self.log_soft(x)

		return x

	# Function to load Lua torch pretrained model weights
	def _load_weights(self, path):
		self.load_state_dict(torch.load(path))

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


class Vgg13(nn.Module):
	def __init__(self, feature_extract=False):
		super(Vgg13, self).__init__()

		self.sparse = False
		self.feature_extract = feature_extract

		self.block1 = self._make_block(2, [1, 64, 64])
		self.block2 = self._make_block(2, [64, 128, 128])
		self.block3 = self._make_block(3, [128, 256, 256, 256])
		self.block4 = self._make_block(3, [256, 256, 256, 256])

		self.classifier = nn.Sequential(
			nn.Linear(4096, 1024),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(1024, 1024),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(1024, 7),
		)

		self.log_soft = nn.LogSoftmax(dim=1)

		self._initialize_weights()

	def _make_block(self, n_layer, features):
		assert n_layer == len(features) - 1
		layers = []

		for idx in range(n_layer):
			layers.append(nn.Conv2d(features[idx], features[idx + 1], 3, padding=1))
			layers.append(nn.ReLU())

		layers.append(nn.Dropout(0.25))
		layers.append(nn.MaxPool2d(2, 2))

		return nn.Sequential(*layers)

	def forward(self, x):
		# Input imgae (64x64)
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)

		x = torch.flatten(x, 1)

		x = self.classifier(x)

		if not self.feature_extract:
			x = self.log_soft(x)

		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


# model = VggFace()
# print(model.fc_6.in_features)
# print(model.fc_6.out_features)
