# Local Learning with Deep and Handcrafted Features for Facial Expression Recognition Implementation

This project implements the method described in this [report](https://arxiv.org/abs/1804.10892). It consists of three parts: deep model part, handcraft model part, local learning part.

## Demo

tbc

## `dataset.py`

This file is used to load and preprocess the data which can be used as inputs of deep model and handcraft model.

`FER2013Dataset` class: It reads data from `data` folder, and there should be three files for training, validating and testing respectively named `train.csv`, `validation.csv` and `test.csv`. The image size can be scaled by specifing `scale_size` and it should be compatible with size `(48, 48)`. The image is augmented by randomly horizontally flip with possibility of `0.5`. For deep model usage, you can use `Dataloader (PyTorch)` to iterate the whole dataset, and for handcraft model, you can call `get_data` to obtain the data.

## `deep_model.py`

This file implements three deep models including `vgg_face`, `vgg_f` and `vgg_13`. All models supports `DSD` training process, it can be achieved by calling `make_sparse`. When `make_sparse` is called with `True` and sparsity value, the model is switched into *sparse* mode, and you can make it back by call `make_sparse` with `False`. If you want to use these model as feature extractor, you can call `extract_features` and outputs of this method will be the output of the last activation linear layer instead of `log_softmax` layer.

`VggFace` class: It implements `vgg_face` model. You can specify `pertrain` parameter to get the pretrained model and the model data should save in `models` folder with name `vgg_face.t7`. Since the pretrained model data is in `torch` format, `_load_weights` is used to transfer it into `Pytorch` format.

`Vggf` class: It has similar structure with `VggFace` class except that it also implements *Local Response Normalisatio(LRN)* as model layers.

`Vgg13` class: It has similar structure with `VggFace` class, but it does not support loading pretrained model.

## `handcraft_model.py`

This file implements methods which can be used for extracting image features and then can be inputs of handcraft model.

`calc_densen_SIFT()`: It computes dense SFIT from the image using step size of `5`, and it returns descriptors of each image.

`calc_densen_SIFT_single()`: It can only deal with singe image.

`kmeans()`: It creates `KMeans` model based on given inputs. If `n_clusters` contains mutilple numbers, it will return the same number of `KMeans` models.

`calc_image_features_spatial_pyramid()`: It form histogram with *Spatial Pyramid Matching* upto level 2 for each image using given `KMeans` models predict each feature label.

`get_histogram_spatial_pyramid()`: It deals with batch of images and calls `calc_image_features_spatial_pyramid`.

`iter_minibatches()`: Since the data is too large to fit into RAM, this method can generate mini-bach of data which can then be fitted into SVM classifier. It shuffles data each time.

## `train.py`

`set_parameter_requires_grad()`: It freezes each layer parameter of the given model.

`conduct_train()`: General method for performing training process.

`extract_deep_model()`: Extract feature vectors from three deep models for each trainning sample and they will be saved in `data` folder.

`extract_handcraft_model()`: It trains `KMeans` models firsr using dense SIFT extracted from images, and then calculates BOVW features with spatial information based on the `KMeans` model for each training and testing sample and they will be saved in `data` folder.

`combine_features()`: It combines deep model features and handcraft model features and saves them in `data` folder.

`cal_similarity()`: It calculates cosine similarity between testing sample features and training sample features, and the result will be saved in `data` folder.

`train_combine()`: It trains model using local SVM learning algorithm. Firstly, it finds 200 nearest neighbours of the testing sample using cosine similarity, and then apply linear SVM on the part of sample and make prediction for the testing sample.

`train_combine_global()`: It trains model using global SVM algorithm which fit the whole training samples into the model. Since the inputs are too big to load, it uses mini-batch to train the model and the batch size is set to `1000`. The trained model will be saved in `models` folder

Rest of methods functions explicitly as their names.

**Note:** During the deep model training process, we found that `DSD` training process does not work as good as what we expect. Thus, we only perform the first dense phase which is just regular train process.

## `main.py`