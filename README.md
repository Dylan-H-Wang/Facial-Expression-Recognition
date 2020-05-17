# Local Learning with Deep and Handcrafted Features for Facial Expression Recognition Implementation

This project implements the method described in this [report](https://arxiv.org/abs/1804.10892). It consists of three parts: deep model part, handcraft model part, local learning part.

## Demo

### Dependencies
To make the program running, you need to make sure following packages are installed in your environment. All of them can be installed using `pip`.

* `pytorch`: deep learning framework.
* `sklearn`: machine learning framework.
* `opencv`: machine learning framework.
* `PIL`: Process images.
* `matplotlib`: Plot training process.
* `numpy`: process data.
* `pandas`: process data.
* `h5py`: manage file I/O.
* `torchfile`: manage `torch(lua)` file in `pytorch`.
* `tqdm`: progress bar.

### Usage

#### Download code and data files

You can go to [release](https://github.com/Dylan-H-Wang/Facial-Expression-Recognition/releases) page to download the coding and all of data files.

In the `models` package, it contains pre-trained `vgg_face` and `vgg_f` models, and our models including `vgg_face_no_dsd`, `vgg_f_no_dsd`, `vgg_13_no_dsd` and `kmeans`.

In the `data` package, it contains dataset `train`, `validation` and `test`, and the rest is used to train handcraft model, since the data is too big to fit into RAM, you have to save it in the disk to conduct the training.

Make sure the `data` and `models` packages are in the same level of other python files. e.g.
```
|____Facial-Expression-Recognition
     |_____models
           |_____ model files...
           
     |_____data
           |_____ data files ...
           
     |_____python files
```

#### Run the code

To run the code, you can use flag `--mode` or `-m` to specify which function you want to test. There are two modes `demo` and `train`. 

If you choose `train` mode, it will train the model fram scratch. 

If you choose `demo` mode, it will show the accuracy of trained models including `vgg_face`, `vgg_f` and `vgg_13`. The training process including training loss, val loss and val accuracy will be saved as `vgg_face_process.png`, `vgg_f_process.png` and `vgg_13_process.png` in the current folder. Also, it will pick three images from the testing dataset and make predictions. Picked images will be saved as `test_sample_0.png`, `test_sample_1.png` and `test_sample_3.png` in the current folder. The `demo` mode is set by default.

Example command line input:

```
python main.py -m demo
```

Example output:
```
------ demo mode ------
Display testing accuracy of each model:
Preparing test dataset ...
test dataset contains 3589 samples
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:13<00:00,  2.14it/s]
Finish vgg_face model test with accuracy 0.6959
Preparing test dataset ...
test dataset contains 3589 samples
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:01<00:00, 11.24it/s]
Finish vgg_f model test with accuracy 0.6120
Preparing test dataset ...
test dataset contains 3589 samples
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:01<00:00,  5.40it/s]
Finish vgg_13 model test with accuracy 0.6324

Saving deep model training process ...

Selecting 3 images from testing set
Preparing test dataset ...
test dataset contains 3589 samples
Preparing test dataset ...
test dataset contains 3589 samples
Making prediction ...

 Labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

Object: test_sample_0.jpg
Ground truth: 0
vgg_face: 0; vgg_f: 0; vgg_13: 0
------

Object: test_sample_1.jpg
Ground truth: 4
vgg_face: 4; vgg_f: 6; vgg_13: 6
------

Object: test_sample_2.jpg
Ground truth: 2
vgg_face: 0; vgg_f: 4; vgg_13: 6
------
```

Training process

![vgg_face_train](https://github.com/Dylan-H-Wang/Facial-Expression-Recognition/blob/master/vgg_face_process.png)
![vgg_f_train](https://github.com/Dylan-H-Wang/Facial-Expression-Recognition/blob/master/vgg_f_process.png)
![vgg_13_train](https://github.com/Dylan-H-Wang/Facial-Expression-Recognition/blob/master/vgg_13_process.png)

Test images

![sample_1](https://github.com/Dylan-H-Wang/Facial-Expression-Recognition/blob/master/test_sample_0.png)
![sample_2](https://github.com/Dylan-H-Wang/Facial-Expression-Recognition/blob/master/test_sample_1.png)
![sample_3](https://github.com/Dylan-H-Wang/Facial-Expression-Recognition/blob/master/test_sample_2.png)

### Results

Since the dataset is augmented by random horizontally flip, the result may vary a bit.

| Models   | Accuracy |
|----------|----------|
| vgg_face | 0.69     |
| vgg_f    | 0.64     |
| vgg_13   | 0.65     |

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
