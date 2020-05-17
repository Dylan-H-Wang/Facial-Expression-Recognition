import random

from tqdm import tqdm
from tqdm import trange

import cv2

import numpy as np

import h5py

from sklearn.cluster import MiniBatchKMeans


def calc_densen_SIFT(images):
    print("Calculating dense SIFT ...")

    sift = cv2.xfeatures2d.SIFT_create()

    step_size = 5
    descriptors = []

    for img in tqdm(images):

        kp = [
            cv2.KeyPoint(x, y, step_size)
            for y in range(0, img.shape[0], step_size)
            for x in range(0, img.shape[1], step_size)
        ]
        kp, dense_feature = sift.compute(img, kp)

        descriptors.extend(dense_feature)

    return descriptors


def calc_densen_SIFT_single(img):
    sift = cv2.xfeatures2d.SIFT_create()
    step_size = 5

    kp = [
        cv2.KeyPoint(x, y, step_size)
        for y in range(0, img.shape[0], step_size)
        for x in range(0, img.shape[1], step_size)
    ]
    kp, dense_feature = sift.compute(img, kp)

    return dense_feature


def kmeans(n_clusters, descriptors):
    print("Calculating k-means ...")

    kmeans_classifier = []
    for i in tqdm(n_clusters):
        kmeans = MiniBatchKMeans(n_clusters=i, n_init=10)
        kmeans.fit(descriptors)
        kmeans_classifier.append(kmeans)

    return kmeans_classifier


# Reference: https://github.com/TrungTVo/spatial-pyramid-matching-scene-recognition/blob/master/spatial_pyramid.ipynb
# Form histogram with Spatial Pyramid Matching upto level L with codebook kmeans and k codewords
def calc_image_features_spatial_pyramid(img, kmeans):
    level = 2
    width = img.shape[1]
    height = img.shape[0]
    all_hist = []

    for l in range(level + 1):
        w_step = int(np.floor(width / (2 ** l)))
        h_step = int(np.floor(height / (2 ** l)))
        x, y = 0, 0
        k = len(kmeans[l].cluster_centers_)

        for i in range(1, 2 ** l + 1):
            x = 0

            for j in range(1, 2 ** l + 1):
                desc = calc_densen_SIFT_single(img[y : y + h_step, x : x + w_step])
                idx = kmeans[l].predict(desc)
                histogram = np.bincount(idx, minlength=k).reshape(1, -1).ravel()

                weight = 2 ** (l - level)
                all_hist.append(weight * histogram)
                x = x + w_step

            y = y + h_step

    hist = np.concatenate(all_hist)

    # Normalize histogram
    dev = np.std(hist)
    hist -= np.mean(hist)
    hist /= dev
    return hist.astype(np.float32)


# Get histogram representation for training/testing data
def get_histogram_spatial_pyramid(data, kmeans, filename):
    print("Calculating spatial pyramid ...")

    length = len(data)

    with h5py.File(filename, "w") as f:
        all_hist = f.create_dataset("histogram", (length, 201000), compression="lzf")

        for idx in trange(len(data)):
            hist = calc_image_features_spatial_pyramid(data[idx], kmeans)
            all_hist[idx] = hist


def get_histogram(data, kmeans, hist):
    k = len(kmeans.cluster_centers_)

    for i in trange(len(data)):
        desc = calc_densen_SIFT_single(data[i])
        predict = kmeans.predict(desc)
        hist_ = np.bincount(predict, minlength=k).reshape(1, -1).ravel()

        dev = np.std(hist_)
        hist_ = hist_ - np.mean(hist_)
        hist_ = hist_ / dev

        hist[i] = hist_


def iter_minibatches(data, label, batch_size):
    length = data.shape[0]
    # length = 5000
    index = list(range(length))
    random.shuffle(index)

    cursor = 0
    while cursor < length:
        # print("Generating mini-batch ...")
        start = cursor
        if cursor + batch_size > length:
            end = length
        else:
            end = cursor + batch_size

        cur_idx = index[start:end]
        cur_idx.sort()
        sample = np.zeros((len(cur_idx), data.shape[1]))
        for idx, value in enumerate(tqdm(cur_idx)):
            sample[idx] = data[value]
        gt = label[cur_idx]

        yield sample, gt

        cursor = end
