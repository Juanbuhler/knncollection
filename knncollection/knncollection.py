import numpy as np
import cv2
import mxnet as mx
from collections import namedtuple
import os
import pickle

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import KNeighborsClassifier

class HashCalculator(object):
    def __init__(self):
        path = 'http://data.mxnet.io/models/imagenet-11k/'
        [mx.test_utils.download(path + 'resnet-152/resnet-152-symbol.json'),
         mx.test_utils.download(path + 'resnet-152/resnet-152-0000.params'),
         mx.test_utils.download(path + 'synset.txt')]
        sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
        mod = mx.mod.Module(symbol=sym, context=mx.cpu())
        mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
        mod.set_params(arg_params, aux_params)
        all_layers = sym.get_internals()
        fe_sym = all_layers['flatten0_output']

        self.model = mx.mod.Module(symbol=fe_sym, context=mx.cpu(), label_names=None)
        self.model.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
        self.model.set_params(arg_params, aux_params)

    def get_feature_vector(self, images):
        Batch = namedtuple('Batch', ['data'])
        self.model.forward(Batch([mx.nd.array(np.array(images))]))
        features = self.model.get_outputs()[0].asnumpy()
        features = np.squeeze(features)

        mxh = np.clip((features * 16).astype(int), 0, 15) + 65
        mxh = np.asarray(mxh, dtype=np.float64)

        return mxh


class KnnModel(object):
    def __init__(self, name):
        self.name = name + '.pickle'

        if os.path.isfile(self.name):
            self.labels, self.predictions, self.confidences = pickle.load(open(self.name, 'rb'))
            return

        self.labels = {}
        self.predictions = None
        self.confidences = None

    def label(self, file, sim_hash, value, scope='train'):
        self.labels[file] = [sim_hash, value, scope]

    def auto_label(self, files, hashes, n_clusters = 10, n_max = 5000):
        sorter = np.argsort(np.random.rand(len(files)))
        cluster_files = [files[j] for j in sorter][:n_max]
        cluster_hashes = [hashes[j] for j in sorter][:n_max]

        cluster_hashes = np.asarray(cluster_hashes, dtype=np.float64)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cluster_hashes)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, cluster_hashes)

        for n, i in enumerate(closest):
            self.label(cluster_files[i], cluster_hashes[i], 'auto'+str(n))
        self.save_model()
        self.train_apply(hashes)


    def save_model(self):
        print('saving ' + self.name)
        pickle.dump((self.labels, self.predictions, self.confidences), open(self.name, 'wb'))

    def train_apply(self, hashes):
        xt = [value[0] for key, value in self.labels.items()]
        yt = [value[1] for key, value in self.labels.items()]

        xtrain = np.asarray(xt, dtype=np.float64)
        ytrain = np.asarray(yt)

        classifier = KNeighborsClassifier(
            n_neighbors=1, p=1, weights='distance', metric='manhattan')
        classifier.fit(xtrain, ytrain)

        prediction = classifier.predict(hashes)
        dist, ind = classifier.kneighbors(hashes, n_neighbors=1, return_distance=True)
        conf = 1 - np.concatenate(dist) / np.max(np.concatenate(dist))

        self.predictions = prediction
        self.confidences = conf

        self.save_model()


class KnnCollection(object):
    def __init__(self, files, save_name='knnCollection'):
        save_name = save_name + '.pickle'

        # If there's a pickle file in there, just read it and return
        if os.path.isfile(save_name):
            self.files, self.hashes = pickle.load(open(save_name, 'rb'))

            diff = [x for x in files if x not in self.files]
            if not diff:
                return
            files = diff
            print('Working on ' + str(len(files)) + ' files')
        else:
            self.files = []
            self.hashes = np.empty([0, 2048], dtype=int)

        calculator = HashCalculator()

        while files:
            print(len(files))
            f = files[:100]
            files = files[100:]
            new_images = [np.transpose(cv2.resize(cv2.imread(file), (224, 224)), (2, 0, 1)) for file in f]

            mxh = calculator.get_feature_vector(new_images)
            self.hashes = np.concatenate((self.hashes, mxh))
            self.files = self.files + f

            pickle.dump((self.files, self.hashes), open(save_name, 'wb'))

    def similarity_search(self, target, num_assets=50):
        knn = NearestNeighbors(n_neighbors=num_assets, p=1)
        knn.fit(self.hashes)
        neighbors = knn.kneighbors(target.reshape(1, -1), return_distance=False)

        return neighbors[0].tolist()
