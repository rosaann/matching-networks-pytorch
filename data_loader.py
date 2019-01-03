# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: BoyuanJiang
# College of Information Science & Electronic Engineering,ZheJiang University
# Email: ginger188@gmail.com
# Copyright (c) 2017

# @Time    :17-8-27 10:46
# @FILE    :data_loader.py
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import os
import pickle
import tqdm
import concurrent.futures
import cv2

class OmniglotNShotDataset():
    def __init__(self, batch_size, classes_per_set=20, samples_per_class=1, seed=2017, shuffle=True, use_cache=True):
        """
        Construct N-shot dataset
        :param batch_size:  Experiment batch_size 
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        :param seed: seed for random function
        :param shuffle: if shuffle the dataset
        :param use_cache: if true,cache dataset to memory.It can speedup the train but require larger memory ff
        """
        np.random.seed(seed)
        print('enter 1')
      #  self.x = np.load('data/data.npy')
      #  print('x ', self.x)
      #  print('enter x.shape ', self.x.shape)
      #  self.x = np.reshape(self.x, newshape=(self.x.shape[0], self.x.shape[1], 28, 28, 1))
     #   print('enter after x.shape ', self.x.shape)
     #   if shuffle:
     #       np.random.shuffle(self.x)
      #  self.x_train, self.x_val, self.x_test = self.x[:1200], self.x[1200:1411], self.x[1411:]
        # self.mean = np.mean(list(self.x_train) + list(self.x_val))
        self.indexes_of_folders_indicating_class=[-2, -3]
        self.train_val_test_split=(1200/1622, 211/1622, 211/1622)
        self.x_train, self.x_val, self.x_test = self.load_dataset()
        self.x_train = self.processes_batch(self.x_train, np.mean(self.x_train), np.std(self.x_train))
        self.x_test = self.processes_batch(self.x_test, np.mean(self.x_test), np.std(self.x_test))
        self.x_val = self.processes_batch(self.x_val, np.mean(self.x_val), np.std(self.x_val))
        # self.std = np.std(list(self.x_train) + list(self.x_val))
        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datatset = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        self.use_cache = use_cache
        if self.use_cache:
            self.cached_datatset = {"train": self.load_data_cache(self.x_train),
                                    "val": self.load_data_cache(self.x_val),
                                    "test": self.load_data_cache(self.x_test)}
    def load_dataset(self):
        data_image_paths, index_to_label_name_dict_file, label_to_index = self.load_datapaths()
        total_label_types = len(data_image_paths)
        print(total_label_types)
        # data_image_paths = self.shuffle(data_image_paths)
        x_train_id, x_val_id, x_test_id = int(self.train_val_test_split[0] * total_label_types), \
                                          int(np.sum(self.train_val_test_split[:2]) * total_label_types), \
                                          int(total_label_types)
        print(x_train_id, x_val_id, x_test_id)
        x_train_classes = (class_key for class_key in list(data_image_paths.keys())[:x_train_id])
        x_val_classes = (class_key for class_key in list(data_image_paths.keys())[x_train_id:x_val_id])
        x_test_classes = (class_key for class_key in list(data_image_paths.keys())[x_val_id:x_test_id])
        x_train, x_val, x_test = {class_key: data_image_paths[class_key] for class_key in x_train_classes}, \
                                 {class_key: data_image_paths[class_key] for class_key in x_val_classes}, \
                                 {class_key: data_image_paths[class_key] for class_key in x_test_classes},
        return x_train, x_val, x_test
    
    def load_datapaths(self):
        self.dataset_name = "omniglot_dataset"
        data_path_file = "datasets/{}.pkl".format(self.dataset_name)
        self.index_to_label_name_dict_file = "datasets/map_to_label_name_{}.pkl".format(self.dataset_name)
        self.label_name_to_map_dict_file = "datasets/label_name_to_map_{}.pkl".format(self.dataset_name)

     #   if self.reset_stored_filepaths == True:
     #       if os.path.exists(data_path_file):
     #           os.remove(data_path_file)
     #           self.reset_stored_filepaths=False

        try:
            data_image_paths = self.load_dict(data_path_file)
            label_to_index = self.load_dict(name=self.label_name_to_map_dict_file)
            index_to_label_name_dict_file = self.load_dict(name=self.index_to_label_name_dict_file)
            return data_image_paths, index_to_label_name_dict_file, label_to_index
        except:
            print("Mapped data paths can't be found, remapping paths..")
            data_image_paths, code_to_label_name, label_name_to_code = self.get_data_paths()
            self.save_dict(data_image_paths, name=data_path_file)
            self.save_dict(code_to_label_name, name=self.index_to_label_name_dict_file)
            self.save_dict(label_name_to_code, name=self.label_name_to_map_dict_file)
            return self.load_datapaths()
        
    def get_data_paths(self):
        self.data_path = "datasets/omniglot_dataset"

        print("Get images from", self.data_path)
        data_image_path_list_raw = []
        labels = set()
        for subdir, dir, files in os.walk(self.data_path):
            for file in files:
                if (".jpeg") in file.lower() or (".png") in file.lower() or (".jpg") in file.lower():
                    filepath = os.path.join(subdir, file)
                    label = self.get_label_from_path(filepath)
                    data_image_path_list_raw.append(filepath)
                    labels.add(label)

        labels = sorted(labels)
        idx_to_label_name = {idx: label for idx, label in enumerate(labels)}
        label_name_to_idx = {label: idx for idx, label in enumerate(labels)}
        data_image_path_dict = {idx: [] for idx in list(idx_to_label_name.keys())}
        with tqdm.tqdm(total=len(data_image_path_list_raw)) as pbar_error:
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                # Process the list of files, but split the work across the process pool to use all CPUs!
                for image_file in executor.map(self.load_test_image, (data_image_path_list_raw)):
                    pbar_error.update(1)
                    if image_file is not None:
                        label = self.get_label_from_path(image_file)
                        data_image_path_dict[label_name_to_idx[label]].append(image_file)


        return data_image_path_dict, idx_to_label_name, label_name_to_idx
    def get_label_from_path(self, filepath):
        label_bits = filepath.split("/")
        label = "_".join([label_bits[idx] for idx in self.indexes_of_folders_indicating_class])
      #  if self.labels_as_int:
      #      label = int(label)
        return label
    
    def load_test_image(self, filepath):
        try:
            image = cv2.imread(filepath)
            image = cv2.resize(image, dsize=(28, 28))
        except RuntimeWarning:
            os.system("convert {} -strip {}".format(filepath, filepath))
            print("converting")
            image = cv2.imread(filepath)
            image = cv2.resize(image, dsize=(28, 28))
        except:
            print("Broken image")
            os.remove(filepath)

        if image is not None:
            return filepath
        else:
            os.remove(filepath)
            return None
    def save_dict(self, obj, name):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_dict(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)
        
    def processes_batch(self, x_batch, mean, std):
        """
        Normalizes a batch images
        :param x_batch: a batch images
        :return: normalized images
        """
        return (x_batch - mean) / std

    def _sample_new_batch(self, data_pack):
        """
        Collect 1000 batches data for N-shot learning
        :param data_pack: one of(train,test,val) dataset shape[classes_num,20,28,28,1]
        :return: A list with [support_set_x,support_set_y,target_x,target_y] ready to be fed to our networks
        """
        support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                  data_pack.shape[3], data_pack.shape[4]), np.float32)

        support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
        target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]), np.float32)
        target_y = np.zeros((self.batch_size, 1), np.int32)

        for i in range(self.batch_size):
            classes_idx = np.arange(data_pack.shape[0])
            samples_idx = np.arange(data_pack.shape[1])
            choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
            choose_label = np.random.choice(self.classes_per_set, size=1)
            choose_samples = np.random.choice(samples_idx, size=self.samples_per_class + 1, replace=False)

            x_temp = data_pack[choose_classes]
            x_temp = x_temp[:, choose_samples]
            y_temp = np.arange(self.classes_per_set)
            support_set_x[i] = x_temp[:, :-1]
            support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
            target_x[i] = x_temp[choose_label, -1]
            target_y[i] = y_temp[choose_label]

        return support_set_x, support_set_y, target_x, target_y

    def _rotate_data(self, image, k):
        """
        Rotates one image by self.k * 90 degrees counter-clockwise
        :param image: Image to rotate
        :return: Rotated Image
        """
        return np.rot90(image, k)

    def _rotate_batch(self, batch_images, k):
        """
        Rotates a whole image batch
        :param batch_images: A batch of images
        :param k: integer degree of rotation counter-clockwise
        :return: The rotated batch of images
        """
        batch_size = batch_images.shape[0]
        for i in np.arange(batch_size):
            batch_images[i] = self._rotate_data(batch_images[i], k)
        return batch_images

    def _get_batch(self, dataset_name, augment=False):
        """
        Get next batch from the dataset with name.
        :param dataset_name: The name of dataset(one of "train","val","test")
        :param augment: if rotate the images
        :return: a batch images
        """
        if self.use_cache:
            support_set_x, support_set_y, target_x, target_y = self._get_batch_from_cache(dataset_name)
        else:
            support_set_x, support_set_y, target_x, target_y = self._sample_new_batch(self.datatset[dataset_name])
        if augment:
            k = np.random.randint(0, 4, size=(self.batch_size, self.classes_per_set))
            a_support_set_x = []
            a_target_x = []
            for b in range(self.batch_size):
                temp_class_set = []
                for c in range(self.classes_per_set):
                    temp_class_set_x = self._rotate_batch(support_set_x[b, c], k=k[b, c])
                    if target_y[b] == support_set_y[b, c, 0]:
                        temp_target_x = self._rotate_data(target_x[b], k=k[b, c])
                    temp_class_set.append(temp_class_set_x)
                a_support_set_x.append(temp_class_set)
                a_target_x.append(temp_target_x)
            support_set_x = np.array(a_support_set_x)
            target_x = np.array(a_target_x)
        support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                                               support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
        support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])
        return support_set_x, support_set_y, target_x, target_y

    def get_train_batch(self, augment=False):
        return self._get_batch("train", augment)

    def get_val_batch(self, augment=False):
        return self._get_batch("val", augment)

    def get_test_batch(self, augment=False):
        return self._get_batch("test", augment)

    def load_data_cache(self, data_pack, argument=True):
        """
        cache the dataset in memory
        :param data_pack: shape[classes_num,20,28,28,1]
        :return:
        """
        cached_dataset = []
        classes_idx = np.arange(data_pack.shape[0])
        samples_idx = np.arange(data_pack.shape[1])
        for _ in range(1000):
            support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                      data_pack.shape[3], data_pack.shape[4]), np.float32)

            support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
            target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]),
                                np.float32)
            target_y = np.zeros((self.batch_size, 1), np.int32)
            for i in range(self.batch_size):
                choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
                choose_label = np.random.choice(self.classes_per_set, size=1)
                choose_samples = np.random.choice(samples_idx, size=self.samples_per_class + 1, replace=False)

                x_temp = data_pack[choose_classes]
                x_temp = x_temp[:, choose_samples]
                y_temp = np.arange(self.classes_per_set)
                support_set_x[i] = x_temp[:, :-1]
                support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
                target_x[i] = x_temp[choose_label, -1]
                target_y[i] = y_temp[choose_label]
            cached_dataset.append([support_set_x, support_set_y, target_x, target_y])
        return cached_dataset

    def _get_batch_from_cache(self, dataset_name):
        """

        :param dataset_name:
        :return:
        """
        if self.indexes[dataset_name] >= len(self.cached_datatset[dataset_name]):
            self.indexes[dataset_name] = 0
            self.cached_datatset[dataset_name] = self.load_data_cache(self.datatset[dataset_name])
        next_batch = self.cached_datatset[dataset_name][self.indexes[dataset_name]]
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target = next_batch
        return x_support_set, y_support_set, x_target, y_target
