import json
import logging
import os
import random
# import coordinate_conversion as cc
import numpy as np
import torch
import torch.utils.data as tu_data

class DataGenerator:
    def __init__(self, data_path, minibatch_len, interval=1, use_preset_data_ranges=False,
                 train=True, test=True, dev=True, train_shuffle=True, test_shuffle=False, dev_shuffle=True):
        print(data_path)
        assert os.path.exists(data_path)
        self.attr_names = ['lon', 'lat', 'alt', 'spdx', 'spdy', 'spdz']
        self.data_path = data_path

        self.interval = interval
        self.minibatch_len = minibatch_len

        self.data_status = np.load('data_ranges.npy', allow_pickle=True).item()

        assert type(self.data_status) is dict

        self.preset_data_ranges = {'lon': {'min': -1.6302769184112549, 'max': -0.027176039293408394}, 
                                   'lat': {'min': -0.6707441806793213, 'max': 0.7444194555282593}, 
                                   'alt': {'min': 0.9661445021629333, 'max': 1.0859311819076538}, 
                                   'spdx': {'min': -3.1776148080825806, 'max': 2.0996814966201782}, 
                                   'spdy': {'min': -2.551273852586746, 'max': 2.1026498079299927}, 
                                   'spdz': {'min': -0.19478201866149902, 'max': 0.23588240146636963}}
            
        self.use_preset_data_ranges = use_preset_data_ranges
        if train:
            self.train_set = mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'train'), shuffle=train_shuffle))
            print(self.train_set)
        if dev:
            self.dev_set = mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'dev'), shuffle=dev_shuffle))
            print(self.dev_set)
        if test:
            self.test_set = mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'test'), shuffle=test_shuffle))
            print(self.test_set)
        if use_preset_data_ranges:
            assert self.preset_data_ranges is not None


    def readtxt(self, data_path, shuffle=True):
        assert os.path.exists(data_path)
        data = []
        for root, dirs, file_names in os.walk(data_path):
            for file_name in file_names:
                if not file_name.endswith('txt'):
                    continue
                with open(os.path.join(root, file_name)) as file:
                    lines = file.readlines()
                    lines = lines[::self.interval]
                    if len(lines) == self.minibatch_len:
                        data.append(lines)
                    elif len(lines) < self.minibatch_len:
                        continue
                    else:
                        for i in range(len(lines)-self.minibatch_len+1):
                            data.append(lines[i:i+self.minibatch_len])
        print(f'{len(data)} items loaded from \'{data_path}\'')
        if shuffle:
            random.shuffle(data)
        return data

    def scale(self, inp, attr):
        assert type(attr) is str and attr in self.attr_names
        data_status = self.data_status if not self.use_preset_data_ranges else self.preset_data_ranges
        inp = (inp-data_status[attr]['min'])/(data_status[attr]['max']-data_status[attr]['min'])
        return inp

    def unscale(self, inp, attr):
        assert type(attr) is str and attr in self.attr_names
        data_status = self.data_status if not self.use_preset_data_ranges else self.preset_data_ranges
        inp = inp*(data_status[attr]['max']-data_status[attr]['min'])+data_status[attr]['min']
        return inp

    def collate(self, inp):
        '''
        :param inp: batch * n_sequence * n_attr
        :return:
        '''
        oup = []
        for minibatch in inp:
            tmp = []
            for line in minibatch:
                items = line.strip().split(" ")
                # lon, lat, alt, spdx, spdy, spdz = float(items[4]), float(items[5]), int(float(items[6]) / 10), \
                #                                   float(items[7]), float(items[8]), float(items[9])
                lon, lat, alt, spdx, spdy, spdz = float(items[0]), float(items[1]), float(items[2]), float(items[3]), float(items[4]), float(items[5])
                # print(lon, lat, alt, spdx, spdy, spdz)
                tmp.append([lon, lat, alt, spdx, spdy, spdz])
            minibatch = np.array(tmp)
            for i in range(minibatch.shape[-1]):
                minibatch[:, i] = self.scale(minibatch[:, i], self.attr_names[i])
            oup.append(minibatch)
        return np.array(oup)


class mini_DataGenerator(tu_data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
