#from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import json
import numpy as np
import tensorflow as tf
import sys



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path=os.path.abspath(os.path.join(BASE_DIR, '../../../3d-point-capsule-affordances-DCG/dataset/affordances/'))

class PartDataset(data.Dataset):
    def __init__(self, root=dataset_path, npoints=2500, classification=False, class_choice=None, split='train', normalize=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.classification = classification
        self.normalize = normalize

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        self.meta = {}
        
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str("/".join(d.split('/')[-2:])) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str("/".join(d.split('/')[-2:])) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str("/".join(d.split('/')[-2:])) for d in json.load(f)])

        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            fns = sorted(os.listdir(dir_point))
            fns = [f'{self.cat[item]}/{fn}' for fn in fns]
            
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)
            
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'),self.cat[item], token))            
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2], fn[3]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath)//50):
                l = len(np.unique(np.loadtxt(self.datapath[i][2]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 18000

    def __getitem__(self, index):
        if index in self.cache:
#            point_set, seg, cls= self.cache[index]
            point_set, seg, cls, foldername, filename = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
#            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            if self.normalize:
                point_set = self.pc_normalize(point_set)
            seg = np.loadtxt(fn[2]).astype(np.int64) - 1
            foldername = fn[3]
            filename = fn[4]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, seg, cls, foldername, filename)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        
        # To Pytorch
        # point_set = tf.convert_to_tensor(point_set)
        # seg = tf.convert_to_tensor(seg)
        # cls = tf.convert_to_tensor(np.array([cls]), dtype=tf.float32)
        cls = np.array(cls,dtype=np.int32)
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg , cls
        

    def __len__(self):
        return len(self.datapath)
       
    def pc_normalize(self, pc):
        """ pc: NxC, return NxC """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


if __name__ == '__main__':
    d = PartDataset( root=dataset_path,classification=True, class_choice='Mug', npoints=2048, split='test')
    ps, cls = d[0]
    print(ps.shape, cls.shape)
