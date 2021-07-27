import os
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
#from utils.provider import rotate_point_cloud_SO3, rotate_point_cloud_y
import pickle as pkl
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path=os.path.abspath(os.path.join(BASE_DIR, '../../../3d-point-capsule-affordances-DCG/dataset/affordance_net'))

all_classes = {'Bag': 0, 'Bed': 1, 'Bottle': 2, 'Bowl': 3, 'Chair': 4, 'Clock': 5, 'Dishwasher': 6, 'Display': 7, 'Door': 8, 'Earphone': 9, 'Faucet': 10, 'Hat': 11, 'Keyboard': 12, 'Knife': 13, 'Laptop': 14, 'Microwave': 15, 'Mug': 16, 'Refrigerator': 17, 'Scissors': 18, 'StorageFurniture': 19, 'Table': 20, 'TrashCan': 21, 'Vase': 22}
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

class AffordNetDataset(Dataset):
    def __init__(self, data_dir=dataset_path, split='train', partial=False, rotate='None', semi=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.split_file = "train" if split == "test" else split

        self.partial = partial
        self.rotate = rotate
        self.semi = semi

        self.load_data()

        self.affordance = self.all_data[0]["affordance"]

        return

    def load_data(self):
        self.all_data = []
        if self.semi:
            with open(opj(self.data_dir, 'semi_label_1.pkl'), 'rb') as f:
                temp_data = pkl.load(f)
        else:
            if self.partial:
                with open(opj(self.data_dir, 'partial_%s_data.pkl' % self.split_file), 'rb') as f:
                    temp_data = pkl.load(f)
            elif self.rotate != "None" and self.split_file != 'train':
                with open(opj(self.data_dir, 'rotate_%s_data.pkl' % self.split_file), 'rb') as f:
                    temp_data_rotate = pkl.load(f)
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split_file), 'rb') as f:
                    temp_data = pkl.load(f)
            else:
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split_file), 'rb') as f:
                    temp_data = pkl.load(f)
        if self.split_file == "train":
            temp_data_train, temp_data_test = train_test_split(temp_data, test_size=0.20, random_state=42) 
            if self.split == "train":
                temp_data = temp_data_train
            else:
                temp_data = temp_data_test
        for index, info in enumerate(temp_data):
            if self.partial:
                partial_info = info["partial"]
                for view, data_info in partial_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["view_id"] = view
                    temp_info["data_info"] = data_info
                    self.all_data.append(temp_info)
            elif self.split != 'train' and self.rotate != 'None':
                rotate_info = temp_data_rotate[index]["rotate"][self.rotate]
                full_shape_info = info["full_shape"]
                for r, r_data in rotate_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["data_info"] = full_shape_info
                    temp_info["rotate_matrix"] = r_data.astype(np.float32)
                    self.all_data.append(temp_info)
            else:
                temp_info = {}
                temp_info["shape_id"] = info["shape_id"]
                temp_info["semantic class"] = info["semantic class"]
                temp_info["affordance"] = info["affordance"]
                temp_info["data_info"] = info["full_shape"]
                self.all_data.append(temp_info)

    def __getitem__(self, index):

        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]
        modelcat = data_dict["semantic class"]

        data_info = data_dict["data_info"]
        model_data = data_info["coordinate"].astype(np.float32)
        labels = data_info["label"]
        for aff in self.affordance:
            temp = labels[aff].astype(np.float32).reshape(-1, 1)
            model_data = np.concatenate((model_data, temp), axis=1)

        datas = model_data[:, :3]
        targets = model_data[:, 3:]

        if self.rotate != 'None':
            if self.split == 'train':
                if self.rotate == 'so3':
                    datas = rotate_point_cloud_SO3(
                        datas[np.newaxis, :, :]).squeeze()
                elif self.rotate == 'z':
                    datas = rotate_point_cloud_y(
                        datas[np.newaxis, :, :]).squeeze()
            else:
                r_matrix = data_dict["rotate_matrix"]
                datas = (np.matmul(r_matrix, datas.T)).T

        datas, _, _ = pc_normalize(datas)

        max_target_index = np.argmax(targets,axis=1)
        max_target_value = np.max(targets,axis=1)

        max_target_index = np.where(max_target_value == 0.,18,max_target_index)

        modelcat_index = all_classes[modelcat]

        return datas, max_target_index, modelcat_index

    def __len__(self):
        return len(self.all_data)

if __name__ == '__main__':
    d = AffordNetDataset(dataset_path,split='train')

    for i in d:
        
        ps, targets, modelcat = i

        print(modelcat)
