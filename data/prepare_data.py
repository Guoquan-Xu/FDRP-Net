from PIL import Image
from torch.utils.data import Dataset
import random
import data.utils as Utils
import lmdb
import os

class prepareDataset(Dataset):
    def __init__(self,dataroot,datatype,phase,img_size,data_len=-1):
        super(prepareDataset,self).__init__()
        self.datatype=datatype
        self.phase=phase
        self.img_size=img_size
        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                dataset_len = int(txn.get("length".encode("utf-8")))
            if data_len <= 0:
                self.data_len = dataset_len
            else:
                self.data_len = min(data_len, dataset_len)
        elif datatype == 'img':
            self.train_path = Utils.get_paths_from_images(dataroot['train'])
            self.reference_path = Utils.get_paths_from_images(dataroot['reference'])
            dataset_len = len(self.train_path)

            if data_len <= 0:
                self.data_len = dataset_len
            else:
                self.data_len = min(data_len, dataset_len)
                self.train_path = self.train_path[:self.data_len]
                self.reference_path = self.reference_path[:self.data_len]

            # if phase == 'train':
            #     self.targets = []
            #     for path in self.train_path:
            #         filename = os.path.basename(path)
            #         if 'LSUI' in filename:
            #             label = 0
            #         elif 'UIEB' in filename:
            #             label = 1
            #         else:
            #             label = -1
            #         self.targets.append(label)
            #     assert all(label in [0, 1] for label in self.targets), "存在未识别的类别标签"
            #     assert len(self.targets) == self.data_len, "<UNK>"

        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))
    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        if self.datatype == 'img':
            train_path = self.train_path[item]
            ref_path = self.reference_path[item]
            train_name = os.path.splitext(os.path.basename(train_path))[0]
            ref_name = os.path.splitext(os.path.basename(ref_path))[0]
            assert train_name == ref_name, \
                f"图像对不匹配: 训练图 '{train_name}' 与参考图 '{ref_name}' 文件名不一致 (索引 {item})"

            trainImg = Image.open(train_path).convert('RGB')
            referenceImg = Image.open(ref_path).convert('RGB')
        trainImg, referenceImg = Utils.transform_augment(
            [trainImg, referenceImg], img_size=self.img_size, phase=self.phase, min_max=(-1, 1)
        )
        return {'trainImg': trainImg, 'referenceImg': referenceImg, 'Item': item},train_name

    def get_labels(self):
        labels = []
        for path in self.train_path:
            filename = os.path.basename(path)
            if 'LSUI' in filename:
                label = 0
            elif 'UIEB' in filename:
                label = 1
            else:
                label = -1

            labels.append(label)
        assert all(label in [0, 1] for label in labels), "存在未识别的类别标签"
        assert len(labels) == self.data_len, "<UNK>"

        return labels

