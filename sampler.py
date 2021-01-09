#为dataloader构建sampler
import math
import torch
import utils
import config as cfg
import numpy as np
from torch.utils.data import Sampler

class groupSampler(Sampler):
    #代码来源于mmdetection
    #不采用分布式时的sampler
    def __init__(self, dataset):
        self.dataset = dataset
        self.samples_per_gpu = cfg.samples_per_gpu
        self.img_flag = dataset.img_flags.astype(np.int64)

        #按照长宽比对图片进行分类，尽可能减少显存
        self.group_sizes = np.bincount(self.img_flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += math.ceil(size/self.samples_per_gpu) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.img_flag == i)[0]
            np.random.shuffle(indice)
            num_extra = math.ceil(size/self.samples_per_gpu)*self.samples_per_gpu-len(indice)
            indice = np.concatenate([indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [indices[i*self.samples_per_gpu:(i+1)*self.samples_per_gpu] for i in np.random.permutation(range(len(indices)//self.samples_per_gpu))]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples

class distributedGroupSampler(Sampler):
    def __init__(self,dataset):
        self.dateset=dataset
        rank, world_size=utils.get_dist_info()
        self.rank=rank
        self.world_size=world_size
        self.samples_per_gpu=cfg.samples_per_gpu
        self.epoch=0
        self.img_flag = dataset.img_flags.astype(np.int64)
        self.group_sizes = np.bincount(self.img_flag)

        self.num_samples=0
        for i,size in enumerate(self.group_sizes):
            self.num_samples+=math.ceil(size/(self.samples_per_gpu * self.world_size))*self.samples_per_gpu
        self.total_size=self.num_samples*self.world_size

    def __iter__(self):
        #设置随机种子，保证了每个进程产生的随机序列是相同的，保证这个随机序列的同步，然后每个进程取这个序列的一部分。
        g=torch.Generator()
        g.manual_seed(self.epoch)

        indices=[]
        for i,size in enumerate(self.group_sizes):
            if size==0:
                continue
            indice=np.where(self.img_flag==i)[0]
            indice=indice[list(torch.randperm(int(size),generator=g))].tolist()
            extra=math.ceil(size/(self.samples_per_gpu*self.world_size))*self.samples_per_gpu*self.world_size-len(indice)

            tmp=indice.copy()
            for _ in range(extra//size):
                indice.extend(tmp)
            indice.extend(tmp[:extra % size])
            indices.extend(indice)

        #截止到目前，站在图片长宽比的角度，indices还没有被打乱，因此需要为长宽比引入随机
        indices=[indices[j] for i in list(torch.randperm(len(indices) // self.samples_per_gpu,generator=g))
                 for j in range(i*self.samples_per_gpu,(i+1)*self.samples_per_gpu)
                 ]

        offset=self.num_samples*self.rank
        indices=indices[offset:offset+self.num_samples]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
