import os
import torch
import numpy
import random


def create_path_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def make_a_ndarray_batch(ndarray, batch_size):
    tensor = torch.from_numpy(ndarray).to(torch.float32)
    size = torch.ones((batch_size, *ndarray.shape))
    return torch.mul(tensor, size).cuda()


def random_in_range(r_range: tuple):
    scale = abs(r_range[1] - r_range[0])
    return random.random() * scale + r_range[0]


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

# a=numpy.array([[1.2, 3.2],[3.3, 4.4]])
# batch_num = 4
# print(make_a_ndarray_batch(a, batch_num))
