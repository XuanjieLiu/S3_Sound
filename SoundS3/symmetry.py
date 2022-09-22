import math
import random
import time

import torch
import numpy as np

ROTATION = 'Rot'
TRANSLATION = 'Trs'


def rand_translation(dim=np.array([1, 0, 1]), is_std_normal=False, t_range=(-1, 1)):
    scale = t_range[1] - t_range[0]
    if is_std_normal:
        s = torch.randn(len(dim))
    else:
        s = torch.rand(len(dim)) * scale + t_range[0]
    s = s.mul(torch.from_numpy(dim))
    s_r = -s
    return s, s_r


# def rate_2d(angle, dim=np.array([1,0,1]))

def rota_Y(angle):
    s = np.array([
        [math.cos(angle), 0, -math.sin(angle)],
        [0, 1, 0],
        [math.sin(angle), 0, math.cos(angle)]
    ])
    s_r = np.array([
        [math.cos(-angle), 0, -math.sin(-angle)],
        [0, 1, 0],
        [math.sin(-angle), 0, math.cos(-angle)]
    ])
    return s, s_r


def rand_rotation_Y(angel_range=(-0.25 * math.pi, 0.25 * math.pi)):
    scale = angel_range[1] - angel_range[0]
    theta = random.random() * scale + angel_range[0]
    s, s_r = rota_Y(theta)
    return torch.from_numpy(s).to(torch.float32), torch.from_numpy(s_r).to(torch.float32), theta


def rotation_y_mat(theta, batch_size):
    return torch.stack([
        torch.stack([torch.cos(theta), torch.zeros(batch_size), -torch.sin(theta)], dim=0),
        torch.stack([torch.zeros(batch_size), torch.ones(batch_size), torch.zeros(batch_size)], dim=0),
        torch.stack([torch.sin(theta), torch.zeros(batch_size), torch.cos(theta)], dim=0)
    ], dim=0).cuda().permute(2, 0, 1).contiguous()


def make_rotation_Y_batch(batch_size, angle_range=(-0.25 * math.pi, 0.25 * math.pi)):
    scale = angle_range[1] - angle_range[0]
    angle = torch.rand(batch_size) * scale + angle_range[0]

    s = rotation_y_mat(angle, batch_size)
    s_r = rotation_y_mat(-angle, batch_size)
    return s, s_r, angle.numpy()


def rotation_x_mat(theta, batch_size):
    return torch.stack([
        torch.stack([torch.ones(batch_size), torch.zeros(batch_size), torch.zeros(batch_size)], dim=0),
        torch.stack([torch.zeros(batch_size), torch.cos(theta), -torch.sin(theta)], dim=0),
        torch.stack([torch.zeros(batch_size), torch.sin(theta), torch.cos(theta)], dim=0)
    ], dim=0).cuda().permute(2, 0, 1).contiguous()


def rotation_2d_mat(theta):
    return torch.stack([
        torch.stack([torch.cos(theta), -torch.sin(theta)], dim=0),
        torch.stack([torch.sin(theta), torch.cos(theta)], dim=0)
    ], dim=0).cuda().permute(2, 0, 1).contiguous()


def make_rotation_2d_batch(batch_size, angle_range=(-0.25 * math.pi, 0.25 * math.pi)):
    scale = angle_range[1] - angle_range[0]
    angle = torch.rand(batch_size) * scale + angle_range[0]

    s = rotation_2d_mat(angle)
    s_r = rotation_2d_mat(-angle)
    return s, s_r, angle.numpy()


def make_rotation_X_batch(batch_size, angle_range=(-0.25 * math.pi, 0.25 * math.pi)):
    scale = angle_range[1] - angle_range[0]
    angle = torch.rand(batch_size) * scale + angle_range[0]

    s = rotation_x_mat(angle, batch_size)
    s_r = rotation_x_mat(-angle, batch_size)
    return s, s_r, angle.numpy()


def rotation_z_mat(theta, batch_size):
    return torch.stack([
        torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros(batch_size)], dim=0),
        torch.stack([torch.sin(theta), torch.cos(theta), torch.zeros(batch_size)], dim=0),
        torch.stack([torch.zeros(batch_size), torch.zeros(batch_size), torch.ones(batch_size)], dim=0)
    ], dim=0).cuda().permute(2, 0, 1).contiguous()


def make_rotation_Z_batch(batch_size, angle_range=(-0.25 * math.pi, 0.25 * math.pi)):
    scale = angle_range[1] - angle_range[0]
    angle = torch.rand(batch_size) * scale + angle_range[0]

    s = rotation_z_mat(angle, batch_size)
    s_r = rotation_z_mat(-angle, batch_size)
    return s, s_r, angle.numpy()


def make_random_rotation_batch(batch_size, angle_range=(-0.25 * math.pi, 0.25 * math.pi)):
    rota_func_list = [make_rotation_X_batch, make_rotation_Y_batch, make_rotation_Z_batch]
    rota_func = random.sample(rota_func_list, 1)[0]
    return rota_func(batch_size, angle_range)


def make_translation_batch(batch_size, dim=np.array([1, 0, 1]), is_std_normal=False, t_range=(-1, 1)):
    scale = t_range[1] - t_range[0]
    if is_std_normal:
        T_mat = torch.randn(batch_size, len(dim))
    else:
        T_mat = torch.rand(batch_size, len(dim)) * scale + t_range[0]
    T = T_mat.mul(torch.from_numpy(dim)).cuda()
    T_R = -T
    return T, T_R


def make_rand_zoom_batch(batch_size, dim=np.array([1, 0, 1]), z_range=(0.5, 2)):
    T_mat, TR_mat = make_translation_batch(batch_size, dim, is_std_normal=False, t_range=(z_range[0]-1, z_range[1]-1))
    Z_mat = T_mat + torch.ones(batch_size, len(dim)).cuda()
    ZR_mat = 1 / Z_mat
    return Z_mat, ZR_mat


# def make_rand_zoom_batch(batch_size, z_range=((0.3, 1.5), (1., 1.), (0.3, 1.5))):
#     zoom_range = torch.tensor(z_range)
#     zoomer_batch = []
#     scale = zoom_range[:, 1] - zoom_range[:, 0]
#     min_num = zoom_range[:, 0]
#     for i in range(batch_size):
#         rand_dim = torch.rand(zoom_range.size(0))
#         zoomer = rand_dim * scale + min_num
#         zoomer_batch.append(zoomer)
#     return torch.stack(zoomer_batch, dim=0).cuda(), 1 / torch.stack(zoomer_batch, dim=0).cuda()


def symm_trans(z, transer):
    return z + transer


def symm_rotate(z, rotator):
    z_R = torch.matmul(z.unsqueeze(1), rotator)
    return z_R.squeeze(1)


def symm_zoom(z, zoomer):
    return z * zoomer


def do_seq_symmetry(z, symm_func):
    z_seq_batch = []
    for i in range(z.size(1)):
        z_S_batch = symm_func(z[:, i])
        z_seq_batch.append(z_S_batch)
    return torch.stack(z_seq_batch, dim=0).permute(1, 0, 2).contiguous()


if __name__ == "__main__":
    Z_mat, ZR_mat = make_rand_zoom_batch(5, np.array([0,0,0,1]), z_range=(0.3, 1.5))
    print(Z_mat * ZR_mat)

    r1, r1r, delta1 = make_rotation_2d_batch(10)
    print(torch.matmul(r1, r1r))

    # T, Tr = make_translation_batch(32, t_range=[-3, 3])
    #
    # aaaaa= torch.tensor(((0.3, 1.5), (1., 1.), (0.3, 1.5)))
    # print(aaaaa)
    # zoom, zoom_R = make_rand_zoom_batch(2)
    # print(zoom, zoom_R)
    # z = torch.tensor([[1,2,3],[0,10,100]]).cuda()
    # zoomed = symm_zoom(z, zoom)
    # print(zoomed)
    # print(symm_zoom(zoomed, zoom_R))

    # t1, t_r1 = rand_translation(is_std_normal=False)
    # print(t1)
    # s1, s_r1, theta = rand_rotation_Y(angel_range=(0, 2 * math.pi))
    # print(f's1: {s1}, s_r1: {s_r1}, mul: {torch.matmul(s1, s_r1)}')
    # a = torch.from_numpy(np.array([
    #     [1.0, 2.0, 3.0]
    # ])).to(torch.float32)
    # a_t = a + t1
    # print(a_t)
    # print(a_t + t_r1)
    # b = torch.matmul(a, s1)
    # c = torch.matmul(b, s_r1)
    # print(b, c)
