import os
import glob
import numpy as np
import os.path as osp
from load_off import read_off
from tf_sampling import triangles_to_pc
from tqdm import tqdm

modelnet40_data_root = '/repository/ModelNet40'
pc_data_root = '/repository/pc'


def check_dir(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    shape_all = glob.glob(osp.join(modelnet40_data_root, '*', '*', '*.off'))
    for shape_dir in tqdm(shape_all):
        pc_dir = shape_dir.replace(modelnet40_data_root, pc_data_root)
        pc_dir = osp.splitext(pc_dir)[0] + '.npy'
        pc_folder = osp.split(pc_dir)[0]

        if osp.exists(pc_dir):
            continue

        check_dir(pc_folder)
        triangles = read_off(shape_dir)
        ret = triangles_to_pc(triangles)
        np.save(pc_dir, ret)
