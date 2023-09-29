import numpy as np

from torch_geometric.datasets import Planetoid, PPI, Reddit, Amazon, KarateClub
import torch_geometric.transforms as transforms


def get_coo_mat():
    dataset = Planetoid(root="tutorial1", name="Cora", transform=transforms.NormalizeFeatures())
    # dataset = KarateClub(transform=transforms.NormalizeFeatures())
    data = dataset[0]
    coo_data = data.edge_index.numpy()
    vtx_count = data.num_nodes
    print("Directed:", data.is_directed())
    print("Edge count: %d" % len(coo_data[0]))
    return coo_data, vtx_count


def create_coo_mat():
    mat = np.zeros((2, 14), dtype=int)
    mat[0, :] = [1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 7, 7]
    mat[1, :] = [2, 7, 4, 6, 2, 1, 6, 5, 4, 5, 1, 3, 4, 7]
    mat -= 1
    return mat, 7


def get_volume(d: dict):
    return sum(len(x) for x in d.values())


def calc_square(arr: list[dict]):
    ret = 0
    for d in arr:
        ret += np.sum(len(v) for v in d.values()) ** 2
    return ret


# used for testing purposes
def count_maps(maps: (int, dict)):
    return np.sum(len(v) for v in maps[1].values())


def get_uniform_mapping(cpu_cnt, vtx_count):
    return np.array([i % cpu_cnt for i in range(vtx_count)])


class DestData:
    def __init__(self, vid):
        self.dict = dict()
        self.volume = 0
        self.id = vid

    # calculate the cost from the dict itself rather than the volume variable
    def cost_raw(self):
        vol = sum(len(v) for v in self.dict.values())
        return vol ** 2

    def cost(self):
        return self.volume ** 2

    def insert(self, key, values):
        if key not in self.dict:
            self.dict[key] = set()
        first_len = len(self.dict[key])
        self.dict[key].update(values)
        self.dict[key].discard(self.id)
        self.volume += len(self.dict[key]) - first_len

    def delta_sqr(self, key, values):
        cntr = 0
        if key in self.dict:
            for v in values:
                if v not in self.dict[key] and v is not self.id:
                    cntr += 1
        else:
            cntr = len(values)
            if self.id in values:
                cntr -= 1
        return (self.volume + cntr) ** 2 - self.cost()

    def __str__(self):
        return self.dict.__str__()

    def __unicode__(self):
        return self.dict.__str__()

    def __repr__(self):
        return self.dict.__str__()


class dist_tracker:
    def __init__(self):
        self.other_idx = -1
        self.other_send_list = []
        self.send_list = []

    def set(self, other_idx: int, other_send_list: list, send_list: list):
        self.other_idx = other_idx
        self.other_send_list = other_send_list
        self.send_list = send_list

    def reset(self):
        self.other_send_list.clear()
        self.send_list.clear()
        self.other_idx = -1
