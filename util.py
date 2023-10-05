import numpy as np

from torch_geometric.datasets import Planetoid, PPI, Reddit, Amazon, KarateClub
import torch_geometric.transforms as transforms
import torch_geometric.utils as tl
import pymetis
from scipy.sparse import coo_matrix
import sys
import matplotlib.pyplot as plt


def get_coo_mat():
    dataset = Reddit(root="tutorial1", transform=transforms.NormalizeFeatures())
    # dataset = KarateClub(transform=transforms.NormalizeFeatures())
    data = dataset[0]
    coo_data = data.edge_index.numpy()
    vtx_count = data.num_nodes
    print("Directed:", data.is_directed())
    mappings = [[] for _ in range(vtx_count)]
    for i in range(len(coo_data[0])):
        mappings[coo_data[0][i]].append(coo_data[1][i])
    return coo_data, vtx_count, mappings


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


def write_results(arr):
    sys.stdout = open(f"out/reddit_{arr[4]}cores.txt", "w")
    print("Vertex Count:", arr[0])
    print("Edge count: %d" % arr[1])
    print(f"Algorithm execution time: {arr[2]} seconds")
    print(f"Algorithm and data parsing execution time: {arr[3]} seconds")
    print("Core Count %d" % arr[4])
    print("Number of extend operations: %d" % arr[5])
    print("Number of non-extended operations: %d" % arr[6])
    print("Min Sum Square: %d, Initial: %d" % (arr[7], arr[8]))
    print("Min Sum Square without delta: %d" % arr[9])
    print("Highest Volume %d, Initial: %d" % (arr[10], arr[11]))
    sys.stdout.close()


def save_basic_plot(x, y1, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.plot(x, y1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(f"out/{title}.png")


def save_2basic_plot(x, y1, y2, xlabel, ylabel, title, y1name, y2name, yscale=None):
    fig, ax = plt.subplots()
    ax.plot(x, y1, label=y1name)
    ax.plot(x, y2, label=y2name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend()
    plt.savefig(f"out/{title}.png")
