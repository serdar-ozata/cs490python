import math
import os
import time
from enum import Enum

import numpy as np
import pymetis
from scipy.io import mminfo, mmread, mmwrite

from torch_geometric.datasets import Planetoid, PPI, Reddit, Amazon, KarateClub, AmazonProducts, Yelp, Flickr
import torch_geometric.transforms as transforms

MIN_REASSIGN_LIMIT = 2


def cv(x):
    return np.std(x, ddof=1) / np.mean(x) * 100


def get_coo_mat(dataset_name):
    is_from_torch = True
    match dataset_name:
        case "Reddit":
            dataset = Reddit(root="data/Reddit")
        case "Amazon":
            dataset = Amazon(root="data/Amazon", name="Computers")
        case "AmazonProducts":
            dataset = AmazonProducts(root="data/AmazonProducts")
        case "KarateClub":
            dataset = KarateClub()
        case "Yelp":
            dataset = Yelp(root="data/Yelp")
        case "Flickr":
            dataset = Flickr(root="data/Flickr")
        case _:
            is_from_torch = False
            # read from mmdsets folder
            fpath = f"mmdsets/{dataset_name}.mtx"
            nr, nc, nnz, fmt, fld, sym = mminfo(fpath)
            mmfile = mmread(fpath)
            coo_data = mmfile.tocoo()
            # remove values from the matrix only row and col
            coo_data = np.array([coo_data.row, coo_data.col])
            vtx_count = nr

    if is_from_torch:
        data = dataset[0]
        coo_data = data.edge_index.numpy()
        vtx_count = data.num_nodes
        # write the matrix market file from coo_data
        mmfpath = f"mmdsets/{dataset_name}.mtx"
        if not os.path.exists(mmfpath):
            with open(mmfpath, "w") as f:
                f.write(f"%%MatrixMarket matrix coordinate real general\n")
                f.write(f"{vtx_count} {vtx_count} {len(coo_data[0])}\n")
                for i in range(len(coo_data[0])):
                    f.write(f"{coo_data[0][i] + 1} {coo_data[1][i] + 1} 1\n")

    adj = [[] for _ in range(vtx_count)]
    wg = np.zeros(dtype=int, shape=vtx_count)
    for i in range(len(coo_data[0])):
        adj[coo_data[0][i]].append(coo_data[1][i])
        wg[coo_data[0][i]] += 1
    return coo_data, vtx_count, adj, wg


def create_coo_mat():
    mat = np.zeros((2, 14), dtype=int)
    mat[0, :] = [1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 7, 7]
    mat[1, :] = [2, 7, 4, 6, 2, 1, 6, 5, 4, 5, 1, 3, 4, 7]
    mat -= 1
    return mat, 7


def get_volume(d: dict[list[int]]):
    return sum([len(x) for x in d.values()])


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
    partition: list[int, int] = []  # key: vertex, value: processor
    vtx_reqs: list[dict[int, set]] = []  # key: prc, vertex, value: set of vertices

    alpha = 0.0

    def __init__(self, vid, recv_vol=0):
        self.expands = dict()
        self.reassign_cores = dict()
        self.send_vol: int = 0
        self.id = vid
        self.recv_vol = recv_vol * DestData.alpha

    # returns tr, oet, ret send volumes
    def tr_oet_ret_vol(self) -> tuple[int, int, int]:
        tr_vol = np.sum([len(self.expands.get(k, [])) for k in self.reassign_cores.keys()], dtype=int)
        oet_vol: int = np.sum([len(v) if DestData.partition[k] == self.id else 0 for k, v in self.expands.items()],
                              dtype=int)
        oet_vol -= tr_vol
        return tr_vol, oet_vol, self.send_vol - oet_vol - tr_vol

    def volume(self):
        return self.send_vol + self.recv_vol

    def real_volume(self):
        if DestData.alpha == 0:
            raise Exception("Invalid alpha value!")
        return self.send_vol + self.recv_vol / DestData.alpha

    def decrease_recv_vol(self, val):
        self.recv_vol -= val * DestData.alpha

    # calculate the cost from the dict itself rather than the volume variable
    def send_cost_raw(self):
        vol = sum(len(v) for v in self.expands.values())
        return vol ** 2

    def cost(self):
        return self.volume() ** 2

    def insert(self, key, values):
        if key not in self.expands:
            self.expands[key] = set()
        first_len = len(self.expands[key])

        self.expands[key].update(values)
        self.expands[key].discard(self.id)
        self.send_vol += len(self.expands[key]) - first_len

    def set_forwarded_core(self, key, core_id):
        # if key in self.reassign_cores:
        #     raise Exception("Double forwarding attempt!")
        # if key not in self.reassign_cores:
        #     self.send_vol += 1
        self.reassign_cores[key] = core_id

    def remove_forwarded_core(self, key):
        ret = self.reassign_cores.pop(key, None)
        # if ret is not None:
        #     self.send_vol -= 1

    def sqr_contribution_of(self, key):
        return self.cost() - (self.volume() - len(self.expands.get(key, []))) ** 2

    def remove_assignment(self, key):
        e_len = len(self.expands.get(key, []))
        self.reassign_cores.pop(key, None)
        self.expands.pop(key, None)
        self.send_vol -= e_len
        return e_len

    def extract_assignment(self, key):
        data = self.expands[key]
        e_len = len(data)
        self.expands.pop(key)
        self.send_vol -= e_len
        return data

    def delta_sqr(self, key, values):
        if isinstance(values, int):
            cntr = values
        else:
            cntr = len(values)
            if self.id in values:
                cntr -= 1
        return (self.volume() + cntr) ** 2 - self.cost()

    # returns whether the key (vertex, row) is owned by the processor
    def is_owner(self, key: int):
        return DestData.partition[key] == self.id

    def is_tr(self, key: int):
        return key in self.reassign_cores

    # assumes the task for this key exists
    def is_oet(self, key: int):
        return self.is_owner(key) and not self.is_tr(key)

    def __str__(self):
        return self.expands.__str__()

    def __unicode__(self):
        return self.expands.__str__()

    def __repr__(self):
        return self.expands.__str__()


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


class Assigment(Enum):
    N_MINUS_1_TO_1 = 1
    HALF_SPLIT = 2
    VOL_EQ_SPLIT = 3
    NO_REASSIGNMENT = 0


class PartitionType(Enum):
    LOWEST_VOLUME = 1
    SUBSET_SUM = 2


def assignment_cost_test(a, b, set_size):
    diff = a - b
    if math.fabs(diff) < set_size:
        if diff > 0:
            other_vol = int(math.ceil((set_size - diff) / 2))
            owner_vol = int(math.floor((set_size + diff) / 2))
        else:
            other_vol = int(math.floor((set_size - diff) / 2))
            owner_vol = int(math.ceil((set_size + diff) / 2))
    else:
        other_vol = set_size - 1
        owner_vol = 1
    print(f"other: {other_vol}, owner: {owner_vol}")


def get_vol_eq_split_vols(a, b, set_size):
    diff = a - b
    other_vol = int(math.ceil((set_size - diff) / 2))
    owner_vol = int(math.floor((set_size + diff) / 2))
    if owner_vol == 0:
        owner_vol += 1
        other_vol -= 1
    elif other_vol == 0:
        owner_vol -= 1
        other_vol += 1
    return other_vol, owner_vol


def assignment_cost(other_data: DestData, core_data: DestData, key: int, send_set: set):
    diff = other_data.send_vol - core_data.send_vol
    set_size = len(send_set)
    if np.abs(diff) < set_size:
        other_vol, owner_vol = get_vol_eq_split_vols(other_data.send_vol, core_data.send_vol, set_size)
        return other_data.delta_sqr(key, other_vol) + core_data.delta_sqr(key, owner_vol), Assigment.VOL_EQ_SPLIT
    else:
        return other_data.delta_sqr(key, send_set) + core_data.delta_sqr(key, 1), Assigment.N_MINUS_1_TO_1


class MetricTracker:
    def __init__(self):
        self.cost = 0
        self.reassign_cnt = 0
        self.non_reassign_cnt = 0
        self.reassign_vol = 0
        self.non_reassignment_vol = 0

    def set_vol_eq_split(self, o_vol, c_vol):
        self.reassign_cnt += 1
        self.non_reassignment_vol += c_vol
        self.reassign_vol += o_vol

    def set(self, atype: Assigment, set_size: int):
        match atype:
            case Assigment.NO_REASSIGNMENT:
                self.non_reassign_cnt += 1
                self.non_reassignment_vol += set_size
            case Assigment.N_MINUS_1_TO_1:
                self.reassign_cnt += 1
                self.non_reassignment_vol += 1
                self.reassign_vol += set_size - 1
            case _:
                raise Exception("This assignment type is not supported")


class VolInit(Enum):
    EMPTY = 0
    METIS = 1


def get_opt_send_list(send_list: list[dict], alpha: float, noconstructive: bool, t: MetricTracker):
    proc_cnt = len(send_list)
    DestData.alpha = alpha
    if alpha > 0:
        recv_vols = np.zeros(dtype=int, shape=proc_cnt)
        for i in range(proc_cnt):
            d = send_list[i]
            for k, v in d.items():
                for idx in v:
                    recv_vols[idx] += 1
        ret = [DestData(v, recv_vols[v]) for v in range(proc_cnt)]
    else:
        ret = [DestData(v) for v in range(proc_cnt)]

    if noconstructive:
        for data in ret:
            d = send_list[data.id]
            for k, v in d.items():
                t.set(Assigment.NO_REASSIGNMENT, len(v))
                data.insert(k, v)
    return ret


def get_sorted_degree_list(send_list: list[dict[int, set]], reverse=True):
    e_degrees = [(len(arr), idx, k) for idx in range(len(send_list)) for k, arr in send_list[idx].items()]
    e_degrees.sort(key=lambda i: i[0], reverse=reverse)
    if reverse:
        it = range(len(e_degrees) - 1, -1, -1)
        increment = 1
        lowest_val = len(e_degrees) - 1
    else:
        it = range(len(e_degrees))
        increment = -1
        lowest_val = 0
    for i in it:
        if MIN_REASSIGN_LIMIT < e_degrees[i][0]:
            if lowest_val == i:
                return e_degrees
            else:
                i += increment
                return [*e_degrees[i:], *e_degrees[0:i]]
    return e_degrees


def get_node_range(core_idx: int, core_cnt: int, node_core_cnt):
    idx = np.floor(core_idx / node_core_cnt)
    start = idx * node_core_cnt
    end = start + node_core_cnt
    if end > core_cnt:
        end = core_cnt - 1
    return int(start), int(end)


def mapping_exists(core_cnt, name):
    return os.path.exists(f"mmdsets/schemes/{name}.inpart.{core_cnt}")


def get_mapping(core_cnt, name):
    with open(f"mmdsets/schemes/{name}.inpart.{core_cnt}") as f:
        return [int(x) for x in f.read().split()]


def gen_send_list(core_cnt, name, adj_mat, coo_data, wg):
    # generate a random ownership list
    # check whether mapping exists
    mexists = mapping_exists(core_cnt, name)
    if mexists:
        mappings = get_mapping(core_cnt, name)
    else:
        mappings = pymetis.part_graph(core_cnt, adjacency=adj_mat, vweights=wg)[1]

    send_list: list[dict[set]] = [dict() for _ in range(core_cnt)]  # keys are vtxs, values are sets of receiver prcrs
    vtx_reqs: list[dict[set]] = [dict() for _ in range(core_cnt)]  # keys are vtxs, values are sets of sender vtxs
    DestData.vtx_reqs = vtx_reqs
    DestData.partition = mappings
    DestData.recv_vtxs = [set() for _ in range(core_cnt)]
    # parse the data
    for i in range(len(coo_data[0])):
        v_i = coo_data[0, i]
        v_j = coo_data[1, i]
        sender_idx = mappings[v_i]
        rec_idx = mappings[coo_data[1, i]]
        if rec_idx == sender_idx:
            continue
        if v_i not in send_list[sender_idx]:
            send_list[sender_idx][v_i] = set()
        send_list[sender_idx][v_i].add(rec_idx)
        if v_j not in vtx_reqs[rec_idx]:
            vtx_reqs[rec_idx][v_j] = set()
        vtx_reqs[rec_idx][v_j].add(v_i)
        DestData.recv_vtxs[rec_idx].add(v_i)
    # save mapping
    if not mexists:
        with open(f"mmdsets/schemes/{name}.inpart.{core_cnt}", "w") as f:
            f.write("\n".join([str(x) for x in mappings]))

    return send_list


def parse_processor_based_lists(send_list, core_cnt):
    recv_lists = [dict() for _ in range(core_cnt)]
    send_lists = [dict() for _ in range(core_cnt)]
    # fill the send & recv lists where the keys are processors and values are vertexes
    for send_idx in range(core_cnt):
        for vtx, rec_idxs in send_list[send_idx].items():
            for rec_idx in rec_idxs:
                if send_idx not in recv_lists[rec_idx]:
                    recv_lists[rec_idx][send_idx] = []
                recv_lists[rec_idx][send_idx].append(vtx)
                if rec_idx not in send_lists[send_idx]:
                    send_lists[send_idx][rec_idx] = []
                send_lists[send_idx][rec_idx].append(vtx)
    return send_lists, recv_lists
