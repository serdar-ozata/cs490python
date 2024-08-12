import numpy as np
import os
import pymetis
import random
from util import DestData, write_bin_file


# writes vertex mappings to binary file
def write_partitions(mappings: list, fname: str):
    with open(fname, 'wb') as file:
        write_bin_file(file, mappings)


def mapping_exists(core_cnt, name):
    return os.path.exists(f"mmdsets/schemes/{name}.inpart.{core_cnt}")


def get_mapping(core_cnt, name):
    with open(f"mmdsets/schemes/{name}.inpart.{core_cnt}", "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    return data.tolist()


def gen_send_list(core_cnt, name, adj_mat, coo_data, wg):
    # generate a random ownership list
    # check whether mapping exists
    mexists = mapping_exists(core_cnt, name)
    if mexists:
        mappings = get_mapping(core_cnt, name)
    else:
        opt = pymetis.Options()
        opt.set_defaults()
        opt.__setattr__("seed", random.randint(0, 1000))
        mappings = pymetis.part_graph(core_cnt, adjacency=adj_mat, vweights=wg)[1]

    send_list: list[dict[set]] = [dict() for _ in range(core_cnt)]  # keys are vtxs, values are sets of receiver prcrs
    vtx_reqs: list[dict[set]] = [dict() for _ in range(core_cnt)]  # keys are vtxs, values are sets of sender vtxs
    DestData.vtx_edges = vtx_reqs
    DestData.partition = mappings
    DestData.recv_vtxs = [set() for _ in range(core_cnt)]
    added_rev_edge = 0
    added_edge = 0
    # parse the data
    print(f"Edge count: {len(coo_data[0])}")
    for i in range(len(coo_data[0])):
        v_i = coo_data[0, i]
        v_j = coo_data[1, i]
        sender_idx = mappings[v_i]
        rec_idx = mappings[v_j]
        if rec_idx == sender_idx:
            DestData.local_edges.append((v_i, v_j))
            continue
        if v_i not in send_list[sender_idx]:
            send_list[sender_idx][v_i] = set()
        if rec_idx not in send_list[sender_idx][v_i]:
            send_list[sender_idx][v_i].add(rec_idx)
            added_edge += 1
        if v_j not in vtx_reqs[rec_idx]:
            vtx_reqs[rec_idx][v_j] = set()
        vtx_reqs[rec_idx][v_j].add(v_i)
        DestData.recv_vtxs[rec_idx].add(v_i)
        # add reverse edge
        # if v_j not in send_list[rec_idx]:
        #     send_list[rec_idx][v_j] = set()
        # if sender_idx not in send_list[rec_idx][v_j]:
        #     added_rev_edge += 1
        #     send_list[rec_idx][v_j].add(sender_idx)
        # if v_i not in vtx_reqs[sender_idx]:
        #     vtx_reqs[sender_idx][v_i] = set()
        # vtx_reqs[sender_idx][v_i].add(v_j)
        # DestData.recv_vtxs[sender_idx].add(v_j)
    total_vol = sum([sum([len(v) for v in d.values()]) for d in send_list])
    print(f"Added {added_rev_edge} reverse edges, {added_edge} edges.")
    print(f"Total volume: {total_vol}")
    # save mapping
    if not mexists:
        write_partitions(mappings, f"mmdsets/schemes/{name}.inpart.{core_cnt}")

    return send_list
