import numpy as np
from torch_geometric.datasets import Planetoid, PPI, Reddit, Amazon, KarateClub
import torch_geometric.transforms as transforms
import util

cpu_cnt = 40
# dataset = Planetoid(root="tutorial1", name="Cora", transform=transforms.NormalizeFeatures())
dataset = KarateClub(transform=transforms.NormalizeFeatures())
data = dataset[0]
coo_data = data.edge_index.numpy()
vtx_count = data.num_nodes
print("Directed:", data.is_directed())
print("Vertex Count:", vtx_count)
# generate a random ownership list
np.random.seed(0)  # set seed
mappings = np.random.randint(0, cpu_cnt, vtx_count)

# (int, set<int>)[] where int is cpu idxs; set's values are the dest cpu idxs
send_list = [(i, set()) for i in range(cpu_cnt)]
rec_list = [(i, set()) for i in range(cpu_cnt)]  # todo I don't need this for now
# fill send & rec lists
for i in range(len(coo_data[0])):
    sender_idx = mappings[coo_data[0, i]]
    rec_idx = mappings[coo_data[1, i]]
    if rec_idx == sender_idx:
        continue
    send_list[sender_idx][1].add(rec_idx)

# sort in decreasing order
# send_list.sort(key=lambda x: sum(len(v) for v in x[1].values()), reverse=True)
send_list.sort(key=lambda x: len(x[1]), reverse=True)

opt_send_list = [set() for _ in range(cpu_cnt)]  # this type of holding send_list can't be used for detecting
# deadlocks unfortunately since it doesn't hold which transmissions should be waited before sending
pcost = 0
best_dist = util.dist_tracker()


def delta_sqr(idx, diff) -> int:
    return np.square(len(opt_send_list[idx]) + diff) - np.square(len(opt_send_list[idx]))


# distribution algorithm
for i in range(cpu_cnt):
    cpu_idx = send_list[i][0]
    send_set = send_list[i][1]
    best_dist.reset()
    for n in send_set:
        if n not in opt_send_list[cpu_idx]:
            best_dist.send_list.append(n)
    best_cost = util.calc_square_wt(opt_send_list, best_dist, cpu_idx)

    for other_idx in range(len(send_set)):
        ts_cnt = 0 if other_idx in opt_send_list[cpu_idx] else 1
        other_ts_cnt = 0
        for n in send_set:
            if n == other_idx:
                continue
            if n not in opt_send_list[other_idx]:
                other_ts_cnt += 1
        cur_cost = pcost + delta_sqr(cpu_idx, ts_cnt) + delta_sqr(other_idx, other_ts_cnt)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_dist.reset()
            best_dist.other_idx = other_idx
            if ts_cnt > 0:
                best_dist.send_list.append(other_idx)
            for n in send_set:
                if n == other_idx:
                    continue
                if n not in opt_send_list[other_idx]:
                    best_dist.other_send_list.append(n)
    pcost = best_cost
    if best_dist.other_idx > -1:
        opt_send_list[best_dist.other_idx].update(best_dist.other_send_list)
    opt_send_list[cpu_idx].update(best_dist.send_list)

print("Min Square: %d" % pcost)
print("Initial Square %d" % (util.calc_square([x[1] for x in send_list])))
print("Initial Distribution:")
print(send_list)
print("Optimized Distribiton (Indexed):")
print(opt_send_list)
