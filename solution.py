import math
import random
import time
import numpy as np
import pymetis
import util
import argparse
from tqdm import tqdm
from util import Assigment, MetricTracker, DestData, MIN_REASSIGN_LIMIT, get_opt_send_list
from binout import partition_phases, write_partitions
from metricsout import create_excel

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('core_cnt', metavar='K', type=int, help='core count')
parser.add_argument("-l", "--loop", type=int, help='end core count (excluded)')

group = parser.add_mutually_exclusive_group()
group.add_argument("-i", "--interval", type=int, help='core count increment', default=1)
group.add_argument("-e", "--exp", type=int, help='multiply core count by n in each iteration')
parser.add_argument('-v', '--volumemode', type=int, help='0: Empty (def), 1: Receive Vol, 2: METIS, 3: Rec + METIS',
                    default=0)
parser.add_argument('-d', '--datasets', type=str, nargs='+', help='Dataset names')

parser.add_argument('--noconstructive', action='store_true')
parser.add_argument('--noiterative', action='store_true')

args = parser.parse_args()

dec_order = True

if args.noconstructive and args.noiterative:
    raise Exception("At least one of the algorithms must be enabled!")

if args.noconstructive and args.volumemode < 2:
    raise Exception(
        "Either METIS (2) or Rec + METIS (3) --volumemode must be chosen if --noconstructive flag is enabled")

if args.datasets is None:
    raise Exception("At least one dataset must be specified")
print(args)


def get_core_iterator():
    if args.exp is None:
        return range(args.core_cnt, args.loop, args.interval)
    else:
        ratio = math.log(args.loop / args.core_cnt, args.exp)
        iter_count = int(np.ceil(ratio))
        return [(args.exp ** i) * args.core_cnt for i in range(iter_count)]


def execute(core_cnt):
    # generate a random ownership list
    # mappings = pymetis.part_graph(core_cnt, adjacency=adj_mat)[1]
    mappings = pymetis.part_graph(core_cnt, adjacency=adj_mat, vweights=wg)[1]

    send_list = [dict() for i in range(core_cnt)]
    DestData.partition = mappings
    parse_start_time = time.perf_counter()

    # parse the data
    for i in range(len(coo_data[0])):
        sender_idx = mappings[coo_data[0, i]]
        rec_idx = mappings[coo_data[1, i]]
        v_i = coo_data[0, i]
        v_j = coo_data[1, i]
        if rec_idx == sender_idx:
            continue
        if v_i not in send_list[sender_idx]:
            send_list[sender_idx][v_i] = set()
        send_list[sender_idx][v_i].add(rec_idx)

    # metric tracker
    t = MetricTracker()

    # sort degrees in decreasing order
    degree_list = util.get_sorted_degree_list(send_list)

    # send list that will be optimized by the algorithms
    opt_send_list = get_opt_send_list(send_list, args.volumemode, t)

    # execute the algorithms
    start_time = time.perf_counter()

    if not args.noconstructive:
        constructive_algorithm(opt_send_list, send_list, t, degree_list)

    if not args.noiterative:
        iterative_improvement(opt_send_list, send_list, t, degree_list)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    p_execution_time = end_time - parse_start_time

    # communication partition
    partition_phases(opt_send_list, core_cnt, name)

    # write partitions as file
    write_partitions(mappings, core_cnt, name)

    vols = [util.get_volume(send_list[opt_d.id]) + opt_d.recv_vol for opt_d in opt_send_list]
    e_degrees = [len(arr) for s in send_list for arr in s.values()]
    e_degrees.sort()
    init_cv = util.cv(vols)
    init_cost = np.sum([x ** 2 for x in vols])

    def extract_results():
        opt_vols = [x.volume() for x in opt_send_list]
        opt_cv = util.cv(opt_vols)
        # 0: vertex cnt, 1: edge count, 2: algo execution_time, 3: parse and algo execution_time, 4: core_cnt,
        # 5: t.reassign_cnt, 6: t.non_reassign_cnt, 7: reassign volume, 8: square sum / initial square sum,
        # 9: avg send vol per processor, 10: i vol cv, 11: opt  vol cv
        # 12: I highest volume, 13: highest volume, 14: I min vol, 15: min vol
        # 16: expand task count, 17: expand degree avg, 18: expand degree sum
        # 19: max expand degree, 20: min expand degree, 21: avg ed, 22: cv ed, 23: highest 10% / total
        # 22: sender's avg expand task size after reassignment

        execution_res = [vtx_count, len(coo_data[0]), execution_time, p_execution_time, core_cnt, t.reassign_cnt,
                         t.non_reassign_cnt, t.reassign_vol, init_cost / t.cost,  # 8
                         np.sum(vols) / core_cnt, init_cv, opt_cv, np.max(vols), np.max(opt_vols), np.min(vols),
                         np.min(opt_vols),
                         degree_len, degree_sum / degree_len, degree_sum, e_degrees[-1], e_degrees[0],
                         util.cv(e_degrees),
                         np.sum(e_degrees[degree_len - twenty_perc - 1:]) / degree_sum,
                         # t.non_reassignment_vol / t.reassign_cnt
                         ]
        execution_res = np.around(execution_res, 2)
        return execution_res

    degree_sum = np.sum(e_degrees)
    degree_len = len(e_degrees)
    twenty_perc = int(math.floor(0.1 * degree_len))

    return extract_results()


def constructive_algorithm(opt_send_list: list[DestData], send_list: list[dict], t: MetricTracker,
                           degree_list: list[tuple[int, int, int]]):
    # distribution algorithm
    for size, core_idx, v_idx in degree_list:
        send_dict = send_list[core_idx]
        send_set = send_dict[v_idx]
        core_dest_data = opt_send_list[core_idx]
        assign(core_dest_data, v_idx, send_set, opt_send_list, t)


# iterative improvement algorithm
def iterative_improvement(opt_send_list: list[DestData], send_list: list[dict],
                          t: MetricTracker, degree_list: list[tuple[int, int, int]]):
    random.shuffle(degree_list)
    for _, processor_idx, vtx in degree_list:
        send_dict = send_list[processor_idx]
        core_dest_data = opt_send_list[processor_idx]
        send_set = send_dict[vtx]

        # check if it was reassigned
        reassigned = vtx in core_dest_data.reassign_cores
        del_cost = core_dest_data.sqr_contribution_of(vtx)
        if reassigned:
            assigned_core = core_dest_data.reassign_cores[vtx]
            assigned_core_dest_data = opt_send_list[assigned_core]
            del_cost += assigned_core_dest_data.sqr_contribution_of(vtx)
            t.reassign_vol -= assigned_core_dest_data.remove_assignment(vtx)
            t.reassign_cnt -= 1
        else:
            t.non_reassign_cnt -= 1

        t.non_reassignment_vol -= core_dest_data.remove_assignment(vtx)
        t.cost -= del_cost
        assign(core_dest_data, vtx, send_set, opt_send_list, t)


def apply_vol_equalizing_split(opt_send_list: list[DestData], expand_core: int, core_dest_data: DestData, h: int,
                               send_set: set):
    other_dest_data = opt_send_list[expand_core]
    core_dest_data.set_forwarded_core(h, expand_core)
    set_size = len(send_set)
    other_vol, core_vol = util.get_vol_eq_split_vols(other_dest_data.send_vol, core_dest_data.send_vol, set_size)

    send_set.remove(expand_core)
    set_list = list(send_set)
    set_list.insert(0, expand_core)

    core_dest_data.insert(h, set_list[0:core_vol])
    other_dest_data.insert(h, set_list[core_vol:])
    send_set.add(expand_core)
    return other_vol, core_vol


def apply_n_minus_1_to_1_split(core_dest_data: DestData, h: int, send_set: set, expand_core: int,
                               opt_send_list: list[DestData]):
    core_dest_data.set_forwarded_core(h, expand_core)
    # core_dest_data.insert(h, [expand_core])
    opt_send_list[expand_core].insert(h, send_set)


# main part of the algorithm that finds and applies the best assignment (or non-assignment) for the given vertex
def assign(core_dest_data: DestData, vtx: int, send_set: set, opt_send_list: list[DestData], t: MetricTracker):
    best_del_cost = core_dest_data.delta_sqr(vtx, send_set)
    best_assignment_type = Assigment.NO_REASSIGNMENT
    expand_core = -1  # means self
    if len(send_set) >= MIN_REASSIGN_LIMIT:
        # square_sum_cost
        #     for other_idx in send_set:
        #         other_dest_data = opt_send_list[other_idx]
        #         delta_cost, assignment_type = util.assignment_cost(other_dest_data, core_dest_data, h, send_set)
        #         if delta_cost < best_del_cost:
        #             best_assignment_type = assignment_type
        #             best_del_cost = delta_cost
        #             expand_core = other_idx
        # lowest vol cost
        other_idx = min(send_set, key=lambda i: opt_send_list[i].volume())
        other_dest_data = opt_send_list[other_idx]
        delta_cost, assignment_type = util.assignment_cost(other_dest_data, core_dest_data, vtx, send_set)
        if delta_cost < best_del_cost:
            best_assignment_type = assignment_type
            best_del_cost = delta_cost
            expand_core = other_idx
    t.cost += best_del_cost
    match best_assignment_type:
        case Assigment.NO_REASSIGNMENT:
            core_dest_data.remove_forwarded_core(vtx)
            t.set(best_assignment_type, len(send_set))
            core_dest_data.insert(vtx, send_set)
        case Assigment.N_MINUS_1_TO_1:
            t.set(best_assignment_type, len(send_set))
            apply_n_minus_1_to_1_split(core_dest_data, vtx, send_set, expand_core, opt_send_list)
        case Assigment.VOL_EQ_SPLIT:
            other_vol, core_vol = apply_vol_equalizing_split(opt_send_list, expand_core, core_dest_data, vtx, send_set)
            t.set_vol_eq_split(other_vol, core_vol)


if len(args.datasets) == 1 and args.loop is None and args.exp is None:
    name = args.datasets[0]
    coo_data, vtx_count, adj_mat, wg = util.get_coo_mat(name)
    r = execute(args.core_cnt)
else:
    results = []
    for name in tqdm(args.datasets):
        coo_data, vtx_count, adj_mat, wg = util.get_coo_mat(name)
        print(str(name) + ": " + str(vtx_count))
        iterator = get_core_iterator()
        for i in iterator:
            r = execute(i)
            # r.insert(0, name)
            results.append(r)
        del adj_mat
    create_excel(args, results, args.datasets)
