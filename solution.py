import math
import random
import statistics
import time

import numpy as np
import pymetis
import util
import argparse
from tqdm import tqdm
from util import Assigment, MetricTracker, DestData

MIN_REASSIGN_LIMIT = 4

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('core_cnt', metavar='K', type=int, help='core count')
parser.add_argument("-l", "--loop", type=int, help='end core count (excluded)')

group = parser.add_mutually_exclusive_group()
group.add_argument("-i", "--interval", type=int, help='core count increment', default=1)
group.add_argument("-e", "--exp", type=int, help='multiply core count by n in each iteration')
parser.add_argument('-d', '--datasets', type=str, nargs='+', help='Dataset names')

args = parser.parse_args()

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
    mappings = pymetis.part_graph(core_cnt, adjacency=adj_mat)[1]

    send_list = [(i, dict()) for i in range(core_cnt)]

    parse_start_time = time.perf_counter()

    # for v_i in range(len(adj_mat)):
    #     for v_j in adj_mat[v_i]:
    #         sender_idx = mappings[v_i]
    #         rec_idx = mappings[v_j]
    #         if rec_idx == sender_idx:
    #             continue
    #         if v_i not in send_list[sender_idx][1]:
    #             send_list[sender_idx][1][v_i] = set()
    #         send_list[sender_idx][1][v_i].add(rec_idx)
    for i in range(len(coo_data[0])):
        sender_idx = mappings[coo_data[0, i]]
        rec_idx = mappings[coo_data[1, i]]
        v_i = coo_data[0, i]
        v_j = coo_data[1, i]
        if rec_idx == sender_idx:
            continue
        if v_i not in send_list[sender_idx][1]:
            send_list[sender_idx][1][v_i] = set()
        send_list[sender_idx][1][v_i].add(rec_idx)

    # sort in decreasing order
    # send_list.sort(key=lambda x: sum(len(v) for v in x[1].values()), reverse=True)
    send_list.sort(key=util.count_maps, reverse=True)
    opt_send_list = [DestData(v) for v in range(core_cnt)]
    t = MetricTracker()
    start_time = time.perf_counter()
    # distribution algorithm
    for i in range(core_cnt):
        cpu_idx = send_list[i][0]
        send_dict = send_list[i][1]
        for h, send_set in send_dict.items():
            core_dest_data = opt_send_list[cpu_idx]
            assign(core_dest_data, h, send_set, opt_send_list, t)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    p_execution_time = end_time - parse_start_time

    vols = [util.get_volume(x[1]) for x in send_list]
    e_degrees = [len(arr) for s in send_list for arr in s[1].values()]
    e_degrees.sort()
    init_cv = util.cv(vols)
    init_cost = np.sum([x ** 2 for x in vols])

    iterative_improvement(core_cnt, opt_send_list, send_list, t)

    def extract_results():
        opt_vols = [x.volume for x in opt_send_list]
        opt_cv = util.cv(opt_vols)
        # 0: vertex cnt, 1: edge count, 2: algo execution_time, 3: parse and algo execution_time, 4: core_cnt,
        # 5: t.reassign_cnt, 6: t.non_reassign_cnt, 7: reassign volume, 8: square sum / initial square sum,
        # 9: highest volume / initial highest volume,
        # 10: opt/init cv of volume between cores, 11: avg send vol per processor, 12: min vol, 13: initial min vol
        # 14: expand task count, 15: expand degree avg, 16: expand degree sum
        # 17: max expand degree, 18: min expand degree, 19: avg ed, 20: cv ed, 21: highest 10% / total
        # 22: sender's avg expand task size after reassignment

        execution_res = [vtx_count, len(coo_data[0]), execution_time, p_execution_time, core_cnt, t.reassign_cnt,
                         t.non_reassign_cnt, t.reassign_vol, init_cost / t.cost,  # 8
                         np.max(vols) / np.max(opt_vols), init_cv / opt_cv,
                         np.sum(vols) / core_cnt, init_cv, opt_cv, np.min(vols), np.min(opt_vols),
                         degree_len, degree_sum / degree_len, degree_sum, e_degrees[-1], e_degrees[0],
                         util.cv(e_degrees),
                         np.sum(e_degrees[degree_len - twenty_perc - 1:]) / degree_sum,
                         t.non_reassignment_vol / t.reassign_cnt
                         ]
        execution_res = np.around(execution_res, 2)
        return execution_res

    degree_sum = np.sum(e_degrees)
    degree_len = len(e_degrees)
    twenty_perc = int(math.floor(0.1 * degree_len))

    return extract_results()


# iterative improvement algorithm
def iterative_improvement(core_cnt: int, opt_send_list: list[DestData], send_list: list[tuple[int, dict]],
                          t: MetricTracker):
    random.shuffle(send_list)
    for i in range(core_cnt):
        cpu_idx = send_list[i][0]
        send_dict = send_list[i][1]
        core_dest_data = opt_send_list[cpu_idx]
        for (h, send_set) in send_dict.items():
            # check if it was reassigned
            reassigned = h in core_dest_data.reassign_cores
            del_cost = core_dest_data.sqr_contribution_of(h)
            if reassigned:
                assigned_core = core_dest_data.reassign_cores[h]
                assigned_core_dest_data = opt_send_list[assigned_core]
                del_cost += assigned_core_dest_data.sqr_contribution_of(h)
                t.reassign_vol -= assigned_core_dest_data.remove_assignment(h)
                t.reassign_cnt -= 1
            else:
                t.non_reassign_cnt -= 1

            t.non_reassignment_vol -= core_dest_data.remove_assignment(h)
            prev_cost = t.cost
            t.cost -= del_cost
            assign(core_dest_data, h, send_set, opt_send_list, t)
            # if t.cost > prev_cost:
            #     print(f"c: {t.cost}, prev: {prev_cost}")


def apply_vol_equalizing_split(opt_send_list: list[DestData], expand_core: int, core_dest_data: DestData, h: int,
                               send_set: set):
    other_dest_data = opt_send_list[expand_core]
    core_dest_data.set_forwarded_core(h, expand_core)
    set_size = len(send_set)
    other_vol, core_vol = util.get_vol_eq_split_vols(other_dest_data.volume, core_dest_data.volume, set_size)
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
    core_dest_data.insert(h, [expand_core])
    opt_send_list[expand_core].insert(h, send_set)


# main part of the algorithm that finds and applies the best assignment (or non-assignment)
def assign(core_dest_data: DestData, h: int, send_set: set, opt_send_list: list[DestData], t: MetricTracker):
    best_del_cost = core_dest_data.delta_sqr(h, send_set)
    best_assignment_type = Assigment.NO_REASSIGNMENT
    expand_core = -1  # means self
    if len(send_set) >= MIN_REASSIGN_LIMIT:
        for other_idx in send_set:
            other_dest_data = opt_send_list[other_idx]
            delta_cost, assignment_type = util.assignment_cost(other_dest_data, core_dest_data, h, send_set)
            if delta_cost < best_del_cost:
                best_assignment_type = assignment_type
                best_del_cost = delta_cost
                expand_core = other_idx
    t.cost += best_del_cost
    match best_assignment_type:
        case Assigment.NO_REASSIGNMENT:
            t.set(best_assignment_type, len(send_set))
            core_dest_data.insert(h, send_set)
        case Assigment.N_MINUS_1_TO_1:
            t.set(best_assignment_type, len(send_set))
            apply_n_minus_1_to_1_split(core_dest_data, h, send_set, expand_core, opt_send_list)
        case Assigment.VOL_EQ_SPLIT:
            other_vol, core_vol = apply_vol_equalizing_split(opt_send_list, expand_core, core_dest_data, h, send_set)
            t.set_vol_eq_split(other_vol, core_vol)


if args.datasets is None:
    edge_cnt, vtx_count, adj_mat = util.get_coo_mat()
    if args.loop is None:
        res = execute(args.core_cnt)
        util.write_results(res)
    else:
        iterator = get_core_iterator()
        results = []
        for i in tqdm(iterator):
            results.append(execute(i))

        results = np.array(results).transpose()
        util.save_2basic_plot(results[4], results[1], results[2], "Core Count", "Time (s)",
                              "Execution Times", "Algorithm", "Algorithm & Parsing", "linear")
        util.save_2basic_plot(results[4], results[5], results[6], "Core Count", None,
                              "Expanded & Non-Expanded Operations", "Extended", "Non-Extended")
        util.save_2basic_plot(results[4], results[7], results[8], "Core Count", None,
                              "Square Sum of Volumes", "Optimized", "Initial")
        util.save_2basic_plot(results[4], results[10], results[11], "Core Count", None,
                              "Highest Volumes", "Optimized", "Initial")
        util.save_basic_plot(results[4], results[12], "Core Count", "Average", "Average send volume")
        util.save_2basic_plot(results[4], results[13], results[14], "Core Count", None,
                              "Standard Deviation of Volumes Between Cores", "Optimized", "Initial")
else:
    results = []
    for name in tqdm(args.datasets):
        coo_data, vtx_count, adj_mat = util.get_coo_mat(name)
        print(str(name) + ": " + str(vtx_count))
        iterator = get_core_iterator()
        for i in iterator:
            r = execute(i)
            # r.insert(0, name)
            results.append(r)
        del adj_mat
    util.create_excel(args.core_cnt, results, args.datasets)
