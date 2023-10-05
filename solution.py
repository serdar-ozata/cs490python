import statistics
import time

import numpy as np
import pymetis
import util
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('core_cnt', metavar='K', type=int, help='core count')
parser.add_argument("-loop", type=int, help='end core count (excluded)')
parser.add_argument("-interval", type=int, help='core count increment', default=1)
parser.add_argument("-exp", help='multiply core count by 2 in each iteration', action='store_true')
args = parser.parse_args()
print(args)
coo_data, vtx_count, adj_mat = util.get_coo_mat()
print("Vertex Count:", vtx_count)


def execute(core_cnt):
    # generate a random ownership list
    mappings = pymetis.part_graph(core_cnt, adjacency=adj_mat)[1]

    send_list = [(i, dict()) for i in range(core_cnt)]

    parse_start_time = time.perf_counter()

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
    opt_send_list = [util.DestData(v) for v in range(core_cnt)]
    pcost = 0

    ext_cnt = 0
    non_ext_cnt = 0

    start_time = time.perf_counter()
    # distribution algorithm
    for i in range(core_cnt):
        cpu_idx = send_list[i][0]
        send_dict = send_list[i][1]
        for h, send_set in send_dict.items():
            core_dest_data = opt_send_list[cpu_idx]
            best_del_cost = core_dest_data.delta_sqr(h, send_set)
            expand_core = -1  # means self
            for other_idx in send_set:
                other_dest_data = opt_send_list[other_idx]

                delta_cost = other_dest_data.delta_sqr(h, send_set) + core_dest_data.delta_sqr(h, [other_idx])
                if delta_cost < best_del_cost:
                    best_del_cost = delta_cost
                    expand_core = other_idx

            pcost += best_del_cost
            if expand_core > -1:
                ext_cnt += 1
                core_dest_data.insert(h, [expand_core])
                # pcost += 2 * core_dest_data.volume - 1
                opt_send_list[expand_core].insert(h, send_set)
            else:
                core_dest_data.insert(h, send_set)
                non_ext_cnt += 1

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    p_execution_time = end_time - parse_start_time

    vols = [util.get_volume(x[1]) for x in send_list]
    counter = np.zeros(dtype=int, shape=core_cnt)
    opt_vols = [x.volume for x in opt_send_list]
    opt_std_dev = np.zeros(core_cnt)
    std_dev = np.zeros(core_cnt)

    for j in range(len(opt_vols)):
        idx = mappings[j]
        opt_std_dev[idx] += opt_vols[j]
        std_dev[idx] += vols[j]
        counter[idx] += 1

    # 0: vertex cnt, 1: edge count, 2: algo execution_time, 3: parse and algo execution_time, 4: core_cnt,
    # 5: ext_cnt, 6: non_ext_cnt, 7: square sum, 8: initial square sum, 9: square sum (calculated from the ground up)
    # 10: highest volume, 11: initial highest volume, 12: avg send vol per processor,
    # -: initial avg send vol per processor, 13: std of volume between cores, 14: initial std of volume between cores
    return np.array([vtx_count, len(coo_data[0]), execution_time, p_execution_time, core_cnt, ext_cnt, non_ext_cnt,
                     pcost, util.calc_square([x[1] for x in send_list]), np.sum([x.cost_raw() for x in opt_send_list]),
                     np.max(opt_vols), np.max(vols), np.sum(vols) / core_cnt, statistics.stdev(opt_std_dev),
                     statistics.stdev(std_dev)])


if args.loop is None:
    res = execute(args.core_cnt)
    util.write_results(res)
else:
    if args.exp is None:
        iter = range(args.core_cnt, args.loop, args.interval)
    else:
        ratio = np.log2(args.loop / args.core_cnt)
        iter_count = int(np.ceil(ratio))
        iter = [(2 ** i) * args.core_cnt for i in range(iter_count)]
    results = []
    for i in tqdm(iter):
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
