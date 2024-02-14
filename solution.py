import math
import random
import time
import numpy as np
import pymetis
import util
import argparse
from tqdm import tqdm
from util import Assigment, MetricTracker, DestData, MIN_REASSIGN_LIMIT, get_opt_send_list
from binout import partition_phases, write_partitions, partition_one_phase
from metricsout import create_excel

epilog_text = '''Matrix market files are read from the ./mmdsets foler. Phase partition file is written to the ./out 
    folder. The other inpart files are written to ./mmdsets/schemes folder.'''
parser = argparse.ArgumentParser(
    description='''TODO description. Example usage: solution.py run --noconstructive 28 Flickr''',
    epilog=epilog_text
)

# Add a subparsers to the parser
subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

# Create a group of arguments

group = argparse.ArgumentParser(add_help=False)
group.add_argument('--noconstructive', action='store_true',
                   help='If set, the constructive algorithm will be disabled.')
group.add_argument('--noiterative', action='store_true',
                   help='If set, the iterative improvement algorithm will be disabled.')
group.add_argument('-v', '--volumemode', type=int,
                   help='Determines the initialization method for the bins. 0 for empty bins, 1 for bins with receive '
                        'volumes. Default is 1.',
                   default=1, choices=range(0, 2))
group.add_argument("--itercvthreshold", type=float,
                   help="Sets the threshold for the coefficient of variance of volumes. The iterative improvement "
                        "algorithm will continue to run until this threshold is met or the maximum iteration count "
                        "(--itermax) is reached. If not specified, the algorithm will run until the maximum iteration "
                        "count is reached.",
                   default=None)
group.add_argument("--itermax", type=int, help="Sets the maximum iteration count. Default is 1.", default=1)
group.add_argument("--onephase", action="store_true",
                   help="If set, an additional one phase partition file will be output.")
group.add_argument("--part_method", type=int,
                   help="Determines the partition method. 1 for filling by lowest volume, 2 for subset-sum. Default "
                        "is 1.",
                   default=1, choices=range(1, 3))

run_parser = subparsers.add_parser('run', help='Runs the algorithm on a specified dataset', parents=[group],
                                   epilog=epilog_text)
benchmark_parser = subparsers.add_parser('benchmark',
                                         help='Benchmarks the algorithm on specified datasets and core counts. '
                                              'Outputs an excel file in the ./out folder',
                                         parents=[group], epilog=epilog_text)

run_parser.add_argument('core_cnt', metavar='K', type=int,
                        help='The number of processors to be used.')
run_parser.add_argument('dataset_name', type=str,
                        help='The name of the dataset. Corresponding matrix market files should be located in the '
                             './mmdests folder.')

benchmark_parser.add_argument('core_cnt', metavar='K', type=int,
                              help='The number of processors to be used.')
benchmark_parser.add_argument("-l", "--loop", type=int,
                              help='The end core count (excluded). K becomes the start point and the program loops '
                                   'for different processor counts. Must be specified if -e or -i is used.',
                              default=None)

group = benchmark_parser.add_mutually_exclusive_group()
group.add_argument("-i", "--interval", type=int,
                   help='Specifies the increment for the processor count.', default=1)
group.add_argument("-e", "--exp", type=int,
                   help='Specifies the multiplier for the processor count in each iteration.')

benchmark_parser.add_argument('-d', '--datasets', type=str, nargs='+',
                              help='The names of the datasets. Corresponding matrix market files should be located in '
                                   'the ./mmdests folder.')

args = parser.parse_args()

dec_order = True

if args.noconstructive and args.noiterative:
    raise Exception("At least one of the algorithms must be enabled!")
if args.noconstructive:
    args.volumemode += 2
if args.mode == 'benchmark' and args.datasets is None:
    raise Exception("At least one dataset must be specified")
if args.mode == 'run' and args.dataset_name is None:
    raise Exception("Dataset name must be specified")
if args.noiterative and args.itercvthreshold is not None:
    raise Exception("itercvthreshold cannot be used without iterative algorithm")
print(args)


def get_core_iterator():
    if args.exp is None:
        return range(args.core_cnt, args.loop, args.interval)
    else:
        ratio = math.log(args.loop / args.core_cnt, args.exp)
        iter_count = int(np.ceil(ratio))
        return [(args.exp ** i) * args.core_cnt for i in range(iter_count)]


def execute(core_cnt, ignore_benchmark):
    # generate a random ownership list
    # mappings = pymetis.part_graph(core_cnt, adjacency=adj_mat)[1]
    mappings = pymetis.part_graph(core_cnt, adjacency=adj_mat, vweights=wg)[1]

    send_list = [dict() for _ in range(core_cnt)]
    DestData.partition = mappings
    parse_start_time = time.perf_counter()

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
        if args.itermax > 1:
            big_number = 100000  # some big number
            cv_threshold = args.itercvthreshold if args.itercvthreshold is not None else big_number
            iter_idx = 0
            cv = big_number + 1
            while iter_idx < args.itermax and cv > cv_threshold:
                iterative_improvement(opt_send_list, send_list, t, degree_list)
                opt_vols = [x.volume() for x in opt_send_list]
                cv = util.cv(opt_vols)
                iter_idx += 1
        else:
            iterative_improvement(opt_send_list, send_list, t, degree_list)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    p_execution_time = end_time - parse_start_time

    # communication partition
    two_phase_delay = partition_phases(opt_send_list, core_cnt, name, util.PartitionType(args.part_method))
    if args.onephase:
        partition_one_phase(send_list, core_cnt, name)

    # write partitions as file
    write_partitions(mappings, core_cnt, name)
    if not ignore_benchmark:
        vols = [util.get_volume(send_list[opt_d.id]) + opt_d.recv_vol for opt_d in opt_send_list]
        e_degrees = [len(arr) for s in send_list for arr in s.values()]
        e_degrees.sort()
        init_cv = util.cv(vols)

        # init_cost = np.sum([x ** 2 for x in vols])

        def extract_results():
            opt_vols = [x.volume() for x in opt_send_list]
            opt_cv = util.cv(opt_vols)
            # 0: vertex cnt, 1: edge count, 2: algo execution_time, 3: parse and algo execution_time, 4: core_cnt,
            # 5: t.reassign_cnt, 6: t.non_reassign_cnt, 7: reassign volume, 8: partition delay ,
            # 9: avg send vol per processor, 10: i vol cv, 11: opt  vol cv
            # 12: I highest volume, 13: highest volume, 14: I min vol, 15: min vol
            # 16: expand task count, 17: expand degree avg, 18: expand degree sum
            # 19: max expand degree, 20: min expand degree, 21: avg ed, 22: cv ed, 23: highest 10% / total
            # 22: sender's avg expand task size after reassignment

            execution_res = [vtx_count, len(coo_data[0]), execution_time, p_execution_time, core_cnt, t.reassign_cnt,
                             t.non_reassign_cnt, t.reassign_vol, two_phase_delay,  # 8
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
    # return nothing if benchmark mode is not enabled


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
    core_dest_data.insert(h, [expand_core])
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


# Check the mode of operation and call the appropriate function
if args.mode == 'run':
    name = args.dataset_name
    coo_data, vtx_count, adj_mat, wg = util.get_coo_mat(name)
    execute(args.core_cnt, ignore_benchmark=True)
elif args.mode == 'benchmark':
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
                r = execute(i, ignore_benchmark=False)
                # r.insert(0, name)
                results.append(r)
            del adj_mat
        create_excel(args, results, args.datasets)
# else part is unreachable
