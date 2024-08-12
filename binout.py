import os
import struct
from itertools import chain
from typing import Sized

import numpy as np

import partition
import util
from util import DestData, PartitionType, FolderM, write_bin_file
from reduce import reduce_map, get_rdc


def assign_comm_list(dest_data: DestData, comm_list: list[(dict, dict)], send: bool, p1_selection: set):
    sender_idx = dest_data.id

    for vtx, rec_idx in dest_data.reassign_cores.items():
        main_idx = sender_idx if send else rec_idx
        other_idx = rec_idx if send else sender_idx
        if other_idx not in comm_list[main_idx][0]:
            comm_list[main_idx][0][other_idx] = []
        comm_list[main_idx][0][other_idx].append(vtx)

    for vtx, prcs in dest_data.expands.items():
        if vtx in p1_selection:
            dict_idx = 0
        else:
            dict_idx = 1
        for rec_idx in prcs:
            main_idx = sender_idx if send else rec_idx
            other_idx = rec_idx if send else sender_idx
            if other_idx not in comm_list[main_idx][dict_idx]:
                comm_list[main_idx][dict_idx][other_idx] = []
            comm_list[main_idx][dict_idx][other_idx].append(vtx)


def write_ranges(file, data_list: dict[int, Sized], core_cnt: int):
    curr_ptr = 0
    file.write(struct.pack('i', curr_ptr))
    for i in range(core_cnt):
        curr_ptr += len(data_list[i]) if i in data_list else 0
        file.write(struct.pack('i', curr_ptr))
    return curr_ptr


def init_bin_file(file, num_processors):
    # Writing placeholder values for processor start positions to the binary file
    placeholder_positions = [0] * num_processors

    # Writing n * 8 bytes as a placeholder for processor start positions
    for _ in range(num_processors):
        file.write(struct.pack('q', 0))


def update_start_positions(file, processor_start_positions):
    file.seek(0)
    # Update the first n * 8 bytes in the file with actual processor start positions
    for position in processor_start_positions:
        file.write(struct.pack('q', position))


# writes phase partitions to binary file
def partition_phases(opt_send_list: list[DestData], core_cnt: int, name: str, partition_type: PartitionType,
                     noreduce: bool):
    # remove reassigned vertices from the list
    total_reassign_count = 0
    for dest_data in opt_send_list:
        total_reassign_count += len(dest_data.reassign_cores)
        for vtx, prc in dest_data.reassign_cores.items():
            # if prc not in dest_data.expands[vtx]:
            #     raise Exception(f"BUG: {vtx} is reassigned to {prc} but not in expands")
            dest_data.expands[vtx].remove(prc)
    tr_vols, phase1_vols = partition.get_p1_threshold(opt_send_list)
    tr_max = np.max(tr_vols)
    phase1_min = np.min(phase1_vols)

    # get the delay which will be returned
    if tr_max > phase1_min:
        max_vol = np.max([x.volume() for x in opt_send_list])
        delay = 100 * (tr_max - phase1_min) / max_vol
        # print(f"Partition of the sample {name}, {core_cnt} will not be perfect: %{delay} slower")
        phase1_vol = tr_max
    else:
        delay = 0
        phase1_vol = int(np.ceil((tr_max + phase1_min) / 2))
    # phase partition
    if partition_type == PartitionType.NONE:
        phase1_selections = [set() for _ in opt_send_list]
    elif partition_type == PartitionType.SUBSET_SUM:
        phase1_selections = partition.get_phs1_subset_sum(opt_send_list, tr_vols, phase1_vol)
    elif partition_type == PartitionType.LOWEST_VOLUME:
        phase1_selections = partition.get_phs1_lowest_volume(opt_send_list, tr_vols, phase1_vol)
    else:
        raise Exception("Unknown partition type")

    # get recv & send lists (phase1, phase2)
    recv_lists: list[(dict, dict)] = [(dict(), dict()) for _ in range(core_cnt)]
    send_lists: list[(dict, dict)] = [(dict(), dict()) for _ in range(core_cnt)]
    for i in range(core_cnt):
        dest_data = opt_send_list[i]
        # get phase1 OETs and TRs
        p1_selection = phase1_selections[i]
        # recv part
        assign_comm_list(dest_data, recv_lists, False, p1_selection)
        # send part
        assign_comm_list(dest_data, send_lists, True, p1_selection)
    # print send_lists and recv_lists volumes
    send_vols = [[sum([len(v) for v in send_lists[i][k].values()]) for i in range(core_cnt)] for k in range(2)]
    recv_vols = [[sum([len(v) for v in recv_lists[i][k].values()]) for i in range(core_cnt)] for k in range(2)]
    if np.sum(send_vols[0]) != np.sum(recv_vols[0]) or np.sum(send_vols[1]) != np.sum(recv_vols[1]):
        print("BUG: send and recv volumes are not equal")
        exit(1)
    total_send_vols = [send_vols[0][i] + send_vols[1][i] for i in range(core_cnt)]
    total_recv_vols = [recv_vols[0][i] + recv_vols[1][i] for i in range(core_cnt)]
    max_send_vol = np.max(total_send_vols)
    max_recv_vol = np.max(total_recv_vols)
    avg_send_vol = int(np.floor(np.mean(total_send_vols)))
    avg_recv_vol = int(np.floor(np.mean(total_recv_vols)))
    send_counts = [len(send_lists[i][0]) + len(send_lists[i][1]) for i in range(core_cnt)]
    recv_counts = [len(recv_lists[i][0]) + len(recv_lists[i][1]) for i in range(core_cnt)]
    max_send_count = np.max(send_counts)
    max_recv_count = np.max(recv_counts)
    avg_send_count = int(np.floor(np.mean(send_counts)))
    avg_recv_count = int(np.floor(np.mean(recv_counts)))
    print(
        f"{name} {core_cnt}:{max_send_vol},{avg_send_vol},{max_recv_vol},{avg_recv_vol},{avg_send_count},{max_send_count},{avg_recv_count},{max_recv_count}")
    print(f"total send volume: {np.sum(send_vols[0]) + np.sum(send_vols[1])}")
    # test
    # dep_counts = np.zeros((core_cnt, core_cnt), dtype=int)
    # dep_sets = [set() for _ in range(core_cnt * core_cnt)]
    # for i in range(core_cnt):
    #     for reassigned_vtx, reassigned_prc in opt_send_list[i].reassign_cores.items():
    #         for rec_idx in opt_send_list[reassigned_prc].expands[reassigned_vtx]:
    #             dep_sets[reassigned_prc * core_cnt + rec_idx].add(i)
    # for i in range(core_cnt):
    #     for j in range(core_cnt):
    #         dep_counts[i][j] = len(dep_sets[i * core_cnt + j])
    # del dep_sets
    # print_stats_in_csv_format_2d(dep_counts)
    # del dep_counts
    # avg_send_counts = np.zeros(core_cnt, dtype=int)
    # for i in range(core_cnt):
    #     avg_send_counts[i] += len(send_lists[i][0].keys()) + len(send_lists[i][1].keys())
    # print(f"send count avg: {np.mean(avg_send_counts)}, max: {np.max(avg_send_counts)}, min: {np.min(avg_send_counts)}")
    # END test
    # test
    # check if there is any duplicate
    # for sl in send_lists:
    #     for p in range(2):
    #         for prc, vtxs in sl[p].items():
    #             if len(vtxs) != len(set(vtxs)):
    #                 print(vtxs)
    #                 exit(1)

    # END test
    # map reduced vtxs into processors
    reduced_vtx_prc_map: list[list[int]] = [[] for _ in range(core_cnt)]
    for i in range(DestData.initial_vtx_cnt, len(reduce_map) + DestData.initial_vtx_cnt):
        reduced_vtx_prc_map[DestData.partition[i]].append(i)
    # Create and open the binary file with placeholder values
    reduce_text = "noreduce" if noreduce else "reduced"
    fname = FolderM.get_name(f"{name}.phases.{core_cnt}.{reduce_text}.bin")
    with open(fname, 'w+b') as file:
        init_bin_file(file, core_cnt)
        proc_ptrs = []
        for i in range(core_cnt):
            proc_ptrs.append(file.tell())
            # selected[] and TRs will be uploaded in phase 1

            # uncomment these if you need them
            # tr_v, oet_v, ret_v = com_type_vols[i]
            # selections, selected_sz = phase1_selections[i]
            # write counts
            write_bin_file(file, [send_vols[0][i], recv_vols[0][i], send_vols[1][i], recv_vols[1][i]])
            # write ranges
            write_ranges(file, send_lists[i][0], core_cnt)
            write_ranges(file, recv_lists[i][0], core_cnt)
            write_ranges(file, send_lists[i][1], core_cnt)
            write_ranges(file, recv_lists[i][1], core_cnt)
            # convert to list
            vertexes = []
            for p in range(2):
                for l in range(core_cnt):
                    vertexes.extend(sorted(send_lists[i][p][l]) if l in send_lists[i][p] else [])
                for l in range(core_cnt):
                    vertexes.extend(sorted(recv_lists[i][p][l]) if l in recv_lists[i][p] else [])
            write_bin_file(file, vertexes)
            # write reduced vertexes
            write_reduced_vtxs(file, reduced_vtx_prc_map[i])

        # write proc ptrs
        update_start_positions(file, proc_ptrs)

    return delay


def write_reduced_vtxs(file, reduced_vtxs: list[int]):
    # write reduced vertexes
    write_bin_file(file, [len(reduced_vtxs)])
    for vtx in reduced_vtxs:
        vtxs_reduced_from = get_rdc(vtx)
        # vtx_id, vtx_reduced_from_count, ...vtxs_reduced_from
        write_bin_file(file, [vtx, len(vtxs_reduced_from), *vtxs_reduced_from])


def partition_one_phase(vtx_based_send_list: list[dict[set]], core_cnt: int, name: str, node_core_cnt: int):
    recv_lists, send_lists = util.parse_processor_based_lists(vtx_based_send_list, core_cnt)

    # get their volumes
    send_vols = [sum(len(v) for v in send_lists[i].values()) for i in range(core_cnt)]
    recv_vols = [sum(len(v) for v in recv_lists[i].values()) for i in range(core_cnt)]
    # write to binary file
    fname = f"out/{name}.phases.{core_cnt}.one.bin"
    with open(fname, 'w+b') as file:
        init_bin_file(file, core_cnt)
        proc_ptrs = []
        for i in range(core_cnt):
            proc_ptrs.append(file.tell())
            # write counts
            write_bin_file(file, [send_vols[i], recv_vols[i]])
            # write ranges
            write_ranges(file, send_lists[i], core_cnt)
            write_ranges(file, recv_lists[i], core_cnt)
            # write send and recv vertexes
            vertexes = [val for list in (send_lists, recv_lists) for l in range(core_cnt) for val in
                        (sorted(list[i][l]) if l in list[i] else [])]
            write_bin_file(file, vertexes)

        # write proc ptrs
        update_start_positions(file, proc_ptrs)


def get_out_of_node_vol_info(lists, core_cnt, node_core_cnt):
    out_node_vols = [0 for _ in range(core_cnt)]
    for i in range(core_cnt):
        sl = lists[i]
        range_st, range_end = util.get_node_range(i, core_cnt, node_core_cnt)
        dsum = 0
        for k in range(2):
            for j in range(range_st, range_end):
                if j in sl[k]:
                    dsum += len(sl[k][j])
        out_node_vols[i] = dsum
    # sum each node core_cnt elements
    out_node_vols = [sum(out_node_vols[i:i + node_core_cnt]) for i in range(0, len(out_node_vols), node_core_cnt)]
    return np.max(out_node_vols), np.min(out_node_vols), np.mean(out_node_vols)


def get_out_of_node_vol_info_one_phs(lists, core_cnt, node_core_cnt):
    out_node_vols = [0 for _ in range(core_cnt)]
    for i in range(core_cnt):
        sl = lists[i]
        range_st, range_end = util.get_node_range(i, core_cnt, node_core_cnt)
        dsum = 0
        for j in range(range_st, range_end):
            if j in sl:
                dsum += len(sl[j])
        out_node_vols[i] = dsum
    # sum each node_core_cnt elements
    out_node_vols = [sum(out_node_vols[i:i + node_core_cnt]) for i in range(0, len(out_node_vols), node_core_cnt)]
    # out_node_vols = [out_node_vols[i:i + node_core_cnt] for i in range(0, len(out_node_vols), node_core_cnt)]
    return np.max(out_node_vols), np.min(out_node_vols), np.mean(out_node_vols)


def print_stats_in_csv_format_2d(data):
    total_max = np.max(data)
    total_avg = np.mean(data)
    total_min = np.min(data)
    cv = np.std(data) / total_avg
    prc_avgs = np.mean(data, axis=0)
    max_prc_avg = np.max(prc_avgs)
    min_prc_avg = np.min(prc_avgs)
    # max,avg,min,cv,max_prc_avg,min_prc_avg
    print(f"{total_max},{total_avg},{total_min},{cv},{max_prc_avg},{min_prc_avg}")
