import struct
from itertools import chain
from typing import Sized

import numpy as np

import partition
import util
from util import DestData, PartitionType


def assign_comm_list(dest_data: DestData, comm_list: list[(dict[list], dict[list])], send: bool, p1_selection: set):
    sender_idx = dest_data.id
    for vtx, v in dest_data.expands.items():
        if vtx in p1_selection:
            dict_idx = 0
        else:
            dict_idx = 1
        for rec_idx in v:
            main_idx = sender_idx if send else rec_idx
            other_idx = rec_idx if send else sender_idx
            if other_idx not in comm_list[main_idx][dict_idx]:
                comm_list[main_idx][dict_idx][other_idx] = []
            comm_list[main_idx][dict_idx][other_idx].append(vtx)
    # for vtx, rec_idx in dest_data.reassign_cores.items():
    #     main_idx = sender_idx if send else rec_idx
    #     other_idx = rec_idx if send else sender_idx
    #     if other_idx not in comm_list[other_idx][0]:
    #         comm_list[main_idx][0][other_idx] = []
    #     comm_list[main_idx][0][other_idx].append(vtx)


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


def write_bin_file(file, arr):
    # Writing array
    for element in arr:
        file.write(struct.pack('i', element))


def update_start_positions(file, processor_start_positions):
    file.seek(0)
    # Update the first n * 8 bytes in the file with actual processor start positions
    for position in processor_start_positions:
        file.write(struct.pack('q', position))


# writes phase partitions to binary file
def partition_phases(opt_send_list: list[DestData], core_cnt: int, name: str, partition_type: PartitionType,
                     node_core_cnt: int):
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
        # phase1_vol = tr_max
    # phase partition
    if partition_type == PartitionType.SUBSET_SUM:
        phase1_selections = partition.get_phs1_subset_sum(opt_send_list, tr_vols, phase1_vol)
    elif partition_type == PartitionType.LOWEST_VOLUME:
        phase1_selections = partition.get_phs1_lowest_volume(opt_send_list, tr_vols, phase1_vol)
    else:
        raise Exception("Unknown partition type")

    # get recv & send lists (phase1, phase2)
    recv_lists: list[(dict[list], dict[list])] = [(dict(), dict()) for _ in range(core_cnt)]
    send_lists: list[(dict[list], dict[list])] = [(dict(), dict()) for _ in range(core_cnt)]
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
    # difference between lowest and highest volume
    # p1_vols = [send_vols[0][i] + recv_vols[0][i] for i in range(core_cnt)]
    # min_vol = np.min(p1_vols)
    # max_vol = np.max(p1_vols)
    # print(
    #     f"Partition of the sample {name}, {core_cnt}: min: {min_vol}, max: {max_vol}, delay: {delay}, threshold: {phase1_vol}")

    # Create and open the binary file with placeholder values
    fname = f"out/{name}.phases.{core_cnt}.bin"
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
            # write send and recv vertexes
            vertexes = list(chain(chain(*send_lists[i][0].values()), chain(*recv_lists[i][0].values()),
                                  chain(*send_lists[i][1].values()), chain(*recv_lists[i][1].values())))
            write_bin_file(file, vertexes)
        # write proc ptrs
        update_start_positions(file, proc_ptrs)

    return delay


def partition_one_phase(send_list: list[dict[set]], core_cnt: int, name: str, node_core_cnt: int):
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
            vertexes = list(chain(*send_lists[i].values(), *recv_lists[i].values()))

            write_bin_file(file, vertexes)

        # write proc ptrs
        update_start_positions(file, proc_ptrs)


# writes vertex mappings to binary file
def write_partitions(mappings: list, core_cnt: int, name: str):
    fname = f"mmdsets/schemes/{name}.inpart.{core_cnt}"
    with open(fname, 'w') as file:
        for m in mappings:
            file.write(f"{m}\n")


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
