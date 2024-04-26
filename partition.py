import numpy as np

from util import DestData
from reduce import has_dependency


def get_p1_threshold(opt_send_list):
    core_cnt = len(opt_send_list)
    volumes = [0 for _ in range(core_cnt)]
    # first only insert the TRs
    for i in range(core_cnt):
        dest_data = opt_send_list[i]
        for k, prc in dest_data.reassign_cores.items():
            volumes[i] += 1
            volumes[prc] = DestData.alpha
    tr_vols = list(volumes)
    # then insert the OETs
    for i, dest_data in enumerate(opt_send_list):
        for k, v in dest_data.expands.items():
            # if not TR or RET
            if dest_data.is_oet(k):
                volumes[i] += len(v)
                for rec_idx in v:
                    volumes[rec_idx] += DestData.alpha
    return tr_vols, volumes


def get_phase1_oets(phase_vol, dest_data: DestData, tr_v):  # todo add reduce vertex check
    target_vol = phase_vol - tr_v
    selected = set()
    expands = [(k, len(v)) for k, v in dest_data.expands.items()]
    expands.sort(key=lambda i: i[1], reverse=True)
    # partition phase
    for k, sz in expands:
        # skip if it's a RET
        if not dest_data.is_owner(k):
            continue
        if dest_data.is_tr(k):
            selected.add(k)
            continue

        if target_vol >= sz:
            # if target_vol > 0:
            target_vol = target_vol - sz
            selected.add(k)
    selected_sz = phase_vol - tr_v - target_vol
    return selected


def get_phs1_subset_sum(opt_send_list, tr_vols, phase1_vol):
    return [get_phase1_oets(phase1_vol, dest_data, tr_v) for dest_data, tr_v in
            zip(opt_send_list, tr_vols)]


def get_phs1_lowest_volume(opt_send_list, tr_vols: list[int], threshold):
    volumes = tr_vols  # init with tr_vols (doesn't copy)
    expands = [
        [(k, v) for k, v in dest_data.expands.items() if dest_data.is_oet(k)]
        for dest_data in opt_send_list]  # get only OETs
    expands = [sorted(arr, key=lambda i: len(i[1]), reverse=True) for arr in expands]
    selected = [set() for _ in range(len(opt_send_list))]
    # select OETs
    while True:
        # find the lowest volume
        min_vol = min(volumes)
        if min_vol >= threshold:
            break

        # find the lowest volume's index
        min_idx = volumes.index(min_vol)

        # if there's no expand left, disable this processor
        if len(expands[min_idx]) == 0:
            volumes[min_idx] = threshold + 1
            continue

        # find the lowest volume's highest expand [0]: vtx, [1]: processors
        min_expand = expands[min_idx][0]

        # discard if it's too big or if it's a reduced vertex that has to be sent in phase 2
        if len(min_expand[1]) + min_vol > threshold or has_dependency(min_expand[0]):
            expands[min_idx].pop(0)
            continue

        # add it to the selected
        selected[min_idx].add(min_expand[0])
        # update send volume
        volumes[min_idx] += len(min_expand[1])
        # update receive volumes
        for v in min_expand[1]:
            volumes[v] += DestData.alpha

        # remove the highest expand
        expands[min_idx].pop(0)
    return selected
