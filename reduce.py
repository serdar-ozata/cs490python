import numpy as np

from util import DestData

reduce_map = []
og_vtx_cnt = -1


def get_rdc(i: int) -> list[int]:
    return reduce_map[i - og_vtx_cnt]


def add_rdc_vtx(vtxs: list[int], owner: int, recv_vtx: int, recv_prc: int) -> int:
    """Returns the index of the added reduce vertex"""
    reduce_map.append(vtxs.copy())
    DestData.partition.append(owner)
    gen_vtx = len(reduce_map) + og_vtx_cnt - 1
    DestData.vtx_reqs[recv_prc].update({recv_vtx: {gen_vtx}})
    return gen_vtx


BETA = 2.0
REDUCE_VTX_THRESHOLD = 2


def get_volumes(dest_data: list[DestData]) -> list[int]:
    return [d.volume() for d in dest_data]


def reduce_post_processing(data: list[DestData], core_cnt: int):
    # parse every expand task and sort them according to their degree
    parsed_expands: list[list[(int, int, set)]] = [[] for _ in range(core_cnt)]
    while True:
        vls = get_volumes(data)
        max_idx = np.argmax(vls)
        avg_vol = np.mean(vls)
        # check whether reduce op is needed
        if avg_vol + BETA > vls[max_idx]:
            break
        # this data structure holds which vertices are requested by which vertex. In other words: the original edges
        local_vtx_reqs = DestData.vtx_reqs[max_idx]
        # if not parsed, parse
        if not parsed_expands[max_idx]:
            local_prs_expands = dict()
            for (recv_vtx, vtxs) in local_vtx_reqs.items():
                if DestData.partition[recv_vtx] == max_idx and len(vtxs) > REDUCE_VTX_THRESHOLD:
                    for vtx_to_reduce in vtxs:
                        owner_prc = DestData.partition[vtx_to_reduce]
                        owner_data = data[owner_prc]
                        if vtx_to_reduce in owner_data.reassign_cores:  # if reassigned:
                            owner_prc = owner_data.reassign_cores[vtx_to_reduce]
                            owner_data = data[owner_prc]
                        if vtx_to_reduce not in owner_data.expands[max_idx]:
                            raise ValueError("Vertex not found in owner's expand list")  # this should never happen
                        key_tuple = (owner_prc, recv_vtx)
                        if key_tuple not in local_prs_expands:
                            local_prs_expands[key_tuple] = set()
                        local_prs_expands[key_tuple].add(vtx_to_reduce)
            # sort the parsed expands according to their degree
            parsed_expands[max_idx].extend([(k[0], k[1], v) for k, v in local_prs_expands.items()
                                            if len(v) > REDUCE_VTX_THRESHOLD])
            parsed_expands[max_idx].sort(key=lambda x: len(x[2]))

        all_skipped = True

        while len(parsed_expands[max_idx]) > 0:
            sender_idx, rcv_vtx, vtxs_to_reduce = parsed_expands[max_idx].pop()
            all_skipped = False
            # reduce operation
            gen_vtx = add_rdc_vtx(vtxs=vtxs_to_reduce, owner=sender_idx, recv_vtx=rcv_vtx, recv_prc=max_idx)
            # check whether reduced vertices can be removed from the expand list
            reduced_sender_data = data[sender_idx]
            for f_recv_vtx, vtxs_needed in local_vtx_reqs.items():
                if f_recv_vtx == rcv_vtx:
                    continue
                for vtx_needed in vtxs_needed:
                    if vtx_needed in vtxs_to_reduce:
                        vtxs_to_reduce.remove(vtx_needed)
                        if len(vtxs_to_reduce) == 0:
                            break
            # the remaining vertices can be removed from the expand list
            for vtx_to_remove in vtxs_to_reduce:
                reduced_sender_data.expands[max_idx].remove(vtx_to_remove)

            if avg_vol + BETA > data[max_idx].volume():  # can't use vls since it's outdated
                break
        if all_skipped:
            break
