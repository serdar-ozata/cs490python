import numpy as np

from util import DestData, write_partitions, FolderM

reduce_map = []


def get_rdc(i: int) -> list[int]:
    return reduce_map[i - DestData.initial_vtx_cnt]


def has_dependency(vtx: int):
    if vtx < DestData.initial_vtx_cnt:
        return False
    owner_prc = DestData.partition[vtx]
    for dep_vtx in get_rdc(vtx):
        if DestData.partition[dep_vtx] == owner_prc:
            return True
    return False


def add_rdc_vtx(vtxs: list[int], owner: int, recv_vtx: int, recv_prc: int) -> int:
    """Returns the index of the added reduce vertex"""
    reduce_map.append(vtxs.copy())
    DestData.partition.append(owner)
    gen_vtx = len(reduce_map) + DestData.initial_vtx_cnt - 1
    DestData.vtx_edges[recv_prc].update({recv_vtx: {gen_vtx}})
    return gen_vtx


BETA = 2.0
REDUCE_VTX_THRESHOLD = 1


def get_volumes(dest_data: list[DestData]) -> list[int]:
    return [d.volume() for d in dest_data]


def get_max(volumes: list[int], filled_expands: list[bool]) -> int:
    max_volume = max(volumes)
    for i, volume in enumerate(volumes):
        if volume == max_volume and not filled_expands[i]:
            return i
    return -1  # return -1 if no such index exists


def reduce_post_processing(data: list[DestData], core_cnt: int, name: str):
    reduced_edges = [[] for _ in range(DestData.initial_vtx_cnt)]
    for d in DestData.vtx_edges:
        for rcv_vtx, edge_vtxs in d.items():
            reduced_edges[rcv_vtx] = list(edge_vtxs)

    # int: sender prc, int: receive vtx, set: vtxs owned by the sender and have edges to receive vtx after our algorithm
    parsed_expands: list[list[(int, int, set)]] = [[] for _ in range(core_cnt)]  # filled on demand for opt. reasons
    filled_expands = [False for _ in range(core_cnt)]
    while True:
        vls = get_volumes(data)
        max_idx = get_max(vls, filled_expands)
        if max_idx == -1:
            break
        avg_vol = np.mean(vls)
        # check whether reduce op is needed
        if avg_vol + BETA > vls[max_idx]:
            break
        # this data structure holds which vertices are requested by which vertex. In other words: the original edges
        local_vtx_edges = DestData.vtx_edges[max_idx].copy()
        # if not parsed, parse
        if not filled_expands[max_idx]:
            filled_expands[max_idx] = True
            # parse every edge group and sort them according to their degree
            local_prs_expands = dict()
            for (recv_vtx, vtxs) in local_vtx_edges.items():
                if DestData.partition[recv_vtx] == max_idx and len(vtxs) > REDUCE_VTX_THRESHOLD:
                    for vtx_to_reduce in vtxs:
                        owner_prc = DestData.partition[vtx_to_reduce]
                        owner_data = data[owner_prc]
                        if vtx_to_reduce in owner_data.reassign_cores:  # if reassigned:
                            reassigned_prc = owner_data.reassign_cores[vtx_to_reduce]
                            if reassigned_prc == max_idx:  # this vtx cannot be reduced if this is the case
                                continue
                            if max_idx not in owner_data.expands[vtx_to_reduce]:  # means the reassigned prc sends it
                                owner_prc = reassigned_prc
                                owner_data = data[reassigned_prc]
                        if max_idx not in owner_data.expands[vtx_to_reduce]:
                            raise ValueError("Vertex not found in owner's expand list")  # this should never happen
                        key_tuple = (owner_prc, recv_vtx)
                        if key_tuple not in local_prs_expands:
                            local_prs_expands[key_tuple] = set()
                        local_prs_expands[key_tuple].add(vtx_to_reduce)
            # sort the parsed expands according to their degree
            parsed_expands[max_idx].extend([(k[0], k[1], v) for k, v in local_prs_expands.items()
                                            if len(v) > REDUCE_VTX_THRESHOLD])
            parsed_expands[max_idx].sort(key=lambda x: len(x[2]))
            # keep 30 percent of the expands
            # parsed_expands[max_idx] = parsed_expands[max_idx][int(len(parsed_expands[max_idx]) * 0.7):]
            print(f"avg expand degree: {np.mean([len(x[2]) for x in parsed_expands[max_idx]])}")
        else:
            break
            # raise ValueError("If this case does happen, I have to change some stuff")

        all_skipped = True
        while len(parsed_expands[max_idx]) > 0:
            sender_idx, rcv_vtx, vtxs_to_reduce = parsed_expands[max_idx].pop()
            all_skipped = False
            # reduce operation
            gen_vtx = add_rdc_vtx(vtxs=vtxs_to_reduce, owner=sender_idx, recv_vtx=rcv_vtx, recv_prc=max_idx)
            # update reduced_edges
            reduced_edges[rcv_vtx] = [gen_vtx]

            # check whether reduced vertices can be removed from the expand list
            reduced_sender_data = data[sender_idx]

            for vtx in list(vtxs_to_reduce):
                if vtx in reduced_sender_data.reassign_cores and reduced_sender_data.reassign_cores[vtx] == max_idx:
                    vtxs_to_reduce.remove(vtx)  # cannot discard a vertex if it's reassigned
            local_vtx_edges.pop(rcv_vtx, None)  # no need to update with the reduced vertex, just remove it
            try:
                for f_recv_vtx, vtxs_needed in local_vtx_edges.items():
                    # if f_recv_vtx == rcv_vtx:
                    #     continue
                    for vtx_needed in vtxs_needed:
                        if vtx_needed in vtxs_to_reduce:
                            vtxs_to_reduce.remove(vtx_needed)
                            if len(vtxs_to_reduce) == 0:
                                raise StopIteration
            except StopIteration:
                pass
            removed_vtx_cnt = 0
            # the remaining vertices can be removed from the expand list
            for vtx_to_remove in vtxs_to_reduce:
                # print(vtx_to_remove, max_idx, sender_idx)
                if reduced_sender_data.remove_value(vtx_to_remove, max_idx):
                    removed_vtx_cnt += 1
            data[max_idx].decrease_recv_vol(removed_vtx_cnt - 1)
            # finally add the reduced vertex to the expand list
            reduced_sender_data.insert(gen_vtx, [max_idx])

            if avg_vol + BETA > data[max_idx].volume():  # can't use vls since it's outdated
                break
        if all_skipped:
            break
    save_reduced_graph_and_mappings(name, core_cnt, reduced_edges)


def save_reduced_graph_and_mappings(name, core_cnt, reduced_edges):
    write_partitions(DestData.partition, FolderM.get_name(f"{name}.inpart.{core_cnt}"))
    mmfname = FolderM.get_name(f"{name}.reduced.mtx")
    with open(mmfname, 'w') as f:
        total_vtx_cnt = DestData.initial_vtx_cnt + len(reduce_map)
        edge_cnt = sum([len(v) for v in reduced_edges])
        f.write(f"%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{total_vtx_cnt} {total_vtx_cnt} {edge_cnt}\n")
        for i, edges in enumerate(reduced_edges):
            for e in edges:
                f.write(f"{e + 1} {i + 1} 1\n")
