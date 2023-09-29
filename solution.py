import numpy as np
import util

core_cnt = 100

coo_data, vtx_count = util.get_coo_mat()
# coo_data, vtx_count = util.create_coo_mat()

print("Vertex Count:", vtx_count)
# generate a random ownership list
# np.random.seed(0)  # set seed
# mappings = np.array([0, 1, 1, 2, 2, 3, 3])
mappings = np.random.randint(0, core_cnt, vtx_count)
# mappings = util.get_uniform_mapping(core_cnt, vtx_count)
send_list = [(i, dict()) for i in range(core_cnt)]

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
print(send_list)
opt_send_list = [util.DestData(v) for v in range(core_cnt)]
pcost = 0


def delta_sqr(idx, diff) -> int:
    return np.square(len(opt_send_list[idx]) + diff) - np.square(len(opt_send_list[idx]))


ext_cnt = 0
non_ext_cnt = 0

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

print("Core Count %d" % core_cnt)
print("Number of extend operations: %d" % ext_cnt)
print("Number of non-extended operations: %d" % non_ext_cnt)
print("Min Sum Square: %d, Initial: %d" % (pcost, util.calc_square([x[1] for x in send_list])))
print("Min Sum Square without delta: %d" % np.sum([x.cost_raw() for x in opt_send_list]))
print("Highest Volume %d, Initial: %d" % (np.max([x.volume for x in opt_send_list]), np.max([util.get_volume(x[1]) for x in send_list])))
print("Initial Distribution:")
print(send_list)
print("Optimized Distribiton (Indexed):")
print(opt_send_list)
