import numpy as np


def calc_square(arr):
    return np.sum(np.square([len(x) for x in arr]))


# used for testing purposes
def count_maps(maps: (int, dict)):
    return np.sum(len(v) for v in maps[1].values())


class dist_tracker:
    def __init__(self):
        self.other_idx = -1
        self.other_send_list = []
        self.send_list = []

    def set(self, other_idx: int, other_send_list: list, send_list: list):
        self.other_idx = other_idx
        self.other_send_list = other_send_list
        self.send_list = send_list

    def reset(self):
        self.other_send_list.clear()
        self.send_list.clear()
        self.other_idx = -1


def calc_square_wt(arr, tracker: dist_tracker, idx: int):
    return calc_square(arr) - np.square(len(arr[idx])) + np.square(len(arr[idx]) + len(tracker.send_list))
