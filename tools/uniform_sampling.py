import torch


def sampled_id_generation(lower_within, upper_not_within, size):
    return torch.FloatTensor(size).uniform_(lower_within, upper_not_within).long().tolist()
