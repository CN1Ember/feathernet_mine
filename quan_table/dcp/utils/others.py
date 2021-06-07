import torch


def concat_gpu_data(data):
    """
    Concat gpu data from different gpu.
    """

    data_cat = data["0"]
    for i in range(1, len(data)):
        data_cat = torch.cat((data_cat, data[str(i)].cuda(0)))
    return data_cat
