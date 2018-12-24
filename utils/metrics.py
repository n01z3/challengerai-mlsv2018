import numpy as np

def accuracy(outputs, tags, topk=5):
    res = np.zeros(topk)
    if outputs.dim() == 1:
        return ch_metric(outputs, tags, topk)
    for i in range(outputs.shape[0]):
        res += ch_metric(outputs[i], tags[i], topk)
    res /= outputs.shape[0]

    return res 

def ch_metric(output, tags, topk):
    y = tags.numpy().flatten().nonzero()
    y = set(y[0])
    res = np.zeros(topk)
    for i in range(1, topk + 1):
        _, pred = output.topk(i)
        pred = set(pred.numpy())
        res[i - 1] = len(set.intersection(pred, y)) / len(set.union(pred, y))
    return res
