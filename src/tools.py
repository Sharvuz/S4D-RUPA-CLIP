import torch
import numpy as np


def get_batch_mask(lengths, maxlen):
    """Generate boolean padding mask where True indicates padded positions."""
    batch_size = lengths.shape[0]
    mask = torch.zeros(batch_size, maxlen, dtype=torch.bool)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = True
    return mask


def random_extract(feat, t_max):
    r = np.random.randint(feat.shape[0] - t_max)
    return feat[r : r + t_max, :]


def uniform_extract(feat, t_max, avg=True):
    new_feat = np.zeros((t_max, feat.shape[1]), dtype=np.float32)
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    if avg:
        for i in range(t_max):
            if r[i] != r[i + 1]:
                new_feat[i, :] = np.mean(feat[r[i] : r[i + 1], :], axis=0)
            else:
                new_feat[i, :] = feat[r[i], :]
    else:
        indices = np.linspace(0, feat.shape[0] - 1, t_max, dtype=np.int32)
        new_feat = feat[indices, :]
    return new_feat


def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length < min_len:
        return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    return feat


def process_feat(feat, length, is_random=False):
    clip_length = feat.shape[0]
    if clip_length > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length


def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length <= length:
        return pad(feat, length), clip_length
    else:
        split_num = int(np.ceil(clip_length / length))
        split_list = []
        for i in range(split_num):
            start = i * length
            end = min((i + 1) * length, clip_length)
            chunk = feat[start:end, :]
            if chunk.shape[0] < length:
                chunk = pad(chunk, length)
            split_list.append(chunk[np.newaxis, :, :])
        split_feat = np.concatenate(split_list, axis=0)
        return split_feat, clip_length
