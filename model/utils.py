import torch
import math
import torch.nn.functional as F


def reshape(x, window_len, ind=None, dim=1):
    init_len = x.size(dim=dim)
    padding = (math.ceil(window_len / 2), (math.ceil(2 * init_len / window_len) - 2) * math.ceil(window_len / 2) + window_len - init_len)
    x = F.pad(x, padding, 'constant', 0.)

    x = x.unsqueeze(dim=dim)

    sizes_x = [1 for _ in range(len(x.size()) - 2)] + [window_len, 1]
    x = x.repeat(sizes_x)

    if ind is None or ind.size()[:-2] != x.size()[:-2]:
        ind = torch.range(0, (math.ceil(2 * init_len / window_len) - 1) * math.ceil(window_len / 2), math.ceil(window_len / 2))

        ind = ind.unsqueeze(dim=0)
        ind = ind.repeat(window_len, 1)
        col = torch.range(0, window_len - 1).unsqueeze(dim=1)
        ind = ind + col
        ind = ind.unsqueeze(dim=0)

        sizes_ind = [size for size in x.size()[:-2]] + [1, 1]
        ind = ind.repeat(sizes_ind)
        ind = ind.long()

    x = torch.gather(x, dim + 1, ind)

    return x, ind, padding


# get from https://github.com/kaituoxu/Conv-TasNet/blob/94eac1023eaaf11ca1bf3c8845374f7e4cd0ef4c/src/utils.py
def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().reshape(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.reshape(*outer_dimensions, -1)
    return result
