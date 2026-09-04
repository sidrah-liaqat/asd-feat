# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky
import numpy as np
def get_middle_chunk(frame):

    mid_size = frame // 2
    mod1 = frame % 2
    mod2 = mid_size % 2

    start_index = int(mid_size / 2)

    end_index = start_index + mid_size + mod1 + mod2

    return start_index, end_index

def calculate_window_indices(frame_window_size, overlap_windows):
    """
    Computes srt_idx, end_idx, and func_frame based on windowing settings.

    Args:
        frame_window_size (int): The total length of the data window (your 'frame' variable).
        overlap_windows (bool): Whether windowing has overlap (your 'overlap' variable).

    Returns:
        tuple: (srt_idx, end_idx, func_frame)
    """
    if not overlap_windows:
        srt_idx = 0
        end_idx = frame_window_size
        func_frame = frame_window_size
    else:
        srt_idx, end_idx = get_middle_chunk(frame_window_size)
        func_frame = end_idx - srt_idx
    return srt_idx, end_idx, func_frame

