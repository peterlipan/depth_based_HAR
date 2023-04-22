import numpy as np
from numpy.random import randint


class SegmentedSample:
    def __init__(self,
                 new_length,
                 num_segments,
                 test_mode=False,
                 start_index=1):
        """
        Segmented sampling
        :param new_length: 对于RGB模态，clip_len=1；对于RGBDiff模态，clip_len=6
        :param frame_interval: 单次clip中帧间隔
        :param num_clips: 将视频帧均分成num_clips，在每个clip中随机采样clip_len帧
        :param is_train:
        :param start_index:　数据集下标从0或者1开始
        """

        self.new_length = new_length
        self.num_segments = num_segments
        self.test_mode = test_mode
        self.start_index = start_index

    def _get_train_indices(self, num_frames):
        """
        :param num_frames: number of frames
        :return: list
        """
        # split the video into segments
        # each segment contains average_duration frames
        average_duration = (num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            # 1. init the offset for each segment as the first frame of it
            # 2. assign a random int value (0 to average_duration) to the offset
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif num_frames > self.num_segments:
            offsets = np.sort(randint(num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets

    def _get_test_indices(self, num_frames):

        tick = (num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __call__(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            list: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_indices(num_frames)
        else:
            clip_offsets = self._get_train_indices(num_frames)

        # Ensure that the output indices will not exceed the num_frames
        clip_offsets = np.clip(clip_offsets, 0, num_frames - self.new_length)

        # Sample new_length frames for each clip
        indices = np.expand_dims(np.arange(self.new_length), 0) + np.expand_dims(clip_offsets, 1)

        # flatten the list
        indices = indices.flatten().tolist()

        return indices
