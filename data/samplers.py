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
        indices = indices.flatten() + self.start_index

        return indices


class HierarchySample:
    def __init__(self,
                 frame_per_seg,
                 num_segments,
                 margin,
                 test_mode=False,
                 start_index=1):
        """
        Segmented sampling
        :param frame_per_seg: number of frames of each segment to be sampled
        :param num_segments: number of segments
        :param test_mode: if True, use the test indices
        :param start_index:　the starting index for the frames, usually 0 or 1
        """

        self.frame_per_seg = frame_per_seg
        self.num_segments = num_segments
        self.test_mode = test_mode
        self.start_index = start_index
        self.margin = margin

    def _get_train_indices(self, num_frames):
        """
        :param num_frames: number of frames
        :return: list of segments
        """
        # split the indices into segments
        if num_frames >= self.num_segments:
            segments = np.array_split(np.arange(num_frames), self.num_segments)
            # if the number of frames is not enough for sampling with replacement
            replace = num_frames < self.num_segments * self.frame_per_seg
            indices = [np.sort(np.random.choice(seg, self.frame_per_seg, replace=replace)) for seg in segments]
        else:
            indices = np.zeros((self.num_segments, self.frame_per_seg))

        return indices

    def _get_test_indices(self, num_frames):
        """
        :param num_frames: number of frames
        :return: list of segments
        """
        # split the indices into segments
        if num_frames >= self.num_segments:
            segments = np.array_split(np.arange(num_frames), self.num_segments)
            # sample uniformly distributed frames
            # if num_frames < self.num_segments * self.new_length * self.margin, set margin=1
            margin = self.margin if num_frames >= self.num_segments * self.frame_per_seg * self.margin else 1
            if num_frames >= self.num_segments * self.frame_per_seg:
                start_sub_idx = [(len(seg) - self.frame_per_seg * margin) // 2 for seg in segments]
                start_idx = [seg[sub_idx] for seg, sub_idx in zip(segments, start_sub_idx)]
                indices = [np.array([idx + i * margin for i in range(self.frame_per_seg)]) for idx in start_idx]
            # if the number of frames in each segment is less than new_length
            else:
                indices = [np.sort(np.random.choice(seg, self.frame_per_seg, replace=True)) for seg in segments]
        else:
            indices = np.zeros((self.num_segments, self.frame_per_seg))

        return indices

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
        indices = np.clip(clip_offsets, 0, num_frames - 1)

        # Convert the indices to a list of lists format
        indices = [int(idx) + self.start_index for segment in indices for idx in segment]

        return indices
