import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import pandas as pd
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row, root):
        self._data = row
        self._root = root

    @property
    def path(self):
        # absolute path to the video folder
        return os.path.join(self._root, self._data['filename'])

    @property
    def num_frames(self):
        return int(self._data['frame_num'])

    @property
    def label(self):
        return int(self._data['label'])


class TSNDataSet(data.Dataset):
    def __init__(self, data_path, csv_path,
                 num_segments=3, new_length=1, modality='depth',
                 image_tmpl='MDepth-{:08d}.png', transform=None,
                 random_shift=True, test_mode=False):

        self.data_path = data_path
        self.csv = pd.read_csv(csv_path)
        self.data_num = self.csv.shape[0]
        self.num_segments = num_segments
        self.new_length = new_length  # number of frames for each segmentation after sampling
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_class = len(set(self.csv['action_id']))

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'depth' or self.modality == 'depthDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(self.csv.iloc[i], self.data_path) for i in range(self.data_num)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        # split the video into segments
        # each segment contains average_duration frames
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            # 1. init the offset for each segment as the first frame of it
            # 2. assign a random int value (0 to average_duration) to the offset
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
