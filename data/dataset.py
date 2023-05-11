import os
import pandas as pd
from PIL import Image
import numpy as np
import torch.utils.data as data
from skimage import io
from .samplers import SegmentedSample, HierarchySample


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
                 num_segments=3, frame_per_seg=1, margin=3,modality='depth',
                 image_tmpl='MDepth-{:08d}.ppm', transform=None,
                 test_mode=False, start_index=1):

        self.data_path = data_path
        self.csv = pd.read_csv(csv_path)
        self.data_num = self.csv.shape[0]
        self.num_segments = num_segments
        self.frame_per_seg = frame_per_seg  # number of frames for each segmentation after sampling
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.test_mode = test_mode
        self.num_class = len(set(self.csv['action_id']))

        self.sampler = HierarchySample(frame_per_seg=self.frame_per_seg, num_segments=num_segments,
                                       margin=margin, test_mode=test_mode, start_index=start_index)

        self._parse_list()

    def _load_image(self, directory, idx):
        # load the ppm file
        coll = io.ImageCollection(os.path.join(directory, self.image_tmpl.format(idx)))
        # TODO: Make it uint16 instead of uint8
        # PIL Image RGB mode will transform uint16 to uint8
        int8_img = (np.squeeze(np.array(coll)) * 255. / 65535.).astype(np.uint8)

        return Image.fromarray(int8_img).convert('RGB')

    def _parse_list(self):
        self.video_list = [VideoRecord(self.csv.iloc[i], self.data_path) for i in range(self.data_num)]

    def __getitem__(self, index):

        record = self.video_list[index]
        segment_indices = self.sampler(record.num_frames)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = [self._load_image(record.path, p) for p in indices]
        img_augmented = self.transform(images)

        return img_augmented, record.label

    def __len__(self):
        return len(self.video_list)
