import os
import os.path
import pandas as pd
from PIL import Image
import numpy as np
import torch.utils.data as data
from skimage import io
from .samplers import SegmentedSample


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
                 image_tmpl='MDepth-{:08d}.ppm', transform=None,
                 test_mode=False, start_index=1):

        self.data_path = data_path
        self.csv = pd.read_csv(csv_path)
        self.data_num = self.csv.shape[0]
        self.num_segments = num_segments
        self.new_length = new_length  # number of frames for each segmentation after sampling
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.test_mode = test_mode
        self.num_class = len(set(self.csv['action_id']))
        self.sampler = SegmentedSample(new_length, num_segments, test_mode, start_index)

        if self.modality == 'depthDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'depth' or self.modality == 'depthDiff':
            coll = io.ImageCollection(os.path.join(directory, self.image_tmpl.format(idx)))
            # rescale and to numpy float array
            arr = (np.squeeze(np.array(coll)) * 255. / 65535.).astype(np.uint8)
            return [Image.fromarray(arr).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(self.csv.iloc[i], self.data_path) for i in range(self.data_num)]

    def __getitem__(self, index):

        record = self.video_list[index]
        segment_indices = self.sampler(record.num_frames)

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
