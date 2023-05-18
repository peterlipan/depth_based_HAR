import cv2
import random
import numpy as np
import numbers
import math
import torch
from PIL import Image, ImageOps, ImageChops
import torchvision.transforms as T


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = T.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomRotate(object):
    """
    Randomly rotates the given PIL.Image group with a given probability
    and a range of rotation angles.
    """

    def __init__(self, prob=0.5, degrees=15):
        """
        Args:
            prob (float): probability of applying the transformation
            degrees (float or int): range of rotation angles, in degrees.
                If a float, the range will be (-degrees, degrees).
                If an int, the range will be (-degrees, degrees).
        """
        self.prob = prob
        self.degrees = degrees

    def __call__(self, img_group):
        """
        Args:
            img_group (list): list of PIL.Image objects to be rotated.

        Returns:
            list: list of rotated PIL.Image objects.
        """
        if random.random() < self.prob:
            angle = random.uniform(-self.degrees, self.degrees)
            ret = [img.rotate(angle, resample=Image.BICUBIC) for img in img_group]
            return ret
        return img_group


class GroupRandomHorizontalFlip(object):
    """
    Randomly horizontally flips the given PIL.Image with a given probability
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group


class GroupRandomVerticalFlip(object):
    """
    Randomly vertically flips the given PIL.Image with a given probability
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img_group):
        v = random.random()
        if v < self.prob:
            ret = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in img_group]
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = T.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if isinstance(img_group[0], Image.Image):
            if img_group[0].mode == 'L':
                return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
            elif img_group[0].mode == 'RGB':
                if self.roll:
                    return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
                else:
                    return np.concatenate(img_group, axis=2)
        elif isinstance(img_group[0], np.ndarray):
            return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=255.):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(self.div) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data


class GroupImageDiff(object):
    """
    Calculates pixel-wise difference between consecutive frames in a given PIL.Image group,
    except for the last one.
    """

    def __init__(self, frame_per_seg, absolute=False):
        """
        Args:
            frame_per_seg (int): number of images in each segmentation
        """
        self.frame_per_seg = frame_per_seg
        self.absolute = absolute

    def __call__(self, img_group):
        """
        Args:
            img_group (list): list of PIL.Image objects.

        Returns:
            list: list of PIL.Image objects, with pixel-wise differences between
            consecutive frames in each segmentation.
        """
        num_images = len(img_group)
        assert num_images % self.frame_per_seg == 0, f"Expected number of images to be divisible by {self.frame_per_seg}"

        diff_group = []
        for i in range(0, num_images, self.frame_per_seg):
            seg_images = [np.array(img, dtype=int) for img in img_group[i:i + self.frame_per_seg]]

            # Calculate pixel-wise difference between consecutive frames in the segmentation
            for j in range(1, self.frame_per_seg):
                if self.absolute:
                    diff = np.abs(seg_images[j] - seg_images[j - 1])
                else:
                    diff = seg_images[j] - seg_images[j - 1]
                diff_group.append(diff)

        return diff_group


class GroupImageGradient(object):
    """
    Calculates the spectrum of gradient for a list of PIL.Image objects.
    """
    @staticmethod
    def _cal_gradient_mag(img):
        # calculate the normalized spectrum of the gradient of the image
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        if np.max(grad_mag) != np.min(grad_mag):
            grad_mag_norm = (grad_mag - np.min(grad_mag)) / (np.max(grad_mag) - np.min(grad_mag))
        elif np.max(grad_mag) != 0:
            grad_mag_norm = grad_mag / np.max(grad_mag)
        else:
            grad_mag_norm = grad_mag

        return grad_mag_norm

    def __call__(self, img_group):
        """
        Args:
            img_group (list): list of PIL.Image objects.

        Returns:
            list: list of PIL.Image objects, with each image showing the
                  magnitude of gradient for the corresponding image in img_list.
        """
        array_list = [np.array(img) for img in img_group]

        return [self._cal_gradient_mag(img) for img in array_list]


class GroupM3d(object):
    """
    projections of 3dv
    """

    def __init__(self, frame_per_seg):
        """
        Args:
            frame_per_seg (int): number of images in each segmentation
        """
        self.frame_per_seg = frame_per_seg

    def __call__(self, img_group):
        """
        Args:
            img_group (list): list of PIL.Image objects.

        Returns:
            list: list of PIL.Image objects, with pixel-wise differences between
            consecutive frames in each segmentation.
        """
        num_images = len(img_group)
        assert num_images % self.frame_per_seg == 0, \
            f"Expected number of images to be divisible by {self.frame_per_seg}"

        m3d_group = []
        for i in range(0, num_images, self.frame_per_seg):
            seg_images = [np.array(img.convert('L'), dtype=int) for img in img_group[i:i + self.frame_per_seg]]
            front_view = np.min(seg_images, axis=0)
            projection = np.mean(seg_images, aixs=0)
            back_view = np.max(seg_images, axis=0)
            m3d = np.stack([front_view, projection, back_view], axis=2)
            m3d_group.append(m3d)

        return m3d_group


class Transforms:
    def __init__(self, modality, input_size, frame_per_seg):
        self.modality = modality
        self.input_size = input_size
        self.frame_per_seg = frame_per_seg

        scale_size = 256

        # same normalization for ImageNet pretrained models
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        self.test_transforms = T.Compose([GroupScale(scale_size), GroupCenterCrop(input_size),
                                          Stack(), ToTorchFormatTensor(div=True), normalize])

        if modality == 'depth':
            self.train_transforms = T.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(prob=0.5),
                                               GroupRandomVerticalFlip(prob=0.5),
                                               Stack(roll=False), ToTorchFormatTensor(div=255.), normalize])
        elif modality == 'depthDiff':
            self.train_transforms = T.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(prob=0.5),
                                               GroupRandomVerticalFlip(prob=0.5),
                                               GroupImageDiff(frame_per_seg=self.frame_per_seg),
                                               Stack(roll=False), ToTorchFormatTensor(div=255.)])

            self.test_transforms = T.Compose([GroupScale(scale_size), GroupCenterCrop(input_size),
                                              GroupImageDiff(frame_per_seg=self.frame_per_seg),
                                              Stack(roll=False), ToTorchFormatTensor(div=255.)])
        elif modality == 'depthGrad':
            self.train_transforms = T.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(prob=0.5),
                                               GroupRandomVerticalFlip(prob=0.5),
                                               GroupImageGradient(),
                                               Stack(roll=False), ToTorchFormatTensor(div=255.)])

            self.test_transforms = T.Compose([GroupScale(scale_size), GroupCenterCrop(input_size),
                                              GroupImageGradient(),
                                              Stack(roll=False), ToTorchFormatTensor(div=255.)])

        elif modality == 'm3d':
            self.train_transforms = T.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(prob=0.5),
                                               GroupRandomVerticalFlip(prob=0.5),
                                               GroupM3d(frame_per_seg=self.frame_per_seg),
                                               Stack(roll=False), ToTorchFormatTensor(div=255.)])

            self.test_transforms = T.Compose([GroupScale(scale_size), GroupCenterCrop(input_size),
                                              GroupM3d(frame_per_seg=self.frame_per_seg),
                                              Stack(roll=False), ToTorchFormatTensor(div=255.)])


if __name__ == "__main__":
    transforms = Transforms(modality='depth', input_size=224, frame_per_seg=1)
    train_transforms = transforms.train_transforms
    test_transforms = transforms.test_transforms

    im = Image.open('/mnt/ssd/li/NTU_RGBD_60/nturgb+d_depth_masked/S001C001P001R001A001/MDepth-00000001.png').convert(
        'RGB')
    print('raw: ', np.array(im).shape)

    color_group = [im] * 3
    rst1 = train_transforms(color_group)
    print('Train trans: ', rst1.shape)

    rst2 = test_transforms(color_group)
    print('Test trans: ', rst2)
