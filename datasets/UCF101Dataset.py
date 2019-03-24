import numpy as np
import os.path as osp
import os
import torch.utils.data as data

from PIL import Image


def center_sample_from_frames(frames_dir, seg_length):
    """Sample from original frames in the same time gap"""
    print(frames_dir)
    frames_path = [osp.join(frames_dir, f) for f in os.listdir(frames_dir)]
    print(len(frames_path))
    if osp.exists(frames_dir):
        # Count number of frames
        frames_path = [osp.join(frames_dir, f) for f in os.listdir(frames_dir)]
        length = len(frames_path)
        if length >= seg_length:
            # Get center frames
            # [len_frames // 2 - seg_length // 2] -> [len_fraMES // 2 + seg_length // 2]
            start = length // 2 - seg_length // 2
            return frames_path[start: start + seg_length]
        else:
            raise ValueError('There are shorter frames than expected')
    else:
        raise ValueError('Please specific a correct file path')


def load_list(path):
    """Load list from file"""
    ls = []
    if isinstance(path, list):
        for p in path:
            with open(p) as f:
                for l in f:
                    ls.append(l.strip())
    else:
        with open(path) as f:
            for l in f:
                ls.append(l.strip())
    return ls


class UCF101Dataset(data.Dataset):
    
    def __init__(self, data_root, list_path, seg_len, crop_size, transform=None):
        self.data_root = data_root
        self.seg_len = seg_len
        self.transform = transform
        self.crop_size = crop_size
        self.videos_path_list = load_list(list_path)
    
    def __getitem__(self, index):
        # item: path/to/video <label>
        item = self.videos_path_list[index]
        video_path, label = item.split(' ')
        all_frames_path = os.path.join(self.data_root, video_path.split('.')[0])
        label = int(label)
        
        # Sample from all frames
        origin_frames = center_sample_from_frames(all_frames_path, self.seg_len)
        item = []
        # Apply transforms in every frame
        for f in origin_frames:
            img = Image.open(f)
            img = img.convert('RGB')
            img.crop()
            # img = self.transform(img)
            item.append(img)
        
        # Return data and label
        return item, label
    
    def __len__(self):
        return len(self.videos_path_list)


if __name__ == '__main__':
    train_list = '/home/dcooo/dataset/UCF101/DatasetSplit/trainlist01.txt'
    d = UCF101Dataset('/home/dcooo/dataset/UCF101/UCF-101-frames', train_list, 64, [224, 224])
    train_loader = data.DataLoader(d, 20, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    # train_loader.
    for i, b in enumerate(train_loader):
        batch_input = b[0]
        batch_label = b[1]
        print(batch_input)
        # print(batch)
        # print(label)
        break
