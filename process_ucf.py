
import time
import numpy as np
import skimage
import imageio
import os
import os.path as osp
import pylab

from PIL import Image

def generate_frames(video_name, video_path, target_dir):
    
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    
    vid = imageio.get_reader(video_path, 'ffmpeg')
    
    for num, im in enumerate(vid):
        frame_name = f'frame_{num}.jpg'
        target_path = osp.join(target_dir, frame_name)
        image = np.array(skimage.img_as_float(im)).astype(np.float32)
        imageio.imwrite(target_path, image)
    
    print(f'Row:{video_path} -> Target:{target_dir}')


def main():
    """Separate video into frames"""
    
    row_ucf_root = '/home/dcooo/dataset/UCF101/UCF-101-row'
    target_root = '/home/dcooo/dataset/UCF101/UCF-101-frames'
    
    for idx, label in enumerate(os.listdir(row_ucf_root)):
        label_row = osp.join(row_ucf_root, label)
        label_frame_dir = osp.join(target_root, label)
        if not osp.exists(label_frame_dir):
            os.mkdir(label_frame_dir)
        for sample in os.listdir(label_row):
            sample_row_path = osp.join(label_row, sample)
            frame_target_dir = osp.join(label_frame_dir, sample.split('.')[0])
            if osp.exists(frame_target_dir):
                print(f'{frame_target_dir} Finished')
            else:
                generate_frames(sample, sample_row_path, frame_target_dir)
            
        
        print(f'{idx + 1}/101')
        


if __name__ == '__main__':
    
    main()
