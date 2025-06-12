import cv2
from pathlib import Path
import os
import av
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import random
from PIL import Image
import numpy as np
from torchvision import transforms
from itertools import islice, chain



def to_torch(x, device='cpu'):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        else:
            x = x.permute(0, 3, 1, 2)
        x = x.type(torch.FloatTensor).to(device)
    return x

def to_numpy(x):
    if torch.is_tensor(x):
        x = x.cpu().detach().permute(0, 2, 3, 1).numpy()
    return x if len(x.shape) == 4 else x[np.newaxis]

def rindex(lst, value):
    lst = list(lst)
    lst.reverse()
    i = lst.index(value)
    lst.reverse()
    return len(lst) - i - 1

def get_batch(video_iter, batch_size):
    batch = islice(video_iter, batch_size)
    batch = list(zip(*batch))
    if len(batch) == 0:
        return (None,) * 6
    images, fns, video_names, paths, is_video = batch
    video_name = video_names[0]
    fn = fns[0]
    path = paths[0]
    is_video = is_video[0]
    if is_video:
        another_video_ind = rindex(video_names, video_name) + 1
    else:
        another_video_ind = 1
    images = images[:another_video_ind]
    video_iter = chain(islice(zip(*batch), another_video_ind, None), video_iter)
    return images, video_name, fn, path, video_iter, is_video



def iter_images(path):
    def iter_file(fn):
        print(fn)
        if Path(fn).suffix in ('.png', '.jpg'):
            image = cv2.imread(fn)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255
            yield image, os.path.basename(fn), os.path.basename(fn), fn, False
        elif Path(fn).suffix in ('.y4m', '.mp4', '.mkv'):
            container = av.open(fn)
            video_stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(video_stream)):
                 image = frame.to_rgb().to_ndarray()
                 image = image / 255.
                 yield image, f'{os.path.basename(fn)}-{i}', os.path.basename(fn), fn, True
            container.close()
        else:
            print('Error path:', fn)
            yield None
    if os.path.isdir(path):
        for fn in tqdm(sorted(os.listdir(path))):
            print(fn)
            for image in iter_file(os.path.join(path, fn)):
                yield image
    else:
        for image in iter_file(path):
            yield image
            
            
def center_crop(image):
  center = image.shape[0] / 2, image.shape[1] / 2
  if center[1] < 256 or center[0] < 256:
    return cv2.resize(image, (256, 256))
  x = center[1] - 128
  y = center[0] - 128

  return image[int(y):int(y+256), int(x):int(x+256)]

class MyCustomDataset(Dataset):
    def __init__(self, 
                 path_gt,
                 device='cpu'
                ):
        
        self._items = [] 
        self._index = 0
        self.device = device
        dir_img = sorted(os.listdir(path_gt))
        img_pathes = dir_img

        for img_path in img_pathes:
          self._items.append((
            os.path.join(path_gt, img_path)
          ))
        random.shuffle(self._items)

    def __len__(self):
      return len(self._items)

    def next_data(self):
      gt_path = self._items[self._index]
      self._index += 1 
      if self._index == len(self._items):
        self._index = 0
        random.shuffle(self._items)

      image = Image.open(gt_path).convert('RGB')
      image = np.array(image).astype(np.float32) 
      image = center_crop(image)

      image = image / 255.
      image = transforms.ToTensor()(image)
      y = image.to(self.device)
      return y

    def __getitem__(self, index):
      gt_path = self._items[index]
      image = Image.open(gt_path).convert('RGB')
      image = np.array(image).astype(np.float32) 

      image = center_crop(image)

      image = image / 255.
      image = transforms.ToTensor()(image)
      y = image.to(self.device)
      return y
