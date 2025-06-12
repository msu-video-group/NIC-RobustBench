from torchvision import transforms
import torch
import cv2
import csv
import numpy as np
from read_dataset import to_numpy, to_torch, iter_images, get_batch
from metrics import PSNR, SSIM, MSE, L_inf_dist, MAE
import av

def predict(img1, img2=None, model=None, device='cpu'):
    model.to(device)
    if not torch.is_tensor(img1):
        if len(img1.shape) == 3:
            img1 = img1[np.newaxis]
        img1 = torch.from_numpy(img1).permute(0, 3, 1, 2)
    img1 = img1.type(torch.FloatTensor).to(device)
    if img2 is not None:
        if not torch.is_tensor(img2):
            if len(img2.shape) == 3:
                img2 = img2[np.newaxis]
            img2 = torch.from_numpy(img2).permute(0, 3, 1, 2)
        img2 = img2.type(torch.FloatTensor).to(device)
        
        res = model(img1, img2)
    else:
        res = model(img1, inference=False)
    return res.item() if res is not None else None

class Encoder:
    def __init__(self, fn, codec):
        self.output = av.open(fn, 'w')
        self.stream = self.output.add_stream(codec, rate=24)
        self.fn = fn
        self.codec = codec
        
    def add_frames(self, imgs):
        imgs = (imgs * 255).astype(np.uint8)
        for i in range(len(imgs)):
            img = imgs[i]
            self.stream.height = img.shape[0]
            self.stream.width = img.shape[1]
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            packet = self.stream.encode(frame)
            self.output.mux(packet)
        
    def close(self):
        packet = self.stream.encode(None)
        self.output.mux(packet)
        self.output.close()
        



def eval_encoded_video(model, encoded_video_path, orig_video_path, is_fr, batch_size=1, device='cpu'):
    encoded_video_iter = iter_images(encoded_video_path)
    orig_video_iter = iter_images(orig_video_path)
    i = 0
    while True:
        encoded_images, _, _, _, encoded_video_iter, _ = get_batch(encoded_video_iter, batch_size)
        if encoded_images is None:
            break
        encoded_images = np.stack(encoded_images)
        orig_images, _, _, _, orig_video_iter, _ = get_batch(orig_video_iter, batch_size)
        orig_images = np.stack(orig_images)
        if is_fr:
            encoded_metric = predict(orig_images, encoded_images, model=model, device=device)
        else:
            encoded_metric = predict(encoded_images, model=model, device=device)
        psnr = PSNR(orig_images, encoded_images)
        ssim = SSIM(orig_images, encoded_images)
        mse = MSE(orig_images, encoded_images)
        linf = L_inf_dist(orig_images, encoded_images)
        mae = MAE(orig_images, encoded_images)
        yield encoded_metric, i, i + len(encoded_images), psnr, ssim, mse, linf, mae
        i += batch_size

    
def compress(img, q, return_torch=False):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    np_batch = to_numpy(img)
    if len(np_batch.shape) == 3:
        np_batch = np_batch[np.newaxis]
    jpeg_batch = np.empty(np_batch.shape)
    for i in range(len(np_batch)):
        result, encimg = cv2.imencode('.jpg', np_batch[i] * 255, encode_param)
        jpeg_batch[i] = cv2.imdecode(encimg, 1) / 255
    return torch.nan_to_num(to_torch(jpeg_batch), nan=0) if return_torch else np.nan_to_num(jpeg_batch, nan=0)


def jpeg_generator(img_gen, jpeg_quality):
    if jpeg_quality is None:
        yield img_gen, None
    else:
        for q in jpeg_quality:
            jpeg_image = compress(img_gen, q, return_torch=True)
            yield img_gen, jpeg_image
            
def write_log(log_file_path, test_dataset, mean_time, preset=-1):
    if log_file_path is None:
        return
    with open(log_file_path, 'a', newline='') as log_file:
        fieldnames = ['test_dataset', 'mean_time_ms', 'attack_preset']
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if log_file.tell() == 0:
            writer.writeheader()
        writer.writerow({
            'test_dataset': test_dataset,
            'mean_time_ms': mean_time,
            'attack_preset':preset
            })
def create_tensor(video_iter, device='cpu'):
    for i, (image, fn) in enumerate(video_iter):
        image = transforms.ToTensor()(image.astype(np.float32))
        image = image.unsqueeze_(0)
        image = image.to(device)
        yield image
        
        

       
            

