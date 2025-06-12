import sys
sys.path.append("/data8/john/inr_adv_defense/liif")
import models as inr_models
from utils import make_coord
from tqdm import tqdm
import json
import torch
from torchvision import transforms
import random
import numpy as np

class INR(object):
    def __init__(self, device, pretrain_inr_path, height=299, width=299):
        self.device = device
        #self.inr_model = inr_models.make(torch.load(pretrain_inr_path)['model'], load_sd=True).to(self.device)
        self.inr_model = []
        for idx in range(len(pretrain_inr_path)):
            self.inr_model.append(inr_models.make(torch.load(pretrain_inr_path[idx])['model'], load_sd=True).to(self.device))

        self.height = height
        self.width = width

        self.coord = make_coord((self.height, self.width)).to(self.device)
        self.cell = torch.ones_like(self.coord)
        self.cell[:, 0] *= 2 / self.height
        self.cell[:, 1] *= 2 / self.width


    def batched_predict(self, inp, coord, cell, bsize):
        with torch.no_grad():
            #self.inr_model.gen_feat(inp)
            for idx in range(len(self.inr_model)):
                self.inr_model[idx].gen_feat(inp)
           
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                idx = random.randint(0, len(self.inr_model)-1)
                pred = self.inr_model[idx].query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                #pred = self.inr_model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred

    def forward(self,x):
        lst_img = []
        for img in x:
           img_tensor = img.unsqueeze(0)
           inr_output = self.batched_predict(((img_tensor - 0.5) / 0.5), self.coord.unsqueeze(0), self.cell.unsqueeze(0), bsize=30000)[0]
           inr_output = (inr_output * 0.5 + 0.5).clamp(0, 1).view(self.height, self.width, 3).permute(2, 0, 1)
           lst_img.append(inr_output)
        
        return x.new_tensor(torch.stack(lst_img))



class INRAdaptiveAttack(object):
    def __init__(self, device, pretrain_inr_path, height=299, width=299):
        self.device = device
        #self.inr_model = inr_models.make(torch.load(pretrain_inr_path)['model'], load_sd=True).to(self.device)
        self.inr_model = []
        for idx in range(len(pretrain_inr_path)):
            self.inr_model.append(inr_models.make(torch.load(pretrain_inr_path[idx])['model'], load_sd=True).to(self.device))

        self.height = height
        self.width = width

        self.coord = make_coord((self.height, self.width)).to(self.device)
        self.cell = torch.ones_like(self.coord)
        self.cell[:, 0] *= 2 / self.height
        self.cell[:, 1] *= 2 / self.width


    def batched_predict(self, inp, coord, cell, bsize):
        #self.inr_model.gen_feat(inp)
        for idx in range(len(self.inr_model)):
            self.inr_model[idx].gen_feat(inp)
           
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            idx = random.randint(0, len(self.inr_model)-1)
            pred = self.inr_model[idx].query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            #pred = self.inr_model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
        return pred

    def forward(self,x):
        lst_img = []
        for img in x:
           img_tensor = img.unsqueeze(0)
           inr_output = self.batched_predict(((img_tensor - 0.5) / 0.5), self.coord.unsqueeze(0), self.cell.unsqueeze(0), bsize=30000)[0]
           inr_output = (inr_output * 0.5 + 0.5).clamp(0, 1).view(self.height, self.width, 3).permute(2, 0, 1)
           lst_img.append(inr_output)
        
        return torch.stack(lst_img)


class INRRandScale(object):
    def __init__(self, device, pretrain_inr_path, height=299, width=299):
        self.device = device
        #self.inr_model = inr_models.make(torch.load(pretrain_inr_path)['model'], load_sd=True).to(self.device)
        self.inr_model = []
        for idx in range(len(pretrain_inr_path)):
            self.inr_model.append(inr_models.make(torch.load(pretrain_inr_path[idx])['model'], load_sd=True).to(self.device))
            #self.inr_model[idx].feat_unfold = False
            #self.inr_model[idx].cell_decode = False
            #self.inr_model[idx].local_ensemble = False
        self.height = height
        self.width = width

    def batched_predict(self, inp, coord, cell, bsize):
        with torch.no_grad():
            #self.inr_model.gen_feat(inp)
            for idx in range(len(self.inr_model)):
                self.inr_model[idx].gen_feat(inp)
           
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                idx = random.randint(0, len(self.inr_model)-1)
                pred = self.inr_model[idx].query_rgb_rand(coord[:, ql: qr, :], cell[:, ql: qr, :])
                #pred = self.inr_model[idx].query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred

    def forward(self,x):

        lst_img = []
        for img in x:

           scale = 1 #2
           height = self.height*scale
           width = self.width*scale
           coord = make_coord((height, width)).to(self.device)
           cell = torch.ones_like(coord)
           cell[:, 0] *= 2 / height
           cell[:, 1] *= 2 / width
          
           img_tensor = img.unsqueeze(0)
           inr_output = self.batched_predict(((img_tensor - 0.5) / 0.5), coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
           inr_output = (inr_output * 0.5 + 0.5).clamp(0, 1).view(height, width, 3).permute(2, 0, 1)
           
           sample_coordx = list(np.random.randint(scale, size=int(height/scale)) + np.arange(0, height, scale, dtype=int)) 
           sample_coordy = list(np.random.randint(scale, size=int(width/scale)) + np.arange(0, width, scale, dtype=int)) 
           inr_output = inr_output[:, sample_coordx, :][:, : , sample_coordy]

           lst_img.append(inr_output)
           
        return x.new_tensor(torch.stack(lst_img))
        #return torch.stack(lst_img)
 
