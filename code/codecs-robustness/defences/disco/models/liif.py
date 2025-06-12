import torch
import torch.nn as nn
import torch.nn.functional as F

import defence.models as models
from defence.models import register
from defence.utils import make_coord
import random
import numpy as np

@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, feat_unfold_kernel=None):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                self.feat_unfold_kernel = 3 if feat_unfold_kernel is None else feat_unfold_kernel
                self.multiplier = self.feat_unfold_kernel*self.feat_unfold_kernel
                imnet_in_dim *= self.multiplier
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def gen_latent(self, inp):
        latent = self.encoder.forward_latent(inp)
        return latent

    def gen_feat_and_latent(self, inp):
        self.feat, latent = self.encoder.forward_feat_and_latent(inp)
        return self.feat, latent


    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret
        
        if self.feat_unfold:
            if self.feat_unfold_kernel == 3:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=1).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])
            elif self.feat_unfold_kernel == 5:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=2).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])
            elif self.feat_unfold_kernel == 7:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=3).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])



        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret


    def query_rgb_test(self, coord, cell=None):
        feat = self.feat
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            if self.feat_unfold_kernel == 3:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=1).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])
            elif self.feat_unfold_kernel == 5:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=2).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])
            elif self.feat_unfold_kernel == 7:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=3).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])



        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                #print("========")
                #print(inp.shape)
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    #print(rel_cell.shape)
                    inp = torch.cat([inp, rel_cell], dim=-1)
                #print(inp.shape)
                #print("========")

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        print(ret.shape)
        return ret



    def query_rgb_rand(self, coord, cell=None):
        feat = self.feat
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        
        if self.feat_unfold:
            if True: #np.random.rand() > 0.5:
                feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            else:
                #feat = F.unfold(feat, 5, padding=2).view(feat.shape[0], feat.shape[1], 25, feat.shape[2], feat.shape[3])     
                #indices = torch.LongTensor(np.sort(np.random.choice(25,9, replace=True))).cuda() 
                '''
                feat = F.unfold(feat, 7, padding=3).view(feat.shape[0], feat.shape[1], 49, feat.shape[2], feat.shape[3])     
                indices = torch.LongTensor(np.sort(np.random.choice(49,9, replace=True))).cuda() 
                feat = torch.index_select(feat, 2, indices)
                feat = feat.view(feat.shape[0], -1, feat.shape[3], feat.shape[4])
                '''
                ''' 
                feat_tmp = feat.repeat(1, 9, 1, 1).clone()
                feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
                weight = 0.4 #np.random.rand()
                feat = weight*feat_tmp + (1-weight)*feat                
                '''

                feat = feat.repeat(1, 9, 1, 1).clone()
                '''    
                feat_dim = feat.shape[1]
                feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1], 9, feat.shape[2], feat.shape[3])
                #feat=torch.mean(feat, 2, True)
                feat=torch.amax(feat, 2, True)
                feat = feat.repeat(1, 1, 9, 1, 1)
                feat = feat.view(feat.shape[0], -1, feat.shape[3], feat.shape[4])
                '''
        local_ensemble_prob = 1 #np.random.rand()
        if self.local_ensemble and local_ensemble_prob > 0.5:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode :
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble and local_ensemble_prob > 0.5:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret



    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

    def forward_with_latent(self, inp, coord, cell):
        _, latent = self.gen_feat_and_latent(inp)
        return self.query_rgb(coord, cell), latent
