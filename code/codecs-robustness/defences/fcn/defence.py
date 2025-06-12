import torch
import torch.nn as nn
import subprocess

class FCN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type='batch', act_type='selu'):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv2d(input_nc, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 3, 3, 1, 1)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(16)
            self.norm2 = nn.BatchNorm2d(8)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(16)
            self.norm2 = nn.InstanceNorm2d(8)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, inputs):
        # Encoder
        # Convolution layers:
        # input is (nc) x 512 x 1024
        e1 = self.conv1(inputs)
        # print(f'e1 {e1.shape}')
        # state size is (ngf) x 256 x 512
        e2 = self.norm2(self.conv2(self.leaky_relu(e1)))
        # print(f'e2 {e2.shape}')
        # state size is (ngf x 2) x 128 x 256
        e3 = self.conv3(self.leaky_relu(e2))
        # print(f'e3 {e3.shape}')

        output = self.tanh(e3)
        return output

class Defense:
    def __init__(self, fcn_path='fcn_mse.pth'):
        #subprocess.run('./defence/setup.sh')
        self.defence_name = 'fcn'
        self.defence_model = FCN(3, 3, 64, norm_type='instance', act_type='relu')

        checkpoint = torch.load(fcn_path, map_location='cpu')
        self.defence_model.load_state_dict(checkpoint['model'])
        self.defence_model.eval()

    def __call__(self, image):
        self.defence_model.to(image.device)
        return self.defence_model(image).clamp(0.0, 1.0)
   
