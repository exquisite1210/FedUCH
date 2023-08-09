import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        # self.fc2 = nn.Linear(4096, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
       # torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std= 0.1)  
        # self.Imgmodel = ImgModule()


    def forward(self, x):
        # x = self.Imgmodel(x)
        x = x.view(x.size(0), -1)

        feat1 = self.relu(self.fc1(x))
        #feat1 = feat1 + self.relu(self.fc2(self.dropout(feat1)))
        # feat2 = self.relu(self.fc2(feat1))
        hid = self.fc_encode(self.dropout(feat1))
        code = torch.tanh(self.alpha * hid)

        return feat1, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len, image_size=4096):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace = True)
        torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std= 0.3)  
              
    def forward(self, x):

        feat = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat))
        code = torch.tanh(self.alpha * hid)

        return feat, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)



# class FC_module(nn.Module):
#     def __init__(self,hidden=64,output_dir=64):
#         super(FC_module,self).__init__()
#         self.fc = nn.Linear(hidden, output_dir)
#         # self.output_dir = output_dir
#         # torch.nn.init.normal(self.fc.weight, mean=0.0, std= 0.3)  

    
#     def forward(self,x,normalize_feat=True):
#         output = self.fc(x)
#         # if normalize_feat:
#         #     output = torch.matmul(F.normalize(x),F.normalize(self.fc).t())
#         # else:
#         #     output = torch.matmul(x,F.normalize(self.fc).t())
#         return output





class ImgModule(nn.Module):
    def __init__(self, pretrain_model=None):
        super(ImgModule, self).__init__()
        self.features = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1
            nn.ReLU(inplace=True),
            # 2 norm1
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 5 relu2
            nn.ReLU(inplace=True),
            # 6 norm2
            nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 9 relu3
            nn.ReLU(inplace=True),
            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 11 relu4
            nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 13 relu5
            nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 16 relu6
            nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 18 relu7
            nn.ReLU(inplace=True),
        )
        # fc8
        self.mean = torch.zeros(3, 224, 224)
        self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))

    def forward(self, x):
        x = x - self.mean.cuda()
        x = self.features(x)
        x = x.squeeze()
        return x
