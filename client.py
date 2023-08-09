import os.path as osp

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from metric import compress, calculate_top_map
import copy


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        
        label = self.labels[index]
        return img, text, label, index

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count

def calc_dis(query_L, retrieval_L, query_dis, top_k=32):
    num_query = query_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = query_dis[iter]
        ind = np.argsort(hamm)[:top_k]
        gnd = gnd[ind]
        tsum = np.int(np.sum(gnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map



class Client(object):
    def __init__(self,client_id,model_img,model_txt,args,dataloader,datatrain,device):
        self.id = client_id

        self.args = args
        
        self.device = device

        self.list_loss_every_global_epoch = []


        self.global_imgs, self.global_txts, self.global_labs = datatrain
        self.local_sample_number = len(self.global_imgs)

        self.global_imgs = F.normalize(torch.Tensor(self.global_imgs)).cuda()
        self.global_txts = F.normalize(torch.Tensor(self.global_txts)).cuda()
        self.global_labs = torch.Tensor(self.global_labs).cuda()
        self.scale = round(0.9*self.local_sample_number)
        self.knn_number = round(0.45*self.scale)
        
        self.gs, self.sa, self.ni = self.cal_similarity(self.global_imgs, self.global_txts)


        self.train_loader = dataloader['train']
        self.val_loader = dataloader['validation']
        self.test_loader = dataloader['query']
        self.database_loader = dataloader['database']
        self.databasev_loader = dataloader['databasev']

        self.CodeNet_I = copy.deepcopy(model_img)
        self.CodeNet_T = copy.deepcopy(model_txt)

        self.args_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum,
                                     weight_decay=self.args.weight_decay)
        self.args_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)

    
        self.best = 0.0
        self.con_criterion = nn.CosineSimilarity(dim=1)
        self.temperature = 1.0
        self.KL = DistillKL(T=1.5)

    def get_sample_number(self):
        return self.local_sample_number
    
    def get_models(self):
        return self.CodeNet_I,self.CodeNet_T

    def train(self, epoch,model_img,model_txt):
        self.global_model_imgs.load_state_dict(model_img)
        self.global_model_txts.load_state_dict(model_txt)
        self.pre_model_img.load_state_dict(self.CodeNet_I.cpu().state_dict())
        self.pre_model_txt.load_state_dict(self.CodeNet_T.cpu().state_dict())
        
        self.global_model_imgs.cuda()
        self.global_model_txts.cuda()
        self.pre_model_img.cuda()
        self.pre_model_txt.cuda()
        
        self.CodeNet_I.load_state_dict(model_img)
        self.CodeNet_T.load_state_dict(model_txt)

        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()

        top_mAP = 0.0
        num = 0.0

        for epoch in range(self.args.local_epochs):
            for idx, (img, txt, labels, index) in enumerate(self.train_loader):
                img = Variable(img.cuda())
                txt = Variable(torch.FloatTensor(txt.numpy()).cuda())
                batch_size = img.size(0)
                I = torch.eye(batch_size).cuda()

                _, code_I = self.CodeNet_I(img)
                _, code_T = self.CodeNet_T(txt)

                S = self.gs[index, :][:, index].cuda()

                loss, all_los = self.loss_cal(code_I, code_T, S, I)
                self.args_I.zero_grad()
                self.args_T.zero_grad()
                loss.backward(retain_graph=True)
                self.args_I.step()
                self.args_T.step()
                
                _, code_I = self.CodeNet_I(img)
                _, code_T = self.CodeNet_T(txt)

                loss_i, _ = self.loss_cal(code_I, code_T.sign().detach(), S, I)

                
                self.args_I.zero_grad()
                loss_i.backward(retain_graph=True)
                self.args_I.step()
                

                loss_t, _ = self.loss_cal(code_I.sign().detach(), code_T, S, I)

                self.args_T.zero_grad()
                loss_t.backward(retain_graph=True)
                self.args_T.step()


                


                
                

                loss1, loss2, loss3, loss4, loss5, loss6 = all_los


                top_mAP += calc_dis(labels.cpu().numpy(), labels.cpu().numpy(), -S.cpu().numpy())
                
                num += 1.
                if (idx + 1) % (len(self.train_loader)) == 0:
                    print(
                        'Epoch [%d/%d]'
                        'Loss1: %.4f Loss2: %.4f Loss3: %.4f '
                        'Loss4: %.4f '
                        'Loss5: %.4f Loss6: %.4f '
                        'Total Loss: %.4f '
                        # 'Loss7: %.4f Loss8: %.4f '
                        'mAP: %.4f'
                        % (
                            epoch + 1, self.args.local_epochs, 
                            loss1.mean().item(), loss2.mean().item(), loss3.mean().item(),
                            loss4.item(),
                            code_T.abs().mean().item(), 
                            code_I.abs().mean().item(),
                            loss.item(),
                            top_mAP / num))

        
        
        return self.CodeNet_I.state_dict(),self.CodeNet_T.state_dict()
  
    def cal_similarity(self, F_I, F_T):
        batch_size = F_I.size(0)
        
        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())

        S_pair = self.args.a1 * S_T + (1 - self.args.a1) * S_I
        
        pro = F_T.mm(F_T.t()) * self.args.a1 + F_I.mm(F_I.t()) * (1. - self.args.a1)

        size = batch_size
        top_size = self.knn_number
        m, n1 = pro.sort()
        pro[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(-1)] = 0.
        pro[torch.arange(size).view(-1), n1[:, -1:].contiguous().view(-1)] = 0.
        pro = pro / pro.sum(1).view(-1, 1)
        pro_dis = pro.mm(pro.t())
        pro_dis = pro_dis * self.scale
        # pdb.set_trace()
        S = (S_pair * (1 - self.args.a2) + pro_dis * self.args.a2)
        S = S * 2.0 - 1
        
        return S, S_pair, pro_dis
    
    def loss_cal(self, code_I, code_T, S, I):
        B_I = F.normalize(code_I)
        B_T = F.normalize(code_T)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())
        # pdb.set_trace()
        diagonal = BI_BT.diagonal()
        all_1 = torch.rand((BT_BT.size(0))).fill_(1).cuda()
        loss_pair = F.mse_loss(diagonal, self.args.K * all_1)

        loss_dis_1 = F.mse_loss(BT_BT * (1-I), S * (1-I))
        loss_dis_2 = F.mse_loss(BI_BT * (1-I), S * (1-I))
        loss_dis_3 = F.mse_loss(BI_BI * (1-I), S * (1-I))

        loss_cons = F.mse_loss(BI_BT, BI_BI) + \
                    F.mse_loss(BI_BT, BT_BT) + \
                    F.mse_loss(BI_BI, BT_BT) + \
                    F.mse_loss(BI_BT, BI_BT.t())

        loss = loss_pair + (loss_dis_1 + loss_dis_2 + loss_dis_3) * self.args.dw \
               + loss_cons * self.args.cw
        loss = loss

        return loss, (loss_pair, loss_dis_1, loss_dis_2, loss_dis_3, loss_cons, loss_cons)
    
    def local_validate(self,epoch):
            databasev_loader = self.database_loader
            val_loader = self.test_loader
            self.load_checkpoints()
            self.CodeNet_I.eval().cuda()
            self.CodeNet_T.eval().cuda()
            if self.args.EVAL == False:
                re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(databasev_loader, val_loader, self.CodeNet_I,self.CodeNet_T)
                MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
                MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
               
                # print('--------------------Evaluation: Calculate top MAP-------------------')
                print('MAP@50 of Image to Text: %.5f, MAP of Text to Image: %.5f' % (MAP_I2T, MAP_T2I))
    
            if self.args.EVAL:
                re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(databasev_loader, val_loader, self.CodeNet_I,
                                                                    self.CodeNet_T)
                MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
                MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
                print('MAP@50 of Image to Text: %.5f, MAP of Text to Image: %.5f' % (MAP_I2T, MAP_T2I))
                
            return MAP_I2T, MAP_T2I
    
    def load_checkpoints(self):
            file_name = '%s_%d_bit_latest.pth' % (str(self.args.dataset), self.args.code_len)
            ckp_path = osp.join(self.args.save_model_path, file_name)
            try:
                obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            except IOError:
                print('********** No checkpoint %s!*********' % ckp_path)
                return
            self.CodeNet_I.load_state_dict(obj['ImgNet'])
            self.CodeNet_T.load_state_dict(obj['TxtNet'])

