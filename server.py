import copy
import numpy as np
from client import Client
from torch.utils.data import DataLoader, Dataset
from utils import fedavg
import os.path as osp
import torch
import torch.nn as nn
class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]

        return img, text, label,index

    def __len__(self):
        count = len(self.images)
        assert len(self.images) == len(self.labels)
        return count

class Server(object):
    def __init__(self,dict_users_retrieval,dict_users_query,images_data, text_data, labels_data,model_img,model_txt, args):
        self.users_retrieval_idxs = dict_users_retrieval
        self.user_query_idxs = dict_users_query
        # self.user_retrieval_idxs = dict_users_retrieval

        # self.images_data_train = images_data['train']
        self.images_data_query = images_data['query']
        self.images_data_retrieval = images_data['retrieval']

        # self.text_data_train = text_data['train']
        self.text_data_query = text_data['query']
        self.text_data_retrieval = text_data['retrieval']

        # self.labels_data_train = labels_data['train']
        self.labels_data_query = labels_data['query']
        self.labels_data_retrieval = labels_data['retrieval']



        self.model_img = model_img
        self.model_txt = model_txt
        self.args = args
        self.device = args.device

        self.list_clients = []

    def setup_clients(self):
        for i in range(self.args.number_of_clients):
            # print(i)
            dataloder_for_this_client,data_train= self.assign_dataloader_for_each_client(client_id=i)

            client = Client(client_id=i,
                            model_img = copy.deepcopy(self.model_img),
                            model_txt = copy.deepcopy(self.model_txt),
                            args=self.args,
                            dataloader=dataloder_for_this_client,
                            datatrain = data_train,
                            device=self.device,
                            )
            self.list_clients.append(client)
    
    def assign_dataloader_for_each_client(self, client_id):

        dataloader1,data_train = self.setup_dataloader(self.users_retrieval_idxs[client_id],self.user_query_idxs[client_id])

        return dataloader1,data_train
 
    def setup_dataloader(self, users_retrieval_idxs_for_this_client,user_query_idxs_for_this_client):
        samples = len(users_retrieval_idxs_for_this_client)

        idxs_train = np.random.choice(users_retrieval_idxs_for_this_client,samples)


        idxs_val = idxs_train[int(0.9 * len(idxs_train)):int(1.0 * len(idxs_train))]#
        idxs_query = user_query_idxs_for_this_client[:int(len(user_query_idxs_for_this_client))]
        idxs_retrieval = users_retrieval_idxs_for_this_client[:int(len(users_retrieval_idxs_for_this_client))]
        """
        能和分配的索引保持一致
        """

        train_X_list = []
        train_Y_list = []
        train_L_list = []
        for idx in idxs_train:
            # print(type(idx))
            train_X_list.append(self.images_data_retrieval[idx])
            train_Y_list.append(self.text_data_retrieval[idx])
            train_L_list.append(self.labels_data_retrieval[idx])
        train_X = np.array(train_X_list)
        train_Y = np.array(train_Y_list)
        train_L = np.array(train_L_list)
        
        
        valid_X_list = []
        valid_Y_list = []
        valid_L_list = []
        for idx in idxs_val:
            valid_X_list.append(self.images_data_retrieval[idx])
            valid_Y_list.append(self.text_data_retrieval[idx])
            valid_L_list.append(self.labels_data_retrieval[idx])
        valid_X = np.array(valid_X_list)
        valid_Y = np.array(valid_Y_list)
        valid_L = np.array(valid_L_list)
        
        query_X_list = []
        query_Y_list = []
        query_L_list = []
        for idx in idxs_query:
            query_X_list.append(self.images_data_query[idx])
            query_Y_list.append(self.text_data_query[idx])
            query_L_list.append(self.labels_data_query[idx])
        query_X = np.array(query_X_list)
        query_Y = np.array(query_Y_list)
        query_L = np.array(query_L_list)
        
        retrieval_X_list = []
        retrieval_Y_list = []
        retrieval_L_list = []
        for idx in idxs_retrieval:
            retrieval_X_list.append(self.images_data_retrieval[idx])
            retrieval_Y_list.append(self.text_data_retrieval[idx])
            retrieval_L_list.append(self.labels_data_retrieval[idx])
        retrieval_X = np.array(retrieval_X_list)
        retrieval_Y = np.array(retrieval_Y_list)
        retrieval_L = np.array(retrieval_L_list)
        # print('---------------shape---------------')
        # print(train_L.shape)
        # print(train_X.shape)
        # print(train_Y.shape)

        imgs = {'train': train_X, 'query': query_X, 'database': retrieval_X, 'databasev': retrieval_X, 'validation': valid_X}
        texts = {'train': train_Y, 'query': query_Y, 'database': retrieval_Y, 'databasev': retrieval_Y, 'validation': valid_Y}
        labels = {'train': train_L, 'query': query_L, 'database': retrieval_L, 'databasev': retrieval_L, 'validation': valid_L}
        
        dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
            for x in ['query', 'train', 'database', 'databasev', 'validation']}
        shuffle = {'query': False, 'train': True, 'database': False, 'validation': False, 'databasev': False}
        dataloader = {x: DataLoader(dataset[x], batch_size=self.args.batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in ['query', 'train', 'database', 'databasev', 'validation']}

        return dataloader,(train_X,train_Y,train_L)

    def train(self):

        num_of_clients_chosen_for_train = max(int(self.args.frac * self.args.number_of_clients), 1)


        
        for round_idx in range(self.args.global_epochs):
            print("################################Communication round : %d################################"%(round_idx+1))
            print('----------------获取参与联邦训练的用户索引----------------')
            list_index_of_clients_for_train = np.random.choice(range(self.args.number_of_clients),
                                                               num_of_clients_chosen_for_train,
                                                               replace=False)
            list_index_of_clients_for_train.sort()
            
            list_all_clients_feedback_model_img_parameters = []
            list_all_clients_feedback_model_txt_parameters = []
            #对每一个用户
            for client_id in list_index_of_clients_for_train:
                print('-------------第%d个用户开始训练-------------'%(client_id+1))
                self.list_clients[client_id].train(round_idx,self.model_img.cpu().state_dict(),self.model_txt.cpu().state_dict())
                
                # self.list_clients[client_id].local_validate(epoch = round_idx)

                dict_feedback_model_img_parameters = self.list_clients[client_id].CodeNet_I.state_dict()
                dict_feedback_model_txt_parameters = self.list_clients[client_id].CodeNet_T.state_dict()
                print('----------------train finished----------------')
                list_all_clients_feedback_model_img_parameters.append((self.list_clients[client_id].get_sample_number(),copy.deepcopy(dict_feedback_model_img_parameters)))
                list_all_clients_feedback_model_txt_parameters.append((self.list_clients[client_id].get_sample_number(),copy.deepcopy(dict_feedback_model_txt_parameters)))

            dict_new_global_model_img_parameters = fedavg(list_all_clients_feedback_model_img_parameters)
            dict_new_global_model_txt_parameters = fedavg(list_all_clients_feedback_model_txt_parameters)
            
        

            self.model_img.load_state_dict(dict_new_global_model_img_parameters)
            self.model_txt.load_state_dict(dict_new_global_model_txt_parameters)
            self.save_checkpoints()




            self.local_validate_on_all_clients(round_idx)

        self.args.EVAL = True
        self.local_validate_on_all_clients(round_idx)

    
    
    #avg
    def local_validate_on_all_clients(self, round_idx):
        MAP_I2T_all, MAP_T2I_all = 0.0,0.0
   
        for idx, client in enumerate(self.list_clients):
            print("----------------第%d个用户验证----------------"%(idx+1))
            MAP_I2T, MAP_T2I = client.local_validate(epoch = round_idx)
            # print("----------------第%d个用户验证结束----------------"%(idx+1))
            MAP_I2T_all += MAP_I2T
            MAP_T2I_all += MAP_T2I

        MAP_I2T_ave = MAP_I2T_all / self.args.number_of_clients
        MAP_T2I_ave = MAP_T2I_all / self.args.number_of_clients
        print('average MAP_I2T_ave:'+str(MAP_I2T_ave))
        print('average MAP_T2I_ave:'+str(MAP_T2I_ave))

    def save_checkpoints(self):
        file_name = '%s_%d_bit_latest.pth' % (str(self.args.dataset) , self.args.code_len)

        ckp_path = osp.join(self.args.save_model_path, file_name)
        obj = {
            'ImgNet': self.model_img.cpu().state_dict(),
            'TxtNet': self.model_txt.cpu().state_dict(),
        }
        torch.save(obj, ckp_path)
        print('**********Save the trained model successfully.**********')
