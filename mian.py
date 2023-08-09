#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import numpy as np
import torch
import copy
import numpy as np
import torch
import scipy.io as scio
import h5py
from sklearn.cluster import KMeans
import random
from models import ImgModule,ImgNet,TxtNet
from server import Server


def add_args(parser):

    parser.add_argument('--code_len', type=int, default=128, metavar='N',
                    help='code_lens')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.005, metavar='N',
                    help='learning_rate')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='weight_decay')
    parser.add_argument('--dw', default=1, type=float, help='loss1-alpha')
    parser.add_argument('--cw', default=1, type=float, help='loss2-beta')
    parser.add_argument('--K', default=1.5, type=float, help='pairwise distance resize')
    parser.add_argument('--a1', default=0.3, type=float, help='1 order distance')
    parser.add_argument('--a2', default=0.3, type=float, help='2 order distance')
    parser.add_argument('--global_epochs', default=50, type=int, help='Number of training epochs.')
    parser.add_argument('--local_epochs', default=5, type=int, help='comm_round.')
    parser.add_argument('--number_of_clients', default=10, type=int, help='client_num_in_total.')
    parser.add_argument('--EVAL', default=False, type=bool, help='train or test')
    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                    help='the frequency of the algorithms')
    parser.add_argument('--save_model_path', default='./checkpoint/', help='path to save_model')
    parser.add_argument('--dataset', default='wiki', help='flickr, nus, wiki')
    parser.add_argument('--gpu', default=1,type=int,help='gpu')
    parser.add_argument('--model', default='DGCPN',type=str,help='hash')
    parser.add_argument('--seed', type=int, default=2048, help='random seed')
    parser.add_argument('--frac', type=float, default=1,help='the fraction of clients: C')
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                of classes")
    parser.add_argument('--beta', type=float, default=1.0, help="number \
        of classes")

    return parser

def load_pretrain_model(path):
    return scio.loadmat(path)

def train_dirichlet_split_noniid(train_labels, alpha, n_clients):
    labelss = []
    for i in range(len(train_labels)):
        for j in range(len(train_labels[i])):
            if train_labels[i][j] == 1:
                labelss.append(j)
    labels = np.array(labelss)
    n_classes = args.num_classes
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idcs = [np.argwhere(labels==y).flatten() 
           for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def query_dirichlet_split_noniid(train_labels, alpha, n_clients):
    labelss = []
    for i in range(len(train_labels)):
        for j in range(len(train_labels[i])):
            if train_labels[i][j] == 1:
                labelss.append(j)
    labels = np.array(labelss)
    n_classes = args.num_classes
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idcs = [np.argwhere(labels==y).flatten() 
           for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def get_loader(args):
    path = "WikiPedia.mat"
    Data = h5py.File(path)
    images = Data['IAll'][:]
    labels = Data['LAll'][:]
    tags = Data['TAll'][:]

    images = images.transpose(3,2,0,1)
    labels = labels.transpose(1,0)
    tags = tags.transpose(1,0)
    pretrain_model = load_pretrain_model("imagenet-vgg-f.mat")
    FeatNet_I = ImgModule(pretrain_model)
    FeatNet_I.cuda().eval()
    num_data = len(images)
    new_images = np.zeros((num_data, 4096))
    for i in range(num_data):
        feature = FeatNet_I(torch.Tensor(images[i]).unsqueeze(0).cuda())
        new_images[i] = feature.cpu().detach().numpy()
    images = new_images.astype(np.float32)
    tags = tags.astype(np.float32)
    labels = labels.astype(int)
    return images,tags,labels

def split_data(images, tags,labels,q_num=462, v_num=231, t_num=2173, d_num=2173):
   
    query_dataset_images = images[0:q_num]
    retrieval_data_iamges = images[q_num:q_num+d_num]

    query_dataset_tags = tags[0:q_num]
    retrieval_data_tags = tags[q_num:q_num+d_num]

    query_dataset_labels = labels[0:q_num]
    retrieval_data_labels = labels[q_num:q_num+d_num]

    return query_dataset_images,query_dataset_tags,query_dataset_labels,\
    retrieval_data_iamges,retrieval_data_tags,retrieval_data_labels

# def one_hot(x,class_count):
#     return torch.eye(class_count)[x,:]

# def get_embedding_Kmeans(corpus,num_clusters):
#     corpus_embeddings = []

#     corpus_embeddings = corpus
#     ### KMEANS clustering
#     print("start Kmeans")
#     #  = 10
#     clustering_model = KMeans(n_clusters=num_clusters)
#     clustering_model.fit(corpus_embeddings)
#     cluster_assignment = clustering_model.labels_
#     print("end Kmeans")
#     # TODO: read the center points

#     return cluster_assignment

def get_split(args,images,tags,labels):
    # cluster_assignment = get_embedding_Kmeans(images,args.number_of_clients)
    # class_count = args.number_of_clients
    # labels_cluster = one_hot(cluster_assignment.astype('int64'),class_count=class_count)
    # np.save('%s_labels_one_hot_%d'%(args.dataset,class_count),labels_cluster)
    # labels_cluster = np.load('flickr_labels_one_hot_10.npy')

    query_dataset_images,query_dataset_tags,query_dataset_labels,\
    retrieval_data_iamges,retrieval_data_tags,retrieval_data_labels= split_data(images,tags,labels)
    
    return query_dataset_images,query_dataset_tags,query_dataset_labels,\
    retrieval_data_iamges,retrieval_data_tags,retrieval_data_labels


if __name__ == '__main__':
    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()  
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    
    images,tags,labels = get_loader(args)
    query_dataset_images,query_dataset_tags,query_dataset_labels,\
    retrieval_data_iamges,retrieval_data_tags,retrieval_data_labels = get_split(args,images,tags,labels)
    client_idcs_retrieval = train_dirichlet_split_noniid(retrieval_data_labels, alpha=0.5, n_clients=args.number_of_clients)
    client_idcs_query = query_dirichlet_split_noniid(query_dataset_labels,alpha=0.5, n_clients=args.number_of_clients)
    X = {}
    X['query'] = query_dataset_images
    X['retrieval'] = retrieval_data_iamges

    Y = {}
    Y['query'] = query_dataset_tags
    Y['retrieval'] = retrieval_data_tags

    L = {}
    L['query'] = query_dataset_labels
    L['retrieval'] = retrieval_data_labels
    print('---加载数据完毕---')

    if args.model == 'DGCPN':
        global_model_Img = ImgNet(args.code_len)
        txt_feat_len = 1024
        global_model_Txt = TxtNet(args.code_len,txt_feat_len)
    global_model_Img.to(args.device)
    global_model_Txt.to(args.device)

    server = Server(
                dict_users_retrieval =client_idcs_retrieval,
                dict_users_query =  client_idcs_query,
                images_data = X,
                text_data = Y,
                labels_data = L,
                model_img=copy.deepcopy(global_model_Img),
                model_txt= copy.deepcopy(global_model_Txt),
                args=args
                )
    server.setup_clients()
    server.train()
    
