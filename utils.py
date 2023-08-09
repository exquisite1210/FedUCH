import copy
from sklearn import model_selection
import torch

def fedavg(w_locals):
    training_num = 0
    for idx in range(len(w_locals)):
        (sample_num, averaged_params) = w_locals[idx]
        training_num += sample_num

    (sample_num,averaged_params) = w_locals[0]
    for k in averaged_params.keys():
        for i in range(0, len(w_locals)):
            local_sample_number, local_model_params = w_locals[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w
    return averaged_params




def globalAvg(dict_new_global_model_img_parameters,dict_new_global_model_txt_parameters):
    # print(dict_new_global_model_img_parameters.keys())#odict_keys(['fc1.weight', 'fc1.bias', 'fc_encode.weight', 'fc_encode.bias'])
    # dict_new_global_model_img_parameters['fc_encode.weight'] = dict_new_global_model_img_parameters['fc_encode.weight']*0.5+dict_new_global_model_txt_parameters['fc_encode.weight']*0.5
    # dict_new_global_model_img_parameters['fc_encode.bias'] = dict_new_global_model_img_parameters['fc_encode.bias']*0.5+dict_new_global_model_txt_parameters['fc_encode.bias']*0.5
    
    dict_new_global_model_txt_parameters['fc_encode.weight'] = dict_new_global_model_img_parameters['fc_encode.weight']*0.5+dict_new_global_model_txt_parameters['fc_encode.weight']*0.5
    dict_new_global_model_txt_parameters['fc_encode.bias'] = dict_new_global_model_img_parameters['fc_encode.bias']*0.5+dict_new_global_model_txt_parameters['fc_encode.bias']*0.5
    return dict_new_global_model_txt_parameters

