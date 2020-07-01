# -*- coding: utf-8 -*-
import datetime
t = str(datetime.datetime.now())

lr = 0.00001
decay=0.0
beta1=0.5
dataset = "brats"
modality = "t1"
split = 1
EXPERIMENT_PARAMS = {
    "name":"midl",
    "validation_split":0.2,
    "l1_reg_weight":100,
    "ncritic": [10,5], # Number of training critic
    "seed": 10,  # Seed for the pseudorandom number generator.
    "folder": "experiments/pseudo_healthy_unsupervised_%s_%s_%d_"%(dataset, modality, split)+t.split(' ')[0]+'-'+t.split(' ')[1].replace('.','-').replace(':','-'),  # Folder to store experiment results.
    "epochs": 300,
    "batch_size": 4,
    "split": split,  # Default split of the dataset.
    "dataset": dataset,  # Training dataset.
    "modality": modality,
    "augment": True,  # Data augmentation
    "model": "pseudo_healthy_unsupervised.pseudo_healthy_unsupervised",  # Selected GAN architecture.
    "executor": "pseudo_healthy_unsupervised_executor.pseudo_healthy_unsupervised_executor",  # Selected experiment executor.
    "out_channels": 1,
    "gp_weight":10,
    "outputs": 1,
    "filters":32,
    # "num_masks": ACDCLoader().num_masks,  # Skip this, belongs to BaseModel.
    "lr": lr,
    "beta1":beta1,
    "beta2":0,
    "decay": decay,  # Skip this, belongs to BaseModel.
    "data_len":30,
    "input_shape":(208,160,1),

    "D_pse_h_params":{
        "depth":5,
        "input_shape":(208,160,1),
        "name":"D_pse_h",
        "beta1":beta1,
        "filters":32,
        "lr": lr,  # Learning rate.
        "decay": decay,  # Decay rate.
    },

    "D_msk_d_params": {
        "depth": 5,
        "input_shape": (208, 160, 1),
        "name": "D_msk_d",
        "beta1": beta1,
        "filters": 32,
        "lr": lr,  # Learning rate.
        "decay": decay,  # Decay rate.
    },
    "gen_params_p_to_h":{
        "depth":4,
        "input_shape": (208,160, 1),
        "name": "G_p_to_h",
        "filters": 32,
        "lr": lr,  # Learning rate.
        "decay": decay,  # Decay rate.
        "G_activation": 'sigmoid',
        "beta1":beta1,
    },
    "gen_params_h_to_p":{
        "depth": 4,
        "input_shape": ( 208,160, 1),
        "name": "G_h_to_p",
        "filters": 32,
        "lr": lr,  # Learning rate.
        "decay": decay,  # Decay rate.
        "G_activation": 'sigmoid',
        "beta1":beta1,
    },

    "seg_params":{
        "depth":4,
        "filters":32,
        "input_shape":(208, 160, 1),
        "name":"S_p",
        "inc_rate":2.
    }

}


def get():
    # shp = params["input_shape"]
    # ratio = params["image_downsample"]
    # shp = (int(shp[0] / ratio), int(shp[1] / ratio), shp[2])
    # params["input_shape"] = shp
    return EXPERIMENT_PARAMS
