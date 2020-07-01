import numpy as np
import os

class data_loader(object):
    """
    Loader for dataset
    """
    def __init__(self):
        self.data_folder = os.path.abspath('dataset')

    def load_data(self, dataset="isles", modality="flair", split="1"):
        """
        load data given modality and split
        """
        if modality =="brats":
            unhealthy_images_tr = np.load(self.data_folder+"/%s/%s/%s/unhealthy_image_training.npy"%(dataset, modality, split),
                                          allow_pickle=True)
            unhealthy_masks_tr = np.load(self.data_folder + "/%s/%s/%s/unhealthy_mask_training.npy"%(dataset, modality, split),
                                         allow_pickle=True)
            healthy_images_tr = np.load(self.data_folder+'/camcan/%s/%s/healthy_image_training.npy'%(modality, split),allow_pickle = True)
            healthy_masks_tr = np.load(self.data_folder+'/camcan/%s/%s/healthy_mask_training.npy'%(modality, split),allow_pickle = True)

            unhealthy_images_va = np.load(self.data_folder+"/%s/%s/%s/unhealthy_image_validation.npy"%(dataset, modality, split),
                                          allow_pickle=True)
            unhealthy_masks_va = np.load(self.data_folder + "/%s/%s/%s/unhealthy_mask_validation.npy"%(dataset, modality, split),
                                         allow_pickle=True)
            healthy_images_va = np.load(self.data_folder+'/camcan/%s/%s/healthy_image_validation.npy'%(modality, split),allow_pickle = True)
            healthy_masks_va = np.load(self.data_folder+'/camcan/%s/%s/healthy_mask_validation.npy'%(modality, split),allow_pickle = True)

        if modality =="isles":
            unhealthy_images_tr = np.load(self.data_folder+"/%s/%s/%s/unhealthy_image_training.npy"%(dataset, modality, split),
                                          allow_pickle=True)
            unhealthy_masks_tr = np.load(self.data_folder + "/%s/%s/%s/unhealthy_mask_training.npy"%(dataset, modality, split),
                                         allow_pickle=True)
            healthy_images_tr = np.load(self.data_folder+"/%s/%s/%s/healthy_image_training.npy"%(dataset, modality, split),
                                          allow_pickle=True)
            healthy_masks_tr = np.load(self.data_folder + "/%s/%s/%s/healthy_mask_training.npy"%(dataset, modality, split),
                                         allow_pickle=True)

            unhealthy_images_va = np.load(self.data_folder+"/%s/%s/%s/unhealthy_image_validation.npy"%(dataset, modality, split),
                                          allow_pickle=True)
            unhealthy_masks_va = np.load(self.data_folder + "/%s/%s/%s/unhealthy_mask_validation.npy"%(dataset, modality, split),
                                         allow_pickle=True)
            healthy_images_va = np.load(self.data_folder+"/%s/%s/%s/healthy_image_validation.npy"%(dataset, modality, split),
                                          allow_pickle=True)
            healthy_masks_va = np.load(self.data_folder + "/%s/%s/%s/healthy_mask_validation.npy"%(dataset, modality, split),
                                         allow_pickle=True)

        return unhealthy_images_tr, unhealthy_masks_tr, healthy_images_tr, healthy_masks_tr, unhealthy_images_va, unhealthy_masks_va, healthy_images_va, healthy_masks_va