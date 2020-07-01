import numpy as np
import os
from configuration.exp_proposed_BraTs_as_old import EXPERIMENT_PARAMS
from easydict import EasyDict

class loader_BraTs(object):
    """
    Loader for BraTs dataset
    """
    def __init__(self, conf):
        self.data_folder = os.path.abspath('dataset')
        self.conf = conf


    def load_BraTs_data(self, dataset=None):
        """
        Load all data and age labels
        Generate age ordinal vectors
        :return: all data and age labels
        """
        # Load data from saved numpy arrays
        if self.conf.modality =="flair":
            unhealthy_images = np.load(self.data_folder+'/BraTs/unhealthy_flair_BraTs.npy',allow_pickle = True)
            unhealthy_masks = np.load(self.data_folder+'/BraTs/unhealthy_mask_BraTs.npy',allow_pickle = True)
            healthy_images = np.load(self.data_folder+'/BraTs/healthy_flair_BraTs.npy',allow_pickle = True)
            healthy_masks =np.load(self.data_folder+'/BraTs/healthy_mask_BraTs.npy',allow_pickle = True)

            # print(np.shape(unhealthy_masks))
            # print(np.shape(unhealthy_images))
        if self.conf.modality =="t2":
            unhealthy_images = np.load(self.data_folder + '/BraTs_t2/unhealthy_t2_training.npy',allow_pickle = True)
            unhealthy_masks = np.load(self.data_folder + '/BraTs_t2/unhealthy_t2_mask_training.npy',allow_pickle = True)
            healthy_images = np.load(self.data_folder + '/BraTs_t2/healthy_t2_training.npy',allow_pickle = True)
            healthy_masks = np.load(self.data_folder + '/BraTs_t2/healthy_t2_mask_training.npy',allow_pickle = True)

            # unhealthy_images_te = np.load(self.data_folder + '/BraTs_t2/unhealthy_t2_testing.npy')
            # unhealthy_masks_te = np.load(self.data_folder + '/BraTs_t2/unhealthy_t2_mask_testing.npy')
            # healthy_images_te = np.load(self.data_folder + '/BraTs_t2/healthy_t2_testing.npy')
            # healthy_masks_te = np.load(self.data_folder + '/BraTs_t2/healthy_t2_mask_testing.npy')

            unhealthy_images = np.concatenate(unhealthy_images, axis=0)
            unhealthy_masks = np.concatenate(unhealthy_masks, axis=0)
            healthy_images = np.concatenate(healthy_images, axis=0)
            healthy_masks = np.concatenate(healthy_masks, axis=0)

            # unhealthy_images_te = np.concatenate(unhealthy_images_te, axis=0)
            # unhealthy_masks_te = np.concatenate(unhealthy_masks_te, axis=0)
            # healthy_images_te = np.concatenate(healthy_images_te, axis=0)
            # healthy_masks_te = np.concatenate(healthy_masks_te, axis=0)

        print(np.min(unhealthy_images))

        try:
            return unhealthy_images, unhealthy_masks,healthy_images, healthy_masks, unhealthy_images_te, unhealthy_masks_te, healthy_images_te, healthy_masks_te
        except:
            return unhealthy_images, unhealthy_masks, healthy_images, healthy_masks

    def load_previous_data(self):
        if self.conf.modality =="flair":
            unhealthy_images = np.load(self.data_folder+'/BraTs/unhealthy_flair_training.npy', encoding="latin1",allow_pickle = True)
            unhealthy_masks  = np.load(self.data_folder+'/BraTs/unhealthy_mask_training.npy', encoding="latin1",allow_pickle = True)
            healthy_images   = np.load(self.data_folder+'/BraTs/healthy_flair_training.npy', encoding="latin1",allow_pickle = True)
            healthy_masks    = np.load(self.data_folder+'/BraTs/healthy_mask_training.npy', encoding="latin1",allow_pickle = True)
        if self.conf.modality =="t2":
            unhealthy_images = np.load(self.data_folder + '/BraTs_t2/unhealthy_t2_training.npy',allow_pickle = True)
            unhealthy_masks = np.load(self.data_folder + '/BraTs_t2/unhealthy_t2_mask_training.npy',allow_pickle = True)
            healthy_images = np.load(self.data_folder + '/BraTs_t2/healthy_t2_training.npy',allow_pickle = True)
            healthy_masks = np.load(self.data_folder + '/BraTs_t2/healthy_t2_mask_training.npy',allow_pickle = True)

        unhealthy_images = np.concatenate(unhealthy_images, axis=0)
        unhealthy_masks  = np.concatenate(unhealthy_masks, axis=0)
        healthy_images   = np.concatenate(healthy_images, axis=0)
        healthy_masks    = np.concatenate(healthy_masks, axis=0)
        # print(np.shape(unhealthy_masks))
        #healthy_images
        # print(np.min(unhealthy_images))
        # print(np.mean(unhealthy_masks))
        #
        # print(np.shape(healthy_masks))
        # print(np.mean(healthy_images))

        return unhealthy_images, unhealthy_masks, healthy_images, healthy_masks

    def load_BraTs_data_for_Unet(self, dataset=None):
        """
        Load all data and age labels
        Generate age ordinal vectors
        :return: all data and age labels
        """
        # Load data from saved numpy arrays
        if self.conf.modality =="flair":
            unhealthy_images = np.load(self.data_folder+'/BraTs/unhealthy_flair_BraTs.npy')
            unhealthy_masks = np.load(self.data_folder+'/BraTs/unhealthy_mask_BraTs.npy')
            healthy_images = np.load(self.data_folder+'/BraTs/healthy_flair_BraTs.npy')
            healthy_masks =np.load(self.data_folder+'/BraTs/healthy_mask_BraTs.npy')

            # print(np.shape(unhealthy_masks))
            # print(np.shape(unhealthy_images))
        if self.conf.modality =="t2":
            unhealthy_images = np.load(self.data_folder + '/BraTs_t2/unhealthy_t2_training.npy')
            unhealthy_masks = np.load(self.data_folder + '/BraTs_t2/unhealthy_t2_mask_training.npy')
            healthy_images = np.load(self.data_folder + '/BraTs_t2/healthy_t2_training.npy')
            healthy_masks = np.load(self.data_folder + '/BraTs_t2/healthy_t2_mask_training.npy')

            unhealthy_images_te = np.load(self.data_folder + '/BraTs_t2/unhealthy_t2_testing.npy')
            unhealthy_masks_te = np.load(self.data_folder + '/BraTs_t2/unhealthy_t2_mask_testing.npy')
            healthy_images_te = np.load(self.data_folder + '/BraTs_t2/healthy_t2_testing.npy')
            healthy_masks_te = np.load(self.data_folder + '/BraTs_t2/healthy_t2_mask_testing.npy')

            unhealthy_images = np.concatenate(unhealthy_images, axis=0)
            unhealthy_masks = np.concatenate(unhealthy_masks, axis=0)
            healthy_images = np.concatenate(healthy_images, axis=0)
            healthy_masks = np.concatenate(healthy_masks, axis=0)

            unhealthy_images_te = np.concatenate(unhealthy_images_te, axis=0)
            unhealthy_masks_te = np.concatenate(unhealthy_masks_te, axis=0)
            healthy_images_te = np.concatenate(healthy_images_te, axis=0)
            healthy_masks_te = np.concatenate(healthy_masks_te, axis=0)

        print(np.min(unhealthy_images))

        try:
            return unhealthy_images, unhealthy_masks,healthy_images, healthy_masks, unhealthy_images_te, unhealthy_masks_te, healthy_images_te, healthy_masks_te
        except:
            return unhealthy_images, unhealthy_masks, healthy_images, healthy_masks


    def load_cam_and_brats_t2_data(self):
        # data_dir ='dataset'
        unhealthy_images = np.load(self.data_folder+'/BraTs_t2/unhealthy_t2_training.npy')
        unhealthy_masks = np.load(self.data_folder+'/BraTs_t2/unhealthy_t2_mask_training.npy')

        healthy_images = np.load(self.data_folder+'/Cam_t2/camcan_imgs_t2.npy')


        unhealthy_images = np.concatenate(unhealthy_images, axis=0)
        unhealthy_masks  = np.concatenate(unhealthy_masks, axis=0)
        healthy_images = np.concatenate(np.swapaxes(np.swapaxes(np.swapaxes(healthy_images,2,3 ),1,2),2,3), axis=0)
        healthy_masks = np.zeros(np.shape(healthy_images))

        print(np.shape(unhealthy_images), np.max(unhealthy_images), np.min(unhealthy_images))
        print(np.shape(unhealthy_masks), np.max(unhealthy_masks), np.min(unhealthy_masks))
        print(np.shape(healthy_images), np.max(healthy_images),np.min(healthy_images))
        print(np.shape(healthy_masks), np.max(healthy_masks),np.min(healthy_masks))

        return unhealthy_images, unhealthy_masks, healthy_images, healthy_masks

    def load_data_for_deformation_classifier(self):
        unhealthy_images = np.load(self.data_folder+'/BraTs_t2/healthy_t2_training.npy')
        unhealthy_masks = np.load(self.data_folder+'/BraTs_t2/healthy_t2_mask_training.npy')
        healthy_images = np.load(self.data_folder+'/Cam_t2/camcan_imgs_t2.npy')

        unhealthy_images = np.concatenate(unhealthy_images, axis=0)
        unhealthy_masks  = np.concatenate(unhealthy_masks, axis=0)
        healthy_images = np.concatenate(np.swapaxes(np.swapaxes(np.swapaxes(healthy_images,2,3 ),1,2),2,3), axis=0)
        healthy_masks = np.zeros(np.shape(healthy_images))

        print(np.shape(unhealthy_images), np.max(unhealthy_images), np.min(unhealthy_images))
        print(np.shape(unhealthy_masks), np.max(unhealthy_masks), np.min(unhealthy_masks))
        print(np.shape(healthy_images), np.max(healthy_images),np.min(healthy_images))
        print(np.shape(healthy_masks), np.max(healthy_masks),np.min(healthy_masks))

        return unhealthy_images, unhealthy_masks, healthy_images, healthy_masks
if __name__=='__main__':
    loader = loader_BraTs(EasyDict(EXPERIMENT_PARAMS))
    unhealthy_images, unhealthy_masks, healthy_images, healthy_masks = loader.load_data_for_deformation_classifier()
