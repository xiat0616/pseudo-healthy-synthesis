import os
from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)
import sys
sys.path.append('layers')
import logging
import matplotlib
matplotlib.use('Agg')
from models.generators.generator_H_to_P import generator_H_to_P
from models.generators.generator_P_to_H import generator_P_to_H
from models.generators.mask_segmentor import mask_segmentor
from models.discriminators.discriminator_image import critic_2D_pse_h
from models.discriminators.discriminatior_mask import critic_2D_mask
from models.basenet import BaseNet
from keras import Input, Model
from keras.optimizers import Adam
from keras.layers import Activation
from functools import partial
from utils.cost import wasserstein_loss, gradient_penalty_loss
from numpy.random import seed
from configuration.exp_supervised import EXPERIMENT_PARAMS
from easydict import EasyDict
from utils.cost import dice_coef_loss
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

log = logging.getLogger('pseudo')

class pseudo_healthy_unsupervised(BaseNet):
    '''
    Class of midl model;
    Discriminator is as old.
    '''
    def __init__(self, conf):
        super(pseudo_healthy_unsupervised, self).__init__(conf)
        self.conf = conf

    def build(self):
        # Build G_d_to_h
        G_d_to_h = generator_P_to_H(self.conf.gen_params_p_to_h)
        G_d_to_h.build()

        log.info("Generator from pathological to healthy")
        self.G_d_to_h = G_d_to_h.model
        self.G_d_to_h.summary(print_fn=log.info)

        # Build pathology segmentor
        S_d_to_msk = mask_segmentor(self.conf.seg_params)
        S_d_to_msk.build()

        log.info("Pathology segmentor")
        self.S_d_to_msk = S_d_to_msk.model
        self.S_d_to_msk.summary(print_fn=log.info)

        # Build G_h_to_d
        G_h_to_d = generator_H_to_P(self.conf.gen_params_h_to_p)
        G_h_to_d.build()

        log.info("Generator from pathological to healthy")
        self.G_h_to_d = G_h_to_d.model
        self.G_h_to_d.summary(print_fn=log.info)

        # Build D_pse_h
        D_pse_h = critic_2D_pse_h(self.conf.D_pse_h_params)
        D_pse_h.build()

        log.info("Pseudo healthy discriminator")
        self.D_pse_h = D_pse_h.model
        self.D_pse_h.summary(print_fn=log.info)


        # Build D_msk_d
        D_msk_d = critic_2D_mask(self.conf.D_msk_d_params)
        D_msk_d.build()

        log.info("Mask for discriminator")
        self.D_msk_d = D_msk_d.model
        self.D_msk_d.summary(print_fn=log.info)

        #=============================== Build G model =====================================
        self.D_pse_h.trainable = False
        self.D_msk_d.trainable = False

        # Cycle 1: p to h to p
        cy1_I_p = Input(shape=self.conf.input_shape)
        cy1_I_pse_h = self.G_d_to_h(cy1_I_p)
        cy1_M_seg_d = self.S_d_to_msk(cy1_I_p)

        cy1_I_rec_d = Activation(activation='linear', name='cy1_I_rec_d')(self.G_h_to_d([cy1_I_pse_h, cy1_M_seg_d]))

        cy1_dis_I_pse_h = Activation(activation='linear', name='cy1_dis_I_pse_h')(self.D_pse_h(cy1_I_pse_h))
        cy1_dis_M_seg_d = Activation(activation='linear', name='cy1_dis_M_seg_d')(self.D_msk_d(cy1_M_seg_d))

        # Cycle 2: h to h to h
        cy2_I_h = Input(shape=self.conf.input_shape)
        cy2_M_h = Input(shape=self.conf.input_shape)
        cy2_I_pse_h = self.G_h_to_d([cy2_I_h, cy2_M_h])

        cy2_dis_I_pse_h =Activation(activation='linear', name='cy2_dis_I_pse_d')(self.D_pse_h(cy2_I_pse_h))
        cy2_I_rec_h = Activation(activation='linear', name='cy2_I_rec_h')(self.G_d_to_h(cy2_I_pse_h))
        cy2_M_rec_h = Activation(activation='linear', name='cy2_M_rec_h')(self.S_d_to_msk(cy2_I_pse_h))

        self.gan = Model(inputs=[cy1_I_p, cy2_I_h, cy2_M_h], outputs=[cy1_dis_I_pse_h, cy1_dis_M_seg_d, cy1_I_rec_d, \
                                                             cy2_dis_I_pse_h, cy2_I_rec_h, cy2_M_rec_h])
        self.gan.compile(loss={'cy1_dis_I_pse_h':wasserstein_loss, 'cy1_dis_M_seg_d': wasserstein_loss , 'cy1_I_rec_d':"MAE",
                               'cy2_dis_I_pse_d':wasserstein_loss, 'cy2_I_rec_h':"MAE", 'cy2_M_rec_h':"MAE"},
                         loss_weights=[2, 1, 20, 1, 10, 10],
                         optimizer=Adam(lr=self.conf.lr, decay=self.conf.decay))
        self.gan.summary(print_fn=log.info)
        #=============================== Build D model ======================================

        self.D_pse_h.trainable = True
        self.D_msk_d.trainable = True

        # Cycle 1: p to h to p
        cy1_I_h = Input(shape=self.conf.input_shape)
        cy1_I_pse_h = Input(shape=self.conf.input_shape)
        cy1_average_I_h = Input(shape=self.conf.input_shape)

        cy1_M_d = Input(shape=self.conf.input_shape)
        cy1_M_seg_d = Input(shape=self.conf.input_shape)
        cy1_average_M_d = Input(shape=self.conf.input_shape)

        cy2_I_h = Input(shape=self.conf.input_shape)
        cy2_I_pse_h = Input(shape=self.conf.input_shape)
        cy2_average_I_h = Input(shape=self.conf.input_shape)


        dis_cy1_I_h = Activation(activation='linear', name='dis_cy1_I_h')(self.D_pse_h(cy1_I_h))
        dis_cy1_I_pse_h = Activation(activation='linear', name='dis_cy1_I_pse_h')(self.D_pse_h(cy1_I_pse_h))
        gp_cy1_I_h = Activation(activation='linear', name='gp_cy1_I_h')(self.D_pse_h([cy1_average_I_h]))

        dis_cy1_M_d = Activation(activation='linear', name='dis_cy1_M_d')(self.D_msk_d(cy1_M_d))
        dis_cy1_M_seg_d = Activation(activation='linear', name='dis_cy1_M_seg_d')(self.D_msk_d(cy1_M_seg_d))
        gp_cy1_M_d = Activation(activation='linear', name='gp_cy1_M_d')(self.D_msk_d([cy1_average_M_d]))

        dis_cy2_I_h = Activation(activation='linear', name='dis_cy2_I_h')(self.D_pse_h(cy2_I_h))
        dis_cy2_I_pse_h = Activation(activation='linear', name='dis_cy2_I_pse_h')(self.D_pse_h(cy2_I_pse_h))
        gp_cy2_I_h = Activation(activation='linear', name='gp_cy2_I_h')(self.D_pse_h([cy2_average_I_h]))
        # Gradient penalty loss
        cy1_partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=cy1_average_I_h,
                                  gradient_penalty_weight=self.conf.gp_weight)
        cy2_partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=cy2_average_I_h,
                                  gradient_penalty_weight=self.conf.gp_weight)
        cy1_partial_gp_loss_mask = partial(gradient_penalty_loss,
                                  averaged_samples=cy1_average_M_d,
                                  gradient_penalty_weight=self.conf.gp_weight)

        # Function need names or Keras will throw an error
        cy1_partial_gp_loss.__name__ = 'cy1_gradient_penalty'
        cy2_partial_gp_loss.__name__ = 'cy2_gradient_penalty'
        cy1_partial_gp_loss_mask.__name__ = 'cy1_gradient_penalty_mask'

        self.critic_model = Model(inputs=[cy1_I_h, cy1_I_pse_h, cy1_average_I_h,
                                          cy1_M_d, cy1_M_seg_d, cy1_average_M_d,
                                          cy2_I_h, cy2_I_pse_h, cy2_average_I_h],
                                  outputs=[dis_cy1_I_h, dis_cy1_I_pse_h, gp_cy1_I_h,
                                           dis_cy1_M_d, dis_cy1_M_seg_d, gp_cy1_M_d,
                                           dis_cy2_I_h, dis_cy2_I_pse_h, gp_cy2_I_h])

        self.critic_model.compile(optimizer=Adam(lr=self.conf.lr, decay=self.conf.decay),
                                  loss={'dis_cy1_I_h': wasserstein_loss,
                                        'dis_cy1_I_pse_h': wasserstein_loss,
                                        'gp_cy1_I_h': cy1_partial_gp_loss,
                                        'dis_cy1_M_d':wasserstein_loss,
                                        'dis_cy1_M_seg_d':wasserstein_loss,
                                        'gp_cy1_M_d': cy1_partial_gp_loss_mask,
                                        'dis_cy2_I_h':wasserstein_loss,
                                        'dis_cy2_I_pse_h':wasserstein_loss,
                                        'gp_cy2_I_h':cy2_partial_gp_loss
                                        })

    def load_models(self):
        if os.path.exists(self.conf.folder + "/MIDL_model"):
            log.info("Loading trained model from file")
            self.gan.load_weights(self.conf.folder + "/MIDL_model")

    def save_models(self):
        log.debug("Saving trained model")
        self.gan.save_weights(self.conf.folder + "/MIDL_model")

if __name__=='__main__':
    conf = EasyDict(EXPERIMENT_PARAMS)
    midl_baseline_w_gan = pseudo_healthy_unsupervised(conf)
    midl_baseline_w_gan.build()
    midl_baseline_w_gan.gan.summary()
    midl_baseline_w_gan.critic_model.summary()