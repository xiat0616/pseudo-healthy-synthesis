from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import logging
from configuration.exp_unsupervised import EXPERIMENT_PARAMS
from keras.utils import Progbar
from model_executor.basic_executor import Executor
from loader.data_loader import data_loader
from easydict import EasyDict
from callbacks.loss_callback import SaveLoss
from callbacks.image_call_back_pseudo_healthy_unsupervised import ImageCallback_pseudo_healthy_unsup as ImageCallback

log = logging.getLogger("proposed_executor")

class pseudo_healthy_unsupervised_executor(Executor):
    def __init__(self, conf, model=None, comet_exp=None):
        super(pseudo_healthy_unsupervised_executor, self).__init__(conf, model, comet_exp)

    def init_train_data(self):
        loader = data_loader()
        p_images, p_masks , h_images, h_masks, _, _ , _, _ = loader.load_data(
                                                                 dataset=self.conf.dataset,
                                                                 modality=self.conf.modality,
                                                                 split=self.conf.modality)

        p_images = p_images[0:int((len(p_images) // self.conf.batch_size) * self.conf.batch_size)]
        p_masks = p_masks[0:int((len(p_masks) // self.conf.batch_size) * self.conf.batch_size)]
        h_images = h_images[0:int((len(h_images) // self.conf.batch_size) * self.conf.batch_size)]
        h_masks = h_masks[0:int((len(h_masks) // self.conf.batch_size) * self.conf.batch_size)]

        self.p_images = np.expand_dims(p_images, axis=-1)
        self.p_masks  = np.expand_dims(p_masks , axis=-1)
        self.h_images = np.expand_dims(h_images, axis=-1)
        self.h_masks  = np.expand_dims(h_masks , axis=-1)
        self.conf.data_len = len(h_images)
        del p_images, p_masks, h_images, h_masks
    def get_loss_names(self):
        return ["d_dis_pse_image_loss", "d_dis_r_image_loss", "d_gp_loss", "d_dis_d_mask_loss",
                "g_dis_pse_image_loss", "g_rec_image_loss", "g_dis_d_mask_loss",
                "test_self_rec_loss"]

    def train(self):
        self.init_train_data()
        # make genetrated data
        gen_dict = self.get_datagen_params()
        p_gen = ImageDataGenerator(**gen_dict).flow(x=self.p_images, y=self.p_masks, batch_size=self.conf.batch_size)
        h_gen = ImageDataGenerator(**gen_dict).flow(x=self.h_images, y=self.h_masks, batch_size=self.conf.batch_size)
        random_p_masks = ImageDataGenerator(**gen_dict).flow(x= self.p_masks, batch_size=self.conf.batch_size)

        # initialize training
        batches = int(np.ceil(self.conf.data_len/self.conf.batch_size))
        progress_bar = Progbar(target=batches * self.conf.batch_size)

        sl = SaveLoss(self.conf.folder)
        cl = CSVLogger(self.conf.folder+'/training.csv')
        cl.on_train_begin()
        img_clb = ImageCallback(self.conf, self.model, self.comet_exp)

        loss_names = self.get_loss_names()
        total_loss = {n: [] for n in loss_names}

        # start training
        for epoch in range(self.conf.epochs):
            log.info("Train epoch %d/%d"%(epoch, self.conf.epochs))
            epoch_loss = {n: [] for n in loss_names}
            epoch_loss_list = []
            pool_to_print_p_img, pool_to_print_p_msk, pool_to_print_h_img, pool_to_print_h_msk = [], [], [], []

            for batch in range(batches):
                p_img, p_msk = next(p_gen)
                h_img, h_msk = next(h_gen)
                r_p_msk = next(random_p_masks)

                if len(pool_to_print_p_img)<30:
                    pool_to_print_p_img.append(p_img[0])
                    pool_to_print_p_msk.append(p_msk[0])


                if len(pool_to_print_h_img)<30:
                    pool_to_print_h_img.append(h_img[0])
                    pool_to_print_h_msk.append(h_msk[0])

                # Adversarial ground truths
                real_pred = -np.ones((h_img.shape[0],1))
                fake_pred = np.ones((h_img.shape[0],1))
                dummy = np.zeros((h_img.shape[0],1))
                dummy_Img = np.ones(h_img.shape)

                if self.conf.self_rec:
                    h_test_sr = self.model.train_self_rec.fit([h_img, h_msk], [h_img, h_img], epochs=1, verbose=0)
                    epoch_loss["test_self_rec_loss"].append(np.mean(h_test_sr.history["loss"]))
                else:
                    epoch_loss["test_self_rec_loss"].append(0)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Get a group of synthetic msks and imgs
                cy1_pse_h_img = self.model.G_d_to_h.predict(p_img)
                cy1_seg_d_msk = self.model.S_d_to_msk.predict(p_img)
                cy2_fake_h_img = self.model.G_h_to_d.predict([h_img, h_msk])

                if epoch<25:
                    for _ in range(self.conf.ncritic[0]):
                        cy1_epsilon = np.random.uniform(0,1, size=(h_img.shape[0],1,1,1))
                        cy1_average = cy1_epsilon * h_img +(1-cy1_epsilon) * cy1_pse_h_img

                        cy1_epsilon_msk = np.random.uniform(0, 1, size=(h_img.shape[0], 1, 1, 1))
                        cy1_average_msk = cy1_epsilon_msk * r_p_msk + (1 - cy1_epsilon) * cy1_seg_d_msk

                        cy2_epsilon = np.random.uniform(0,1, size=(h_img.shape[0],1,1,1))
                        cy2_average = cy2_epsilon * h_img +(1-cy2_epsilon) * cy2_fake_h_img

                        h_d = self.model.critic_model.fit([h_img, cy1_pse_h_img, cy1_average,
                                                           r_p_msk, cy1_seg_d_msk, cy1_average_msk,
                                                           h_img, cy2_fake_h_img, cy2_average],
                                                          [real_pred, fake_pred, dummy, real_pred, fake_pred, dummy,
                                                           real_pred, fake_pred, dummy],
                                                          epochs=1, verbose=0)
                else:
                    for _ in range(self.conf.ncritic[1]):
                        cy1_epsilon = np.random.uniform(0,1, size=(h_img.shape[0],1,1,1))
                        cy1_average = cy1_epsilon * h_img +(1-cy1_epsilon) * cy1_pse_h_img

                        cy1_epsilon_msk = np.random.uniform(0, 1, size=(h_img.shape[0], 1, 1, 1))
                        cy1_average_msk = cy1_epsilon_msk * r_p_msk + (1 - cy1_epsilon) * cy1_seg_d_msk

                        cy2_epsilon = np.random.uniform(0,1, size=(h_img.shape[0],1,1,1))
                        cy2_average = cy2_epsilon * h_img +(1-cy2_epsilon) * cy2_fake_h_img

                        h_d = self.model.critic_model.fit([h_img, cy1_pse_h_img, cy1_average,
                                                           r_p_msk, cy1_seg_d_msk, cy1_average_msk,
                                                           h_img, cy2_fake_h_img, cy2_average],
                                                          [real_pred, fake_pred, dummy, real_pred, fake_pred, dummy,
                                                           real_pred, fake_pred, dummy],
                                                          epochs=1, verbose=0)
                # print(h_d.history)
                d_dis_pse_image_loss = np.mean([h_d.history['dis_cy1_I_pse_h_loss'], h_d.history['dis_cy2_I_pse_h_loss']])
                d_dis_r_image_loss   = np.mean([h_d.history['dis_cy1_I_h_loss'], h_d.history['dis_cy2_I_h_loss']])
                d_dis_d_mask_loss    = np.mean([h_d.history['dis_cy1_M_d_loss'], h_d.history['dis_cy1_M_seg_d_loss']])
                d_gp_loss            = np.mean([h_d.history['gp_cy1_I_h_loss'], h_d.history['gp_cy2_I_h_loss'], h_d.history['gp_cy1_M_d_loss']])
                epoch_loss['d_dis_pse_image_loss'].append(d_dis_pse_image_loss)
                epoch_loss['d_dis_r_image_loss'].append(d_dis_r_image_loss)
                epoch_loss['d_dis_d_mask_loss'].append(d_dis_d_mask_loss)
                epoch_loss['d_gp_loss'].append(d_gp_loss)

                # --------------------
                #  Train Generator
                # --------------------

                h_g = self.model.gan.fit([p_img, h_img, h_msk],[real_pred, real_pred, p_img, real_pred, h_img, h_msk], epochs=1, verbose=0)
                g_dis_pse_image_loss = np.mean([h_g.history['cy1_dis_I_pse_h_loss'], h_g.history['cy2_dis_I_pse_d_loss']])
                g_rec_image_loss = np.mean([h_g.history['cy2_I_rec_h_loss'], h_g.history['cy1_I_rec_d_loss']])
                g_dis_d_mask_loss = np.mean(h_g.history['cy1_dis_M_seg_d_loss'])
                epoch_loss['g_dis_pse_image_loss'].append(g_dis_pse_image_loss)
                epoch_loss['g_rec_image_loss'].append(g_rec_image_loss)
                epoch_loss['g_dis_d_mask_loss'].append(g_dis_d_mask_loss)
                # print(h_g.history)
                # Plot the progress
                progress_bar.update((batch + 1) * self.conf.batch_size)

            for n in loss_names:
                epoch_loss_list.append((n, np.mean(epoch_loss[n])))
                total_loss[n].append(np.mean(epoch_loss[n]))

            for n in loss_names:
                epoch_loss_list.append((n, np.mean(epoch_loss[n])))
                total_loss[n].append(np.mean(epoch_loss[n]))

            log.info(str('Epoch %d/%d: ' + ', '.join([l + ' Loss = %.3f' for l in loss_names])) %
                     ((epoch, self.conf.epochs) + tuple(total_loss[l][-1] for l in loss_names)))
            logs = {l: total_loss[l][-1] for l in loss_names}

            cl.model = self.model.D_pse_h
            cl.model.stop_training = False
            cl.on_epoch_end(epoch, logs)
            sl.on_epoch_end(epoch, logs)
            pool_to_print_p_img = np.asarray(pool_to_print_p_img)
            pool_to_print_p_msk = np.asarray(pool_to_print_p_msk)
            pool_to_print_h_img = np.asarray(pool_to_print_h_img)
            pool_to_print_h_msk = np.asarray(pool_to_print_h_msk)
            print("pool_to_print_p_img: ", np.shape(pool_to_print_p_img))
            img_clb.on_epoch_end(epoch, pool_to_print_p_img, pool_to_print_p_msk,
                                 pool_to_print_h_img, pool_to_print_h_msk)


if __name__=='__main__':
    midl_model = midl_baseline_w_gan_cycleGAN(EasyDict(EXPERIMENT_PARAMS))
    midl_model.build()
    executor_midl_model = executor_wasserstein_cycleGAN(EasyDict(EXPERIMENT_PARAMS), midl_model)
    executor_midl_model.init_train_data()
