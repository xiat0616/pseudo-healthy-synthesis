import logging
import os
import matplotlib
matplotlib.use('agg')
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger("BaseSaveImage")

class ImageCallback_pseudo_healthy_unsup(Callback):
    """Callback for saving training images."""
    def __init__(self, conf, model, comet_exp=None):
        super(ImageCallback_pseudo_healthy_unsup, self).__init__()

        self.folder = os.path.join(conf.folder, "training_images")
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.model = model
        self.comet_exp = comet_exp

    def on_epoch_end(self, epoch=None, p_img = None, p_msk=None, h_img = None, h_msk=None):
        self.model.gan.save(self.folder + '/gan.h5df')
        self.model.critic_model.save(self.folder + '/critic_model.h5df')
        # Plot P_to_H images
        r, c = 4, 6

        pse_h_img = self.model.G_d_to_h.predict(p_img)
        seg_d_msk = self.model.S_d_to_msk.predict(p_img)
        rec_h_img = self.model.G_h_to_d.predict([pse_h_img, seg_d_msk])
        plt.figure()
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            axs[i, 0].set_title("p_img disc: %.1f" % np.mean(self.model.D_pse_h.predict([p_img[cnt:cnt+1]])), size=6)
            axs[i, 0].imshow(np.concatenate([p_img[cnt, :, :, j] for j in range(p_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 0].axis("off")

            axs[i, 1].set_title("pse_h_img disc: %.1f" % np.mean(self.model.D_pse_h.predict(pse_h_img[cnt:cnt+1])), size=6)
            axs[i, 1].imshow(np.concatenate([pse_h_img[cnt, :, :, j] for j in range(pse_h_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 1].axis("off")

            axs[i, 2].set_title("seg_d_msk ", size=6)
            axs[i, 2].imshow(np.concatenate([seg_d_msk[cnt, :, :, j] for j in range(seg_d_msk.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 2].axis("off")


            axs[i, 3].set_title("rec_d_img disc: %.1f" % np.mean(self.model.D_pse_h.predict(rec_h_img[cnt:cnt + 1])), size=6)
            axs[i, 3].imshow(np.concatenate([rec_h_img[cnt, :, :, j] for j in range(rec_h_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 3].axis("off")

            axs[i, 4].set_title("r_p_msk " , size=6)
            axs[i, 4].imshow(np.concatenate([p_msk[cnt, :, :, j] for j in range(p_msk.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 4].axis("off")

            axs[i, 5].set_title("h_img " , size=6)
            axs[i, 5].imshow(np.concatenate([h_img[cnt, :, :, j] for j in range(h_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 5].axis("off")
            cnt += 1

        fig.savefig(self.folder + "/P_to_h_img_%d.png" % epoch)
        plt.close()

        # Plot H_to_P images
        r, c = 4, 6

        del pse_h_img, seg_d_msk, rec_h_img

        pse_d_img = self.model.G_h_to_d.predict([h_img, h_msk])
        rec_h_img = self.model.G_d_to_h.predict(pse_d_img)
        rec_h_msk = self.model.S_d_to_msk.predict(pse_d_img)

        plt.figure()
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            axs[i, 0].set_title("h_img disc: %.1f" % np.mean(self.model.D_pse_h.predict([h_img[cnt:cnt+1]])), size=6)
            axs[i, 0].imshow(np.concatenate([h_img[cnt, :, :, j] for j in range(h_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 0].axis("off")

            axs[i, 1].set_title("h_msk ", size=6)
            axs[i, 1].imshow(np.concatenate([h_msk[cnt, :, :, j] for j in range(h_msk.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 1].axis("off")

            axs[i, 2].set_title("pse_d_img disc: %.1f" % np.mean(self.model.D_pse_h.predict(pse_d_img[cnt:cnt + 1])), size=6)
            axs[i, 2].imshow(np.concatenate([pse_d_img[cnt, :, :, j] for j in range(pse_d_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 2].axis("off")

            axs[i, 3].set_title("rec_h_img disc: %.1f" % np.mean(self.model.D_pse_h.predict(rec_h_img[cnt:cnt + 1])), size=6)
            axs[i, 3].imshow(np.concatenate([rec_h_img[cnt, :, :, j] for j in range(rec_h_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 3].axis("off")

            axs[i, 4].set_title("rec_h_msk ", size=6)
            axs[i, 4].imshow(np.concatenate([rec_h_msk[cnt, :, :, j] for j in range(rec_h_msk.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 4].axis("off")

            axs[i, 5].set_title("p_img " , size=6)
            axs[i, 5].imshow(np.concatenate([p_img[cnt, :, :, j] for j in range(p_img.shape[-1])], axis=1),
                             cmap="gray")
            axs[i, 5].axis("off")

            cnt += 1

        fig.savefig(self.folder + "/h_to_p_img_%d.png" % epoch)
        plt.close()