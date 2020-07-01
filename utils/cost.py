import keras.backend as K
import numpy as np

def l1_regularization(y_true, y_pred):
    return K.mean(K.abs(y_pred))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def l1_regularization_loss(y_true, y_pred,  age_gap):

    epsilon= K.exp(-age_gap/60)
    print("epsilon :",epsilon)
    # print(K.shape(epsilon))
    # print(epsilon)
    # # compute the euclidean norm by squaring ...
    # l1_loss = epsilon * K.abs(y_pred-y_true)
    l1_loss = epsilon * K.mean(K.abs(y_pred-y_true), axis=(1,2,3))

    return K.mean(l1_loss)


def dice_coef(y_true, y_pred, smooth=0.1):
    # Symbolically compute the intersection
    y_int = y_true * y_pred
    # Technically this is the negative of the Sorensen-Dice index. This is done for minimization purposes
    return (2 * K.sum(K.abs(y_int), axis=(1,2,3))+smooth) / (K.sum(K.abs(y_true), axis=(1,2,3)) +smooth+ K.sum(K.abs(y_pred), axis=(1,2,3)))

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

