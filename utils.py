from tensorflow.python.keras import backend as K
import tensorflow as tf
import pydensecrf.densecrf as dcrf
import numpy as np

def acc(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def pre(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    TP = K.sum(tf.multiply(y_true, K.round(y_pred)))
    S = K.sum(K.round(y_pred))
    return TP / S

def rec(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    TP = K.sum(tf.multiply(y_true, K.round(y_pred)))
    S = K.sum(y_true)
    return TP / S

def F_value(y_true, y_pred):
	return ((1+0.3)*pre(y_true,y_pred)*rec(y_true,y_pred))/(0.3*pre(y_true,y_pred)+rec(y_true,y_pred))

def MAE(y_true, y_pred):
	return 1-acc(y_true, y_pred)

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dense_crf(probs, img=None, n_iters=2, 
              sxy_gaussian=(1,1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49,49), compat_bilateral=5,
              srgb_bilateral=(13,13,13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       support 1 image only.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
    
    Args:
      probs: salient probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      
    Returns:
      Refined predictions after MAP inference.
    """
    EPSILON = 1e-8
    n_classes = 2  # salient or not
    _, h, w, _ = probs.shape
    probs = np.concatenate((probs,probs), 3) 
    probs = probs / 255
    probs[:,:,:,0] = 1 - probs[:,:,:,0]
    probs = probs[0].transpose(2, 0, 1).copy(order='C')
    d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
    U = -np.log(EPSILON+probs) # Unary potential.
    U = U.reshape((n_classes, -1)) # Needs to be flat.
    U = np.array(U, dtype=np.float32)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert(img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32)
    preds = preds[1, :]
    preds = preds * 255
    preds = preds.reshape((1, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)

