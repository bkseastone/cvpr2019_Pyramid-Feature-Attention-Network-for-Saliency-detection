import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import epsilon

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

def logit(inputs):
    _epsilon = _to_tensor(epsilon(), inputs.dtype.base_dtype)
    inputs = tf.clip_by_value(inputs, _epsilon, 1 - _epsilon)
    inputs = tf.log(inputs / (1 - inputs))
    return inputs

def tfLaplace(x):
    laplace = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], tf.float32)
    laplace = tf.reshape(laplace, [3, 3, 1, 1])
    edge = tf.nn.conv2d(x, laplace, strides=[1, 1, 1, 1], padding='SAME')
    edge = tf.nn.relu(tf.tanh(edge))
    return edge

def EdgeLoss(y_true, y_pred):
    y_true_edge = tfLaplace(y_true)
    edge_pos = 2.
    edge_loss = K.mean(tf.nn.weighted_cross_entropy_with_logits(y_true_edge,y_pred,edge_pos), axis=-1)
    return edge_loss

def EdgeHoldLoss(y_true, y_pred):
    y_pred2 = tf.sigmoid(y_pred)
    y_true_edge = tfLaplace(y_true)
    y_pred_edge = tfLaplace(y_pred2)
    y_pred_edge = logit(y_pred_edge)
    edge_loss = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_edge,logits=y_pred_edge), axis=-1)
    saliency_pos = 1.12  # for sample-balance
    saliency_loss = K.mean(tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred,saliency_pos), axis=-1)
    alpha = 0.7  # hyperparam
    return alpha*saliency_loss+(1-alpha)*edge_loss

def FLoss(y_true, y_pred):
    log_like = False
    beta = 0.3
    y_pred = tf.sigmoid(y_pred)
    EPS = 1e-16
    TP = K.sum(tf.multiply(y_true, y_pred))
    H = beta * K.sum(y_true) + K.sum(y_pred)
    fmeasure = (1 + beta) * TP / (H + EPS)
    if log_like:
        return -tf.log(fmeasure + EPS)
    else:
        return (1 - fmeasure)

