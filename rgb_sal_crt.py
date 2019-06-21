import os
import numpy as np
import tensorflow as tf
from utils import dense_crf
from scipy import misc
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from infer import load_image, getres
from tensorflow.python.keras.layers import Input
from model import VGG16


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = 'model/PFA_00050.h5'
target_size = (256,256)
dropout = False
with_CPFE = True
with_CA = True
with_SA = True
if target_size[0 ] % 32 != 0 or target_size[1] % 32 != 0:
    raise ValueError('Image height and wight must be a multiple of 32')
model_input = Input(shape=(target_size[0],target_size[1],3))
model = VGG16(model_input,dropout=dropout, with_CPFE=with_CPFE, with_CA=with_CA, with_SA=with_SA)
model.load_weights(model_name,by_name=True)
for layer in model.layers:
    layer.trainable = False

rgb_file = "./tmp/1.jpg"
img_org = misc.imread(rgb_file)
img, shape = load_image(rgb_file)
img = np.array(img, dtype=np.float32)
sa = model.predict(img)
sa = getres(sa, shape)
'''
misc.imsave(rgb_file[:-4]+"_mask.png", sa)
#sa_crt = tf.py_func(dense_crf, [np.expand_dims(np.expand_dims(sa,0),3), np.expand_dims(img_org,0)], tf.float32)
#sa_crt = tf.Session().run(sa_crt)
sa_crt = dense_crf(np.expand_dims(np.expand_dims(sa,0),3), np.expand_dims(img_org,0))
misc.imsave(rgb_file[:-4]+"_mask_crt.png", sa_crt[0,:,:,0])
'''
plt.title('img-saliency-crt')
plt.subplot(131)
plt.imshow(img_org)
plt.subplot(132)
plt.imshow(sa,cmap='gray')
plt.subplot(133)
sa_crt = tf.py_func(dense_crf, [np.expand_dims(np.expand_dims(sa,0),3), np.expand_dims(img_org,0)], tf.float32)
sa_crt = tf.Session().run(sa_crt)
plt.imshow(sa_crt[0,:,:,0],cmap='gray')
plt.savefig(rgb_file[:-4]+"_org_mask_crt2.png")

