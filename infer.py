import time
from tensorflow.python.keras import optimizers
from scipy import misc
import numpy as np
import cv2
import os
from tensorflow.python.keras.layers import Input
from model import VGG16
from utils import *
from data import getTestGenerator, ge_train_pair
from edge_hold_loss import *

import PIL.Image as Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

target_size = (256,256)

def rgba2rgb(img):
    return img[:,:,:3]*np.expand_dims(img[:,:,3],2)

def padding(x):
    h,w,c = x.shape
    size = max(h,w)
    paddingh = (size-h)//2
    paddingw = (size-w)//2
    temp_x = np.zeros((size,size,c))
    temp_x[paddingh:h+paddingh,paddingw:w+paddingw,:] = x
    return temp_x

# [imread](https://blog.csdn.net/renelian1572/article/details/78761278)
def load_image(path):
    x = misc.imread(path)
    if x.shape[2] == 4:
        x = rgba2rgb(x)
    sh = x.shape
    # Zero-center by mean pixel
    g_mean = np.array(([103.939,116.779,123.68])).reshape([1,1,3])
    x = padding(x)
    x = misc.imresize(x.astype(np.uint8), target_size, interp="bilinear").astype(np.float32) - g_mean
    x = np.expand_dims(x,0)
    return x,sh

def cut(pridict,shape):
    h,w,c = shape
    size = max(h, w)
    pridict = cv2.resize(pridict, (size,size))
    paddingh = (size - h) // 2
    paddingw = (size - w) // 2
    return pridict[paddingh:h + paddingh, paddingw:w + paddingw]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def getres(pridict,shape):
    pridict = sigmoid(pridict)*255
    pridict = np.array(pridict, dtype=np.uint8)
    pridict = np.squeeze(pridict)
    pridict = cut(pridict, shape)
    return pridict

def laplace_edge(x):
    laplace = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edge = cv2.filter2D(x/255.,-1,laplace)
    edge = np.maximum(np.tanh(edge),0)
    edge = edge * 255
    edge = np.array(edge, dtype=np.uint8)
    return edge

def background_transparent(image_path, trans_image_path):
    img = Image.open(image_path)
    iw,ih = img.size
    img = img.convert('RGBA')
    color_0 = img.getpixel((0,0))
    for h in range(ih):
        for w in range(iw):
            color_1 = img.getpixel((w,h))
            if color_0 == color_1:
                color_1 = color_1[:-1]+(0,)
                img.putpixel((w,h), color_1)
    img.save(trans_image_path)

def eval_it():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    import argparse
    parser = argparse.ArgumentParser(description='Train model your dataset')
    parser.add_argument('--test_file',default='test_pair.txt',help='your train file', type=str)

    parser.add_argument('--model_weights',default='result_PFA_FLoss/PFA_00500.h5',help='your model weights', type=str)
    batch_size = 32

    args = parser.parse_args()
    model_name = args.model_weights
    test_path = args.test_file
    HOME = os.path.expanduser('~')
    test_folder = os.path.join(HOME, '../ads-creative-image-algorithm/public_data/datasets/SalientDataset/DUTS/DUTS-TE')
    if not os.path.exists(test_path):
        ge_train_pair(test_path, test_folder, "DUTS-TE-Image", "DUTS-TE-Mask")
    target_size = (256,256)
    f = open(test_path, 'r')
    testlist = f.readlines()
    f.close()
    steps_per_epoch = len(testlist)/batch_size
    optimizer = optimizers.SGD(lr=1e-2, momentum=0.9, decay=0)
    loss = EdgeHoldLoss
    metrics = [acc, pre, rec, F_value, MAE]
    with_crf = False
    draw_bound  = False
    draw_poly = False
    draw_cutout = False
    dropout = False
    with_CPFE = True
    with_CA = True
    with_SA = True
    
    if target_size[0 ] % 32 != 0 or target_size[1] % 32 != 0:
        raise ValueError('Image height and wight must be a multiple of 32')
    testgen = getTestGenerator(test_path, target_size, batch_size)
    model_input = Input(shape=(target_size[0],target_size[1],3))
    model = VGG16(model_input,dropout=dropout, with_CPFE=with_CPFE, with_CA=with_CA, with_SA=with_SA)
    model.load_weights(model_name,by_name=True)
    
    for layer in model.layers:
        layer.trainable = False
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    evalSal = model.evaluate_generator(testgen, steps_per_epoch-1, verbose=1)
    print(evalSal)

def main():
    model_name = 'model/PFA_00050.h5'
    model_input = Input(shape=(target_size[0],target_size[1],3))
    model = VGG16(model_input,dropout=dropout, with_CPFE=with_CPFE, with_CA=with_CA, with_SA=with_SA)
    model.load_weights(model_name,by_name=True)
    
    for layer in model.layers:
        layer.trainable = False
    '''
    image_path = 'image/2.jpg'
    img, shape = load_image(image_path)
    img = np.array(img, dtype=np.float32)
    sa = model.predict(img)
    sa = getres(sa, shape)
    plt.title('saliency')
    plt.subplot(131)
    plt.imshow(cv2.imread(image_path))
    plt.subplot(132)
    plt.imshow(sa,cmap='gray')
    plt.subplot(133)
    edge = laplace_edge(sa)
    plt.imshow(edge,cmap='gray')
    plt.savefig(os.path.join('./train_1000_output','alpha.png'))
    #misc.imsave(os.path.join('./train_1000_output','alpha.png'), sa)
    '''
    #HOME = os.path.expanduser('~')
    #rgb_folder = os.path.join(HOME, 'data/sku_wdis_imgs/sku_wdis_imgs_12')
    rgb_folder = './tmp'
    output_folder = './train_1000_output'
    rgb_names = os.listdir(rgb_folder)
    print(rgb_folder, "\nhas {0} pics.".format(len(rgb_names)))
    start = time.time()
    for rgb_name in rgb_names:    
        if rgb_name[-4:] == '.jpg':
            img_org = misc.imread(os.path.join(rgb_folder, rgb_name))
            img, shape = load_image(os.path.join(rgb_folder, rgb_name))
            img = np.array(img, dtype=np.float32)
            sa = model.predict(img)
            sa = getres(sa, shape)
            misc.imsave(os.path.join(output_folder, rgb_name[:-4]+'_mask1.png'), sa)
            #1. densecrf
            if with_crf:
                sa = dense_crf(np.expand_dims(np.expand_dims(sa,0),3), np.expand_dims(img_org,0))
                sa = sa[0,:,:,0]
            #2. reduce contain relationship
            threshold_gray = 2
            threshold_area = 100
            connectivity = 8
            sa = sa.astype(np.uint8)
            _,sa = cv2.threshold(sa,threshold_gray,255,0)
            output = cv2.connectedComponentsWithStats(sa, connectivity, cv2.CV_32S)
            stats = output[2]
            area_img = img_org.shape[0] * img_org.shape[1]
            for rgns in range(1,stats.shape[0]):
                if area_img / stats[rgns,4] <= threshold_area:
                    continue
                x1, y1 = stats[rgns,0], stats[rgns,1] 
                x2, y2 = x1+stats[rgns,2], y1+stats[rgns,3]
                sa[y1:y2,x1:x2] = 0
            img_seg = np.zeros(img_org.shape[:3])
            _,cnts,hierarchy=cv2.findContours(sa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            img_seg=cv2.drawContours(img_seg, cnts, -1, (255, 255, 255), -1)
            misc.imsave(os.path.join(output_folder, rgb_name[:-4]+'_mask2.png'), img_seg)
            if draw_bound:
                #   Changing the connected components to bounding boxes
                # [ref1](https://blog.csdn.net/qq_21997625/article/details/86558178)
                img_org_bound = img_org.copy()
                area_img = img_org_bound.shape[0] * img_org_bound.shape[1]
                for rgns in range(1,stats.shape[0]):
                    if area_img / stats[rgns,4] > threshold_area:
                        continue
                    x1, y1 = stats[rgns,0], stats[rgns,1] 
                    x2, y2 = x1+stats[rgns,2], y1+stats[rgns,3]
                    cv2.rectangle(img_org_bound, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
                cv2.imwrite(os.path.join(output_folder, rgb_name[:-4]+'_bbox.png'), img_org_bound[..., ::-1])
                #print(os.path.join(output_folder, rgb_name[:-4]+'.png'))
            if draw_poly:
                # [0](https://blog.csdn.net/sunny2038/article/details/12889059)
                # [1](https://blog.csdn.net/jjddss/article/details/73527990)
                img_org_poly = img_org.copy()
                if len(cnts) <= 0:
                    continue
                for i in range(len(cnts)):
                    cnt = cnts[i]
                    _n = cnt.shape[0]
                    if _n <= 2:
                        continue
                    cnt = cnt.reshape((_n, 2))
                    cnt = tuple(map(tuple, cnt))
                    for j in range(_n):
                        img_org_poly = cv2.drawMarker(img_org_poly, cnt[j], (255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=2, thickness=2, line_type=cv2.FILLED)
                cv2.imwrite(os.path.join(output_folder, rgb_name[:-4]+'_poly.png'), img_org_poly[..., ::-1])
            if draw_cutout:
                mask_bbox_change_size = cv2.resize(img_seg, (img_org.shape[1], img_org.shape[0]))
                object_image = np.zeros(img_org.shape[:3], np.uint8)
                object_image = np.where(mask_bbox_change_size>0, img_org, object_image)
                #for i in range(3):
                #    object_image[:,:,i] = np.where(mask_bbox_change_size>0, img_org[:,:,i], object_image[:,:,i])
                misc.imsave(os.path.join(output_folder, rgb_name[:-4]+'_tmp.png'), object_image)
                background_transparent(os.path.join(output_folder, rgb_name[:-4]+'_tmp.png'),
                                       os.path.join(output_folder, rgb_name[:-4]+'_trans.png'))
    end = time.time()
    print("processing done in: %.4f time" % (end - start))

if __name__ == "__main__":
    eval_it()
