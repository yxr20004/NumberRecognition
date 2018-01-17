import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.python.platform import gfile
#from tensorflow.python.framework import graph_util
from PIL import Image

w = 60#360
h = 80#178
ws = 150
hs = 49
box  = [
    [ws,hs+1353, w, h],
    [ws,hs+1531, w, h],
    [ws,hs+1711, w, h],
#     [ws,hs+1888, w, h],
    
    [ws+361,hs+1353, w, h],
    [ws+361,hs+1531, w, h],
    [ws+361,hs+1711, w, h],
    [ws+361,hs+1888, w, h],
    
    [ws+721,hs+1353, w, h],
    [ws+721,hs+1531, w, h],
    [ws+721,hs+1711, w, h],
#     [ws+721,hs+1888, w, h],
]

def get_image(filename):
    image = tf.image.decode_jpeg(tf.read_file(filename), channels=1)
    with tf.Session() as sess:
        d = image.eval()
        ss = d.reshape(-1, w*h)
#         print(ss.shape)
#         img_string = ' '.join(str(s) for s in ss[0])
    return ss

def getNumImg(name):
    img = Image.open(name)
    data_list = []
    for i in range(len(box)):
        b = (box[i][0], box[i][1], box[i][0]+box[i][2], box[i][1]+box[i][3])
        roi = img.crop(b)
#         print(type(roi))
        roi.save('tmp.png')
        d = get_image('tmp.png')
        d = d/255.0
#         tf.cast(d, tf.float32)
        ss = d.reshape(w*h,1)
#         image_val = tf.concat(0,ss)
        data_list.append(ss)
    X = np.vstack(data_list)
    X = X.reshape(-1, w*h, 1)
    
    return X
    
def display(name):
    img = Image.open(name)

    box = (0, 1080, 1330, 2060)
    roi = img.crop(box)
#     sq = img.shape[0]
#     print("sq:%s %s" % (sq, img.shape))
    
#     one_image = img.reshape(int(sq), int(sq))   
    plt.axis('off')                         
    plt.imshow(roi)
    plt.show()
    
def getNum(fileName):
    
    test_list = getNumImg(fileName)
    name_str = [
        '数字0',
        '数字1',
        '数字2',
        '数字3',
        '数字4',
        '数字5',
        '数字6',
        '数字7',
        '数字8',
        '数字9',
        ]
    num_label = []
    with tf.gfile.FastGFile('./num.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        out_tensor_name = 'layer/out/p/output:0'
        out_data_tensor = sess.graph.get_tensor_by_name(out_tensor_name)
        batch_xs = test_list[0:10,...]
        batch_xs = batch_xs.reshape(-1, w*h)
        data_tensor = sess.run(out_data_tensor,{'input/x:0':batch_xs,'input/k:0':1.0})
        for i in range(10):
            predictions = np.squeeze(data_tensor[i])
            top_k = predictions.argsort()[-1:][::-1]
            for node_id in top_k:     
                score = predictions[node_id]
                num_label.append(node_id)
        #         print(score)
#                 print('位置 %s %s (识别概率：%.5f)' % (i,name_str[int(node_id)], score))
#             print()
    return num_label