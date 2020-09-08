# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:10:27 2020

@author: Qiuhao Wu
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow import keras
import numpy as np
from scipy.misc import  imresize
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images.reshape(train_images.shape[0],-1) / 255.0
test_images = test_images.reshape(test_images.shape[0],-1) / 255.0
train_labels=np.eye(10)[train_labels]
test_labels=np.eye(10)[test_labels]

learning_rate = 0.001
batch_size=128
size=28
#4f system can be considered as a linear interconection
def sf_system(u,w):
    U=tf.fft2d(u)
    #W=tf.fft2d(w)
    W=tf.fft2d(tf.cast(w,dtype=tf.complex64))
    return tf.ifft2d(U*W)

def make_random(shape):
    return np.random.random(size = shape).astype('float32')

def rang(arr,shape,size=size,base =28):
    x0 = shape[0] * size // base
    y0 = shape[2] * size // base
    delta = (shape[1]-shape[0])* size // base
    return arr[x0:x0+delta,y0:y0+delta]
    #return arr[shape[0]*size//base:shape[1]*size//base,shape[2]*size//base:shape[3]*size//base]
def reduce_mean(tf_):
    return tf.reduce_mean(tf_)
def _ten_regions(a):
    return tf.map_fn(reduce_mean,tf.convert_to_tensor([
        rang(a,(6,9,6,9)),
        rang(a,(6,9,12,15)),
        rang(a,(6,9,18,21)),
        rang(a,(12,15,3,6)),
        rang(a,(12,15,10,13)),
        rang(a,(12,15,17,20)),
        rang(a,(12,15,23,26)),
        rang(a,(18,21,6,9)),
        rang(a,(18,21,12,15)),
        rang(a,(18,21,18,21))
    ])) 

def ten_regions(logits):
    return tf.map_fn(_ten_regions,tf.abs(logits),dtype=tf.float32)

def download_text(msg,epoch,MIN=1,MAX=7,name=''):
    for i in range(1,7):
        np.savetxt("{}_Time_{}_layer_{}.txt".format(name,epoch+1,i),msg[i-1])
    print("Done")
        
def download_image(msg,epoch,MIN=1,MAX=7,name=''):
    print(f"Plot images-[{MIN}:{MAX}]")
    for i in range(MIN,MAX):
        #print("Image {}:".format(i))
        plt.figure(dpi=650.24)
        plt.axis('off')
        plt.grid('off')
        plt.imshow(msg[i-1])
        plt.savefig("{}_Time_{}_layer_{}.jpg".format(name,epoch+1,i))
        #print("Done")

data_x = tf.placeholder(tf.float32,shape=(None,size,size))
data_y = tf.placeholder(tf.float32,shape=(None,10))

def next_batch(batch_size, fake_data=False, shuffle=True):
    self_images = train_images
    self_labels = train_labels
    self_epochs_completed =0
    self_index_in_epoch = 0 
    self_num_examples=len(train_images)
    start = self_index_in_epoch
    if self_epochs_completed == 0 and start == 0 and shuffle:
        perm0 = np.arange(self_num_examples)  
        np.random.shuffle(perm0)
        self_images = train_images[perm0]
        self_labels = train_labels[perm0]
        # Go to the next epoch


    if start + batch_size > self_num_examples:
        self_epochs_completed += 1
            # Get the rest examples in this epoch
        rest_num_examples = self_num_examples - start  
        images_rest_part = self_images[start:self_num_examples]
        labels_rest_part = self_labels[start:self_num_examples]
        # Shuffle the data
        if shuffle: 
            perm = np.arange(self_num_examples)
            np.random.shuffle(perm)
            self_images = train_images[perm]
            self_labels = train_labels[perm]
            # Start next epoch
        start = 0
        self_index_in_epoch = batch_size - rest_num_examples
        end = self_index_in_epoch
        images_new_part = self_images[start:end] 
        labels_new_part = self_labels[start:end]
        return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)    
    else:  
        self_index_in_epoch += batch_size 
        end = self_index_in_epoch 
        return self_images[start:end], self_labels[start:end] 

weight=[
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32)]
#forward propagation
layer1=tf.cast(tf.nn.softmax(tf.abs(sf_system(tf.cast(data_x,dtype=tf.complex64),weight[0]))),dtype=tf.complex64)
layer2=tf.cast(tf.nn.softmax(tf.abs(sf_system(layer1,weight[1]))),dtype=tf.complex64)
layer3=tf.cast(tf.nn.softmax(tf.abs(sf_system(layer2,weight[2]))),dtype=tf.complex64)
layer4=tf.cast(tf.nn.softmax(tf.abs(sf_system(layer3,weight[3]))),dtype=tf.complex64)
layer5=tf.cast(tf.nn.softmax(tf.abs(sf_system(layer4,weight[4]))),dtype=tf.complex64)
out=sf_system(layer5,weight[5])

logits_abs =ten_regions(out)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_abs,labels=data_y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
pre_correct = tf.equal(tf.argmax(data_y,1),tf.argmax(logits_abs,1))
accuracy = tf.reduce_mean(tf.cast(pre_correct,tf.float32))

init = tf.global_variables_initializer()
train_epochs =50
loss_list=[]
acc_list=[]
session = tf.Session()
with tf.device('/gpu:0'):
    session.run(init)
    total_batch = int(len(train_images) / batch_size)

    for epoch in tqdm(range(train_epochs)):
        for batch in tqdm(range(total_batch)):
            batch_x,batch_y = next_batch(batch_size)
            batch_x=batch_x.reshape((batch_size,size,size))
            session.run(train_op,feed_dict={data_x:batch_x,data_y:batch_y})

        loss_,acc = session.run([loss,accuracy],feed_dict={data_x:batch_x,data_y:batch_y})
        loss_list.append(loss_)
        acc_list.append(acc)
        print("epoch :{} loss:{:.4f} train_acc:{:.4f}".format(epoch+1,loss_,acc))
print("Optimizer finished")


fig,ax1=plt.subplots()
ax2=ax1.twinx()
Ins1=ax1.plot(np.arange(50),loss_list,label='Loss')
Ins2=ax2.plot(np.arange(50),acc_list,'r',label='Accuracy')
ax1.set_xlabel('interation')
ax1.set_ylabel('training loss')
ax2.set_ylabel('training accuracy')

Ins=Ins1+Ins2
labels=['Loss','Accuracy']
plt.legend(Ins,labels,loc=7)
plt.show()
weight_= np.array(session.run(weight)) 
download_text(weight_,epoch,name='weight')
download_image(weight_,epoch,name='weight')

test_data=test_images.reshape((-1,size,size))
test_label=test_labels
print("test_accuracy",\
      session.run(accuracy,feed_dict={data_x:test_data,data_y:test_label}))
pred_y = session.run(tf.argmax(logits_abs, 1), feed_dict={data_x:test_data})
m=session.run(tf.argmax(test_label, 1))
cm=confusion_matrix(m,pred_y)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    
plt.imshow(cm, interpolation='nearest')    
plt.title('Normalized Confusion Matrix')    
plt.colorbar()
plt.ylabel('True label')    
plt.xlabel('Predicted label')
plt.show()

