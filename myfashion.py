# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 09:55:02 2021

@author: qg
"""

import tensorflow as tf
from tensorflow import keras
 
import numpy as np
import matplotlib.pyplot as plt
 
#下载数据集
fashion_mnist = keras.datasets.fashion_mnist 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#对数据类别集命名
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat',
             'Sandal','Shirt','Sneaker','Bag','Ankle boot']
#打印训练数据集和测试数据集标签
print("The shape of train_images is ",train_images.shape)
print("The shape of train_labels is ",train_labels.shape)
print("The shape of test_images is ",test_images.shape)
print("The length of test_labels is ",len(test_labels))

#可视化原始数据训练集的第一个样本
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()

#特征的值从整数变成浮点数,其次我们在除以255.

train_images=train_images/255.0
test_images=test_images/255.0
#可视化归一化之后的数据的代码

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#构建模型
'''
下面的代码很显然存在三层,第一层是tf.keras.layers.Flatten(),主要的功能就是将(28,28)像素的图像即对应的2维的数组转成28*28=784的一维的数组.
第二层是Dense,这个层存在128的神经元.
最后一层是softmax层,它返回的是由10个概率值(加起来等于1)组成的1维数组,
每一个代表了这个图像属于某个类别的概率.

'''
#1.配置层
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(512,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
#2.编译模型
'''
优化器是AdamOptimizer(表示采用何种方式寻找最佳答案,有什么梯度下降啊等等),
损失函数是sparse_categorical_crossentropy(就是损失函数怎么定义的,最佳值就是使得损失函数最小).计量标准是准确率,也就是正确归类的图像的概率.

'''
model.compile(tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#3.训练模型
model.fit(train_images,train_labels,epochs=10)
'''

'''

#评估准确率

test_loss,test_acc=model.evaluate(test_images,test_labels)
print('Test Acc:',test_acc)
#预测
predictions=model.predict(test_images)
print("The first picture's prediction is:{},so the result is:{}".format(predictions[0],np.argmax(predictions[0])))
print("The first picture is ",test_labels[0])

#绘制25个样本的预测结果
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                class_names[true_label]),
               color=color)
plt.show()

'''
我们还可以只预测其中一个样本,拿第一个测试集样本举例,你会发现与之前的答案一致,都是9.

第一个空行前面的代码,是在数据集上取得一个样本,根据结果可以得到我们的图像是28*28像素的.---(28.28)

第二个空行前面的代码是为了什么呢?看第二部分我们知道,我们的数据集的大小,因为tf.keras模型对一批数据进行预测结果的优化,所以,我们对于单独的一张图片,我们需要将其加入list中,其实就是增加一个维度.---(1,28,28)

'''

img=test_images[0]
print(img.shape)
 
img=(np.expand_dims(img,0))
print(img.shape)
 
predictions=model.predict(img)
print(predictions)
 
prediction=predictions[0]
print("You will find that the ans is same to the former res",np.argmax(prediction))