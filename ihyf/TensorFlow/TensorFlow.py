# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 9:45
# @Author  : ihyf
# @File    : TensorFlow.py
# @Software: PyCharm
# @ Desc :
import tensorflow as tf

messages = tf.constant("Hello World!")
with tf.Session() as sess:
    print(sess.run(messages).decode())
