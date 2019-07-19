#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:18:01 2018

@author: work
"""

'''
1.reshape all vectors

pass params 
get all W and b
flatten ?
concatenate all 

do the same with the gradients


'''


import numpy as np
import functools         # for reduce function
        

def gradient_check(params,grads, epsilon=1e-7):
    
    #theta = flatten_concat(params)
    #d_theta = flatten_concat(grads)
    theta = list_to_vector(params)
    print('theta: ',theta)
    
    for i in range(len(theta)):
        theta_plus[i] = theta[i]-epsilon
        theta_minus[i] = theta[i]+epsilon
        


    print('theta: ',theta)
    #    
    #dla kazdego elementu z theta dodaj do niego epsilon
  #  a co z grads??????
    #potem flatten concat
        
    
    
    return flatten_concat(params)
       
def list_to_vector(list_of_arrays):
    vector = np.empty(0)
    for e in list_of_arrays:
        vector = np.concatenate((vector,e.flatten()))
    return vector
 
def vector_to_list(vector, template):
    l = []
    curr_pos = 0;
    for e in template:
        length = functools.reduce(lambda x,y: x*y, e.shape)    # to calculate how long the flatten was
        l.append( vector[curr_pos:curr_pos+length].reshape(e.shape) )
        curr_pos = curr_pos + length
    return l
        
def flatten_concat(list_of_params):
    output = np.empty(0)
    for p in list_of_params:
        print(p.shape)
        output = np.concatenate((output,p.flatten()))
    return output

def compare_two_lists(l1,l2):
    if len(l1) != len (l2):
        return False
    for i in range(len(l1)):
        if (l1[i] == l2[i]).all():
            None
        else:
            return False
    return True

'''

x = [np.zeros((3,2)),np.ones((1,2))]
x2 = [np.ones((3,2)),np.ones((1,4))]
a = list_to_vector(x)
b = vector_to_list(a,x)

print(b)

'''

import os
import matplotlib.pyplot as plt

def save_matrix_as_img(A,path,name):
    if path!='':
        if not os.path.exists(path):
            os.makedirs(path)
    plt.matshow(A, cmap='Greys')
    fn = str(path) + str(name) + '.png'
    plt.savefig(fn)
    plt.close(plt.gcf())        # to not to display images

def generate_video(inp,output):
    # input_files - "output/images/img_%d.png"
    command = "ffmpeg -r 10 -i '" + str(inp) + "' -vcodec mpeg4 -y '" + str(output) +"'"
    os.system(command)
    return None