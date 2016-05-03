# coding: utf-8

import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag


class RBF(object):

    def __init__(self,observation_space_high,observation_space_low,feature_number,variance,action_number):
        self.space_high = np.array(observation_space_high)
        self.space_low = np.array(observation_space_low)
        self.state_number = len(np.array(observation_space_high))
        self.feature_number = feature_number
        self.all_feature_number = feature_number**self.state_number
        self.variance = variance
        self.action_number = action_number

    def FeatureElement_list(self):
        feture_space = np.vstack((self.space_high,self.space_low)).T
        return np.array([np.linspace(space[0],space[1],self.feature_number) for space in feture_space])

    def BlockMatrics_convert(self,a,b):
        matrix = np.ones(a)
        repeat_vector_add = np.ones(a)
        for i in range(b-1):
            matrix = block_diag(matrix,repeat_vector_add)
        return matrix

    def FeatureAverage_list(self):
        feature_element = self.FeatureElement_list()
        element_number,element_length = np.shape(feature_element)
        feature_average = np.zeros((element_number,element_length**(element_number)))
        for i in range(element_number):
            feature = feature_element[i]
            repeat_number = element_length**(element_number-i-1)
            convert_matrix = self.BlockMatrics_convert(repeat_number,element_length)
            feature_average_part = np.dot(feature,convert_matrix)
            feature_average_add = feature_average_part
            for j in range(element_length**(element_number-1)/repeat_number-1):
                feature_average_part = np.hstack((feature_average_part,feature_average_add))
            feature_average[i] = feature_average_part
        return feature_average.T

    def mnd(self,_x,_mu,_sig):
        sig = np.matrix(_sig)
        a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**len(sig))
        _b = np.array(np.dot((_x-_mu),sig.I))
        b = np.sum(-0.5*_b*(_x-_mu),axis=1)
        return np.exp(b)/a

    def StateFeatureFunction(self,state,feature_list):
        return mnd(self,state,feature_list,self.variance)

    def ActionFeatureFunction(self,action):
        action_matrix = np.zeros((self.all_feature_number,self.action_number))
        action_matrix.T[action] = np.ones(self.all_feature_number)
        return action_matrix

    def RBF_list(self,state,action,feature_list):
        return self.ActionFeatureFunction(action)*self.StateFeatureFunction(state,feature_list).T.flatten()\

    def RBF_network(self,parameter,state,action,feature_list):
        return np.dot(parameter,self.RBF_list(state,action,feature_list))

    
