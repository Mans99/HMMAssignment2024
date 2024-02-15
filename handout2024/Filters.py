
import random
import numpy as np

from models import *


#
# Add your Filtering / Smoothing approach(es) here
#
class HMMFilter:
    def __init__(self, probs, tm, om, sm):
        self.__tm = tm #transition model
        self.__om = om #observation model
        self.__sm = sm #sensor model
        self.__f = probs #initial prob
        
        
    def filter(self, sensorR) :
        Od = self.__om.get_o_reading(sensorR)
        T = self.__tm.get_T_transp()
        self.__f = Od @ T @ self.__f
        self.__f /= np.sum(self.__f)
 
        return self.__f

    def fb_smooth(self, sensor):
        f_est = self.filter(sensor.pop(0))
        b = np.ones(self.__sm.get_num_of_states())
        for i in reversed(sensor):
            Od = self.__om.get_o_reading(i)
            T = self.__tm.get_T()
            b = T@Od@b
        s = f_est * b
        s /= np.sum(s)
        
        return s
        
        

        
        
        
