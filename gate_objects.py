# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:21:33 2020

@author: XZ-WAVE
"""

import numpy as np
from abc import ABCMeta, abstractmethod
import scipy as sp

class Gate(): #eventually might want to make all gates inherit from an abstract class
    __metaclass__ = ABCMeta
    def __init__(self,matrix):
        super(Gate,self).__init__()
        self.mat = matrix
        try:
            pass
        except:
            pass
    

    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result


class PauliX():
    def __init__(self):
        super(PauliX,self).__init__()
        self.mat = np.array([0,1,1,0])
        self.mat = self.mat.reshape(2,2)
    
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result

class PauliY():
    def __init__(self):
        super(PauliX,self).__init__()
        self.mat = np.array([0,complex(0,-1),complex(0,1),0])
        self.mat = self.mat.reshape(2,2)
    
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result

class PauliZ():
    def __init__(self):
        super(PauliX,self).__init__()
        self.mat = np.array([1,0,0,-1])
        self.mat = self.mat.reshape(2,2)
    
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result

class root_NOT():
    def __init__(self):
        super(root_NOT,self).__init__()
        self.mat = (0.5)*np.array([(1+complex(0,1)),
                                1-complex(0,1),
                                1-complex(0,1),
                                1+complex(0,1)])
        self.mat = self.mat.reshape(2,2)
    
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result

class Hadamard():
    def __init__(self,num_qubits=2):
        super(Hadamard,self).__init__()
        self.num_qubits = num_qubits
        self.factor = (1/np.sqrt(2))**num_qubits
        self.mat = self.factor*sp.linalg.hadamard(n=num_qubits)
        
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result
    
class CNOT():
    def __init__(self):
        super(CNOT,self).__init__()
        self.mat = np.array([1,0,0,0,
                     0,1,0,0,
                     0,0,0,1,
                     0,0,1,0])
        self.mat = self.mat.reshape(4,4)
    
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result

class ControlZ():
    def __init__(self):
        super(ControlZ,self).__init__()
        self.mat = np.array([1,0,0,0,
                      0,0,1,0,
                      0,1,0,0,
                      0,0,0,-1])
        self.mat = self.mat.reshape(4,4)
        
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result
    
class SWAP():
    def __init__(self):
        super(SWAP,self).__init__()
        self.mat = np.array([1,0,0,0,
                     0,0,1,0,
                     0,1,0,0,
                     0,0,0,1])
        self.mat.reshape(4,4)
        
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result

class root_SWAP():
    def __init__(self):
        super(root_SWAP,self).__init__()
        self.mat = np.array([1,0,0,0,
                          0,complex(0.5,0.5),complex(0.5,-0.5),0,
                          0,complex(0.5,-0.5),complex(0.5,0.5),0,
                          0,0,0,1])
        self.mat.reshape(4,4)
        
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result
    
class phase_gate():
    def __init__(self,angle):
        super(phase_gate,self).__init__()
        self.angle = angle
        self.mat =  np.array([1,0,0,np.exp(complex(0,self.angle))])
        self.mat = self.mat.reshape(2,2)
        
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result

class ising_XX():
    def __init__(self,angle):
        super(ising_XX,self).__init__()
        self.angle = angle
        self.mat = np.array([np.cos(self.angle),0,0,complex(0,-1)*np.sin(self.angle),
                         0,np.cos(self.angle),complex(0,-1)*np.sin(self.angle),0,
                         0,complex(0,-1)*np.sin(self.angle),np.cos(self.angle),0,
                         complex(0,-1)*np.sin(self.angle),0,0,np.cos(self.angle)])
        self.mat = self.mat.reshape(4,4)
        
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result

class ising_YY():
    def __init__(self,angle):
        super(ising_XX,self).__init__()
        self.angle = angle
        self.mat = np.array([np.cos(self.angle),0,0,complex(0,1)*np.sin(self.angle),
                         0,np.cos(self.angle),complex(0,-1)*np.sin(self.angle),0,
                         0,complex(0,-1)*np.sin(self.angle),np.cos(self.angle),0,
                         complex(0,1)*np.sin(self.angle),0,0,np.cos(self.angle)])
    
        self.mat = self.mat.reshape(4,4)
        
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result

class ising_ZZ():
    def __init__(self,angle):
        super(ising_XX,self).__init__()
        self.angle = angle
        self.mat = np.array([np.exp(complex(0,self.angle/2)),0,0,0,
                         0,np.exp(complex(0,-self.angle/2)),0,0,
                         0,0,np.exp(complex(0,-self.angle/2)),0,
                         0,0,0,np.exp(complex(0,self.angle/2))])
    
        self.mat = self.mat.reshape(4,4)
        
    def apply(self,q):
        self.result = np.matmul(self.mat,q)
        return self.result