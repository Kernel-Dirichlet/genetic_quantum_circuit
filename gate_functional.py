# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:40:15 2020

@author: XZ-WAVE
"""

import numpy as np
import scipy as sp

def pauli_x(q):
    '''
    Pauli-X gate (Quantum version of the classical "NOT" gate)
    q: list or np.array -> single qubit
    '''
    gate = np.array([0,1,1,0])
    gate = gate.reshape(2,2)
    new_state = np.matmul(gate,q)
    return new_state

def pauli_y(q):
    '''
    Pauli-Y gate
    q: list or np.array -> single qubit
    '''
    gate =  np.array([0,complex(0,-1),complex(0,1),0])
    gate = gate.reshape(2,2)
    new_state = np.matmul(gate,q)
    return new_state

def pauli_z(q):
    '''
    Pauli-Z gate
    q: list or np.array -> single qubit
    '''
    gate = np.array([1,0,0,-1])
    gate = gate.reshape(2,2)
    new_state = np.matmul(gate,q)
    return new_state

def root_not(q):
    '''
    Square root of the "NOT" gate
    q: list or np.array -> single qubit
    '''
    gate = (0.5)*np.array([(1+complex(0,1)),
                                1-complex(0,1),
                                1-complex(0,1),
                                1+complex(0,1)])
    gate = gate.reshape(2,2)
    new_state = np.matmul(gate,q)
    return new_state

def phase_gate(q,angle):
    '''
    phase gate
    angle: float between 0 and 2*pi, representing the phase of the qubit (in radians)
    q: list or np.array -> single qubit
    '''
    gate = np.array([1,0,0,np.exp(complex(0,angle))])
    gate = gate.reshape(2,2)
    new_state = np.matmul(gate,q)
    return new_state

def Hadamard(q,num_qubits):
    '''
    Hadamard matrix
    q: list or np.array -> constructs a hadamard matrix based on the number of qubits, and acts on the qubit system
    eventually want this function to create the matrix based on q alone
    num_qubits: int
    '''
    factor = (1/np.sqrt(2))**num_qubits
    gate = factor*sp.linalg.hadamard(n=2**num_qubits)
    try:
        new_state = np.matmul(gate,q)
        return new_state
    except:
        print('error, the qubit size and matrix dim do not match')

def CNOT(q):
    '''
    CNOT gate
    q: list or np.array -> two qubits
    '''
    gate = np.array([1,0,0,0,
                     0,1,0,0,
                     0,0,0,1,
                     0,0,1,0])
    gate = gate.reshape(4,4)
    new_state = np.matmul(gate,q)
    return new_state

def SWAP(q):
    '''
    SWAP gate
    q: list or np.array -> two qubits
    '''
    gate = np.array([1,0,0,0,
                     0,0,1,0,
                     0,1,0,0,
                     0,0,0,1])
    gate = gate.reshape(4,4)
    new_state = np.matmul(gate,q)
    return new_state

def root_SWAP(q):
    '''
    Square root of the SWAP gate
    q: list or np.array -> two qubits
    '''
    gate = np.array([1,0,0,0,
                          0,complex(0.5,0.5),complex(0.5,-0.5),0,
                          0,complex(0.5,-0.5),complex(0.5,0.5),0,
                          0,0,0,1])
    gate = gate.reshape(4,4)
    new_state = np.matmul(gate,q)
    return new_state

def control_z(q):
    '''
    control-Z gate
    q: list or np.array -> two qubits
    '''
    gate = np.array([1,0,0,0,
                      0,0,1,0,
                      0,1,0,0,
                      0,0,0,-1])
    gate = gate.reshape(4,4)
    new_state = np.matmul(gate,q)
    return new_state

def ising_xx(q,angle=np.pi):
    '''
    Ising-XX gate
    q: list or np.array -> two qubits
    angle: float between 0 and 2*pi
    '''
    gate = np.array([np.cos(angle),0,0,complex(0,-1)*np.sin(angle),
                         0,np.cos(angle),complex(0,-1)*np.sin(angle),0,
                         0,complex(0,-1)*np.sin(angle),np.cos(angle),0,
                         complex(0,-1)*np.sin(angle),0,0,np.cos(angle)])
    gate = gate.reshape(4,4)
    new_state = np.matmul(gate,q)
    return new_state

def ising_yy(q,angle=np.pi):
    '''
    Ising-YY gate
    q: list or np.array -> two qubits
    angle: float between 0 and 2*pi
    '''
    gate = np.array([np.cos(angle),0,0,complex(0,1)*np.sin(angle),
                         0,np.cos(angle),complex(0,-1)*np.sin(angle),0,
                         0,complex(0,-1)*np.sin(angle),np.cos(angle),0,
                         complex(0,1)*np.sin(angle),0,0,np.cos(angle)])
    gate = gate.reshape(4,4)
    new_state = np.matmul(gate,q)
    return new_state

def ising_zz(q,angle=np.pi):
    '''
    Ising-ZZ gate
    q: list or np.array -> two qubits
    angle: float between 0 and 2*pi
    '''
    gate = np.array([np.exp(complex(0,angle/2)),0,0,0,
                         0,np.exp(complex(0,-angle/2)),0,0,
                         0,0,np.exp(complex(0,-angle/2)),0,
                         0,0,0,np.exp(complex(0,angle/2))])
    gate = gate.reshape(4,4)
    new_state = np.matmul(gate,q)
    return new_state

    