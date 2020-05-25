# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:19:15 2020

@author: XZ-WAVE
"""

import numpy as np
import scipy as sp
from scipy.stats import unitary_group,ortho_group
import matplotlib.pyplot as plt
import heapq
import random 
from gate_objects import PauliX,PauliY,PauliZ,CNOT,root_SWAP,SWAP,phase_gate


complex_exp = lambda angle: np.exp(complex(0,angle))
def reshape_gate(gate,num_qubits=1):
    return gate.reshape(2,2)


def gate_fitness(qubit,gate,target):
    A = np.matmul(gate,qubit)
    B = target
    
    result = np.linalg.norm(A-B)
    fitness = np.tanh(1/(result+1e-10))
    return fitness

def make_gate(num_chroms):
    identity = np.array([1,0,0,1])
    pauli_x = np.array([0,1,1,0])
    pauli_y = np.array([0,complex(0,-1),complex(0,1),0])
    pauli_z = np.array([1,0,0,-1])
    root_NOT = (0.5)*np.array([(1+complex(0,1)),
                                1-complex(0,1),
                                1-complex(0,1),
                                1+complex(0,1)])
    T_gate = np.array([1,0,0,np.exp(complex(0,np.pi/4))])
    S_gate = np.array([1,0,0,np.exp(complex(0,np.pi/2))])
    H_gate = 1/(np.sqrt(2))*np.array([1,1,1,-1])
    
    gate_choices = [identity,pauli_x,pauli_y,pauli_z,root_NOT,T_gate,S_gate,H_gate]
    chroms = []
    for chrom in range(num_chroms):
        chroms.append(random.choice(seq=gate_choices))
    return chroms
    '''
    
    entries = 2**(num_qubits+1)
    divisions = int(num_steps)/2
    angles = np.arange(start=0,stop=(2*np.pi)+(np.pi/divisions),step=np.pi/divisions)
    #angles = [0,np.pi,2*np.pi]
    angle_choices = np.random.choice(a=angles,size=(num_chroms,entries))
    vals = np.vectorize(complex_exp)*np.random.uniform(0,1)
    elems = vals(angle_choices)
    return elems
    '''

def make_gates(num_chroms,num_qubits=2):
    swap = np.array([1,0,0,0,
                     0,0,1,0,
                     0,1,0,0,
                     0,0,0,1])
    root_swap = np.array([1,0,0,0,
                          0,complex(0.5,0.5),complex(0.5,-0.5),0,
                          0,complex(0.5,-0.5),complex(0.5,0.5),0,
                          0,0,0,1])
    cnot = np.array([1,0,0,0,
                     0,1,0,0,
                     0,0,0,1,
                     0,0,1,0])
    
    control_Z = np.array([1,0,0,0,
                      0,0,1,0,
                      0,1,0,0,
                      0,0,0,-1])
    
    angles = np.arange(0,2*np.pi+np.pi/4,step=np.pi/4)
    angle = np.random.choice(angles)
    
    ising_XX = np.array([np.cos(angle),0,0,complex(0,-1)*np.sin(angle),
                         0,np.cos(angle),complex(0,-1)*np.sin(angle),0,
                         0,complex(0,-1)*np.sin(angle),np.cos(angle),0,
                         complex(0,-1)*np.sin(angle),0,0,np.cos(angle)])
    
    ising_YY = np.array([np.cos(angle),0,0,complex(0,1)*np.sin(angle),
                         0,np.cos(angle),complex(0,-1)*np.sin(angle),0,
                         0,complex(0,-1)*np.sin(angle),np.cos(angle),0,
                         complex(0,1)*np.sin(angle),0,0,np.cos(angle)])
    
    ising_ZZ = np.array([np.exp(complex(0,angle/2)),0,0,0,
                         0,np.exp(complex(0,-angle/2)),0,0,
                         0,0,np.exp(complex(0,-angle/2)),0,
                         0,0,0,np.exp(complex(0,angle/2))])
    
    chroms = []
    gate_choices = [swap,root_swap,cnot,control_Z,ising_XX,ising_YY,ising_ZZ]
    for chrom in range(num_chroms):
        chroms.append(random.choice(seq=gate_choices))
    return chroms
    

def make_quantum_gates(num_qubits,num_chroms=10):
    chroms = []
    n = 2**num_qubits
    for chrom in range(num_chroms):
        chroms.append(unitary_group.rvs(n))
    return chroms 
    
def evolve_gate(q_in,q_out,num_chroms=10,num_generations=100,elite_frac=0.1):
    q_in = np.array(q_in)
    q_out = np.array(q_out)
    num_elite = int(num_chroms*elite_frac)
    fitness = []
    gates = make_gate(num_chroms)
    for gen in range(num_generations):
        scores = []
        new_pop = []
        for chrom in range(num_chroms):
            gate = gates[chrom]
            gate = gate.reshape(2,2)
            transform = np.matmul(gate,q_in)
            score = gate_fitness(qubit=q_in,
                                 gate=gate,
                                 target=q_out)
            
            print('Generation: {}/{} Chromosome: {}/{}'.format(gen+1,num_generations,chrom+1,num_chroms))
            print('Quantum gate:\n {}\n'.format(gate.reshape(2,2)))
            print('Qubit: {}\nTarget Qubit: {}\nResult: {}'.format(q_in,q_out,transform))
            print('Fitness: {}'.format(score))
            scores.append(score)
            plt.title('qubit state: {}'.format(q_in))
            plt.imshow(np.real(q_in.reshape(2,1))+np.imag(q_in.reshape(2,1)),cmap='bwr')
            plt.show()
            plt.clf()
            plt.title('target: {}'.format(q_out))
            plt.imshow(np.real(q_out.reshape(2,1))+np.imag(q_out.reshape(2,1)),cmap='bwr')
            plt.show()
            plt.clf()
            plt.title('result: {}'.format(transform))
            plt.imshow(np.real(transform.reshape(2,1))+np.imag(transform.reshape(2,1)),cmap='bwr')
            plt.show()
            plt.clf()
        elites = []
        fitness.append(max(scores))
        elites.append(heapq.nlargest(n=num_elite,iterable=scores))
        best_idxs = np.argpartition(scores,-num_elite)[num_elite:]
        for elite in sorted(best_idxs):
            new_pop.append(gates[elite])
       
        new_chroms = make_gate(num_chroms=num_chroms-num_elite)
        new_pop =  np.vstack((np.asarray(new_pop),new_chroms))
    return fitness


def evolve_gates(q_in,q_out,num_chroms=10,num_generations=100,elite_frac=0.1,plot=False):
    q_in = np.array(q_in)
    q_out = np.array(q_out)
    num_elite = int(num_chroms*elite_frac)
    fitness = []
    gates = make_gates(num_chroms)
    for gen in range(num_generations):
        scores = []
        new_pop = []
        for chrom in range(num_chroms):
            gate = gates[chrom]
            gate = gate.reshape(4,4)
            transform = np.matmul(gate,q_in)
            score = gate_fitness(qubit=q_in,
                                 gate=gate,
                                 target=q_out)
            
            print('Generation: {}/{} Chromosome: {}/{}'.format(gen+1,num_generations,chrom+1,num_chroms))
            print('Quantum gate:\n {}\n'.format(gate.reshape(4,4)))
            print('Qubit: {}\nTarget Qubit: {}\nResult: {}'.format(q_in,q_out,transform))
            print('Fitness: {}'.format(score))
            scores.append(score)
            if plot === True:
                
                plt.title('qubit state: {}'.format(q_in))
                plt.imshow(np.real(q_in.reshape(2,2))+np.imag(q_in.reshape(2,2)),cmap='bwr')
                plt.show()
                plt.clf()
                plt.title('target: {}'.format(q_out))
                plt.imshow(np.real(q_out.reshape(2,2))+np.imag(q_out.reshape(2,2)),cmap='bwr')
                plt.show()
                plt.clf()
                plt.title('result: {}'.format(transform))
                plt.imshow(np.real(transform.reshape(2,2))+np.imag(transform.reshape(2,2)),cmap='bwr')
                plt.show()
                plt.clf()
            
        elites = []
        fitness.append(max(scores))
        elites.append(heapq.nlargest(n=num_elite,iterable=scores))
        best_idxs = np.argpartition(scores,-num_elite)[num_elite:]
        for elite in sorted(best_idxs):
            new_pop.append(gates[elite])
       
        new_chroms = make_gates(num_chroms=num_chroms-num_elite)
        new_pop =  np.vstack((np.asarray(new_pop),new_chroms))
    return fitness




def evolve_quantum_gates(q_in,q_out,num_qubits=3,num_chroms=10,num_generations=50,elite_frac=0.1,plot=True):
    
    if len(q_in) != len(q_out):
        print('input and target qubit system have mismatched size')
        return 0
    
    if 2**num_qubits != len(q_in):
        print('number of qubits is incorrect given the input size')
        return 0
    
    q_in = np.array(q_in)
    q_out = np.array(q_out)
    num_elite = int(num_chroms*elite_frac)
    max_fitness = []
    avg_fitness = []
    gates = make_quantum_gates(num_qubits=num_qubits,num_chroms=num_chroms)
    for gen in range(num_generations):
        scores = []
        new_pop = []
        for chrom in range(num_chroms):
            
            gate = gates[chrom]
            transform = np.matmul(gate,q_in)
            score = gate_fitness(qubit=q_in,
                                 gate=gate,
                                 target=q_out)
            
            print('Generation: {}/{} Chromosome: {}/{}'.format(gen+1,num_generations,chrom+1,num_chroms))
            print('Quantum gate:\n {}\n'.format(gate))
            print('Qubit: {}\nTarget Qubit: {}\nResult: {}'.format(q_in,q_out,transform))
            print('Fitness: {}'.format(score))
            scores.append(score)
            if plot == True:
                
                plt.title('qubit state: {}'.format(q_in))
                plt.imshow(np.real(q_in.reshape(len(q_in),1))+np.imag(q_in.reshape(len(q_in),1)),cmap='bwr')
                plt.show()
                plt.clf()
                plt.title('target: {}'.format(q_out))
                plt.imshow(np.real(q_out.reshape(len(q_out),1))+np.imag(q_out.reshape(len(q_out),1)),cmap='bwr')
                plt.show()
                plt.clf()
                plt.title('result: {}'.format(transform))
                plt.imshow(np.real(transform.reshape(len(transform),1))+np.imag(transform.reshape(len(transform),1)),cmap='bwr')
                plt.show()
                plt.clf()
            
        elites = []
        max_fitness.append(max(scores))
        avg_fitness.append(np.mean(scores))
        
        elites.append(heapq.nlargest(n=num_elite,iterable=scores))
        best_idxs = np.argpartition(scores,-num_elite)[num_elite:]
        for elite in sorted(best_idxs):
            new_pop.append(gates[elite])
       
        new_chroms = make_quantum_gates(num_chroms=num_chroms-num_elite,num_qubits=num_qubits)
        new_pop =  np.vstack((np.asarray(new_pop),new_chroms))
        
    return max_fitness

