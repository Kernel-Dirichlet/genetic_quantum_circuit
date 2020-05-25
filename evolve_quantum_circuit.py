# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:27:03 2020

@author: XZ-WAVE
"""

import argparse 
import numpy as np
from genetic_quantum_programming import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evolutionary Quantum Circuits')
    parser.add_argument('--num_generations',type=int,default=100,help='number of generations')
    parser.add_argument('--num_chromosomes',type=int,default=50,help='number of chrosomes')
    parser.add_argument('--num_qubits',type=int,default=3,help='number of qubits to evolve')
    parser.add_argument('--elite_frac',type=float,default=0.1,help='elite fraction')
    
    args = parser.parse_args()
    q_in = np.random.choice(a=[0,1],size=2**args.num_qubits)
    q_out = np.random.choice(a=[0,1],size=2**args.num_qubits)
    fitness=evolve_quantum_gates(q_in=q_in,
                         q_out=q_out,
                         num_qubits=args.num_qubits,num_chroms=args.num_chromosomes,
                         num_generations=args.num_generations,
                         elite_frac=args.elite_frac,
                         plot=False) #keep plot set to False
    
    print('best fitness:{}'.format(fitness[-1]))
        
    