"""
Max-Cut Problem Solver using QAOA

Problem Definition:
Given a graph G = (V, E), partition vertices into two sets S and S̄
to maximize the number of edges between the sets.

Mathematical Formulation:
C(z) = Σ_{(i,j)∈E} w_{ij} * (1 - z_i*z_j) / 2

where z_i ∈ {-1, +1} is the spin of vertex i
"""

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from typing import Dict, List, Tuple
from qaoa_core import QAOAOptimizer
import logging

logger = logging.getLogger(__name__)

class MaxCutSolver:
    """
    QAOA-based Max-Cut Problem Solver
    """
    
    def __init__(self, graph: nx.Graph, p_layers: int = 2):
        """
        Initialize Max-Cut solver
        
        Args:
            graph: Input graph (NetworkX)
            p_layers: Number of QAOA layers
        """
        self.graph = graph
        self.num_vertices = graph.number_of_nodes()
        self.p_layers = p_layers
        
        # Initialize QAOA
        self.qaoa = QAOAOptimizer(self.num_vertices, p_layers)
        
        # Store edge information
        self.edges = list(graph.edges())
        self.edge_weights = nx.get_edge_attributes(graph, 'weight')
        
        # Default weight is 1 if not specified
        for edge in self.edges:
            if edge not in self.edge_weights:
                self.edge_weights[edge] = 1.0
    
    def cost_hamiltonian(self, qc: QuantumCircuit, gamma) -> None:
        """
        Apply cost Hamiltonian for Max-Cut
        
        H_C = Σ_{(i,j)∈E} w_{ij} * (1 - Z_i⊗Z_j) / 2
        
        Implementation using ZZ gates:
        e^(-iγH_C) = ∏_{(i,j)∈E} e^(-iγw_{ij}(1-Z_i⊗Z_j)/2)
        
        Args:
            qc: Quantum circuit
            gamma: Cost parameter
        """
        for edge in self.edges:
            i, j = edge
            weight = self.edge_weights.get(edge, 1.0)
            
            # Apply ZZ interaction: CNOT-RZ-CNOT
            qc.cx(i, j)
            qc.rz(2 * gamma * weight, j)
            qc.cx(i, j)
    
    def compute_cut_value(self, bitstring: str) -> int:
        """
        Compute cut value for a given partition
        
        Args:
            bitstring: Binary string representing partition
            
        Returns:
            Number of edges cut
        """
        cut_value = 0
        partition = [int(bit) for bit in bitstring]
        
        for edge in self.edges:
            i, j = edge
            # Edge is cut if vertices are in different partitions
            if partition[i] != partition[j]:
                weight = self.edge_weights.get(edge, 1.0)
                cut_value += weight
        
        return cut_value
    
    def cost_function(self, bitstring: str) -> float:
        """
        Cost function for optimization (negative cut value for minimization)
        
        Args:
            bitstring: Binary string
            
        Returns:
            Negative cut value
        """
        return -self.compute_cut_value(bitstring)
    
    def solve(self, method: str = 'COBYLA', max_iter: int = 100) -> Dict:
        """
        Solve Max-Cut using QAOA
        
        Args:
            method: Optimization method
            max_iter: Maximum iterations
            
        Returns:
            Solution dictionary
        """
        logger.info(f"Solving Max-Cut for graph with {self.num_vertices} vertices, {len(self.edges)} edges")
        
        # Create QAOA circuit
        circuit = self.qaoa.create_qaoa_circuit(self.cost_hamiltonian)
        
        # Optimize
        opt_result = self.qaoa.optimize(circuit, self.cost_function, method, max_iter)
        
        # Get solution probabilities
        probs = self.qaoa.get_solution_probabilities(circuit, opt_result['optimal_params'])
        
        # Find best solution
        best_bitstring = max(probs.items(), key=lambda x: x[1])[0]
        best_cut_value = self.compute_cut_value(best_bitstring)
        
        # Compute classical upper bound
        classical_bound = self._compute_classical_bound()
        
        # Analyze results
        convergence = self.qaoa.analyze_convergence()
        
        return {
            'success': opt_result['success'],
            'best_partition': best_bitstring,
            'cut_value': best_cut_value,
            'optimal_params': opt_result['optimal_params'].tolist(),
            'iterations': opt_result['iterations'],
            'probability_distribution': dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]),
            'convergence_analysis': convergence,
            'classical_bound': classical_bound,
            'approximation_ratio': best_cut_value / classical_bound if classical_bound > 0 else 0,
            'graph_info': {
                'num_vertices': self.num_vertices,
                'num_edges': len(self.edges),
                'p_layers': self.p_layers
            }
        }
    
    def _compute_classical_bound(self) -> float:
        """
        Compute classical upper bound using greedy algorithm
        
        Returns:
            Classical cut value
        """
        # Simple greedy approach
        partition = [0] * self.num_vertices
        
        # Assign vertices to maximize cut
        for v in range(self.num_vertices):
            cut_0 = sum(self.edge_weights.get((v, u), self.edge_weights.get((u, v), 1.0)) 
                       for u in self.graph.neighbors(v) if partition[u] == 1)
            cut_1 = sum(self.edge_weights.get((v, u), self.edge_weights.get((u, v), 1.0)) 
                       for u in self.graph.neighbors(v) if partition[u] == 0)
            partition[v] = 1 if cut_1 > cut_0 else 0
        
        # Compute cut value
        cut_value = sum(self.edge_weights.get(edge, 1.0) 
                       for edge in self.edges 
                       if partition[edge[0]] != partition[edge[1]])
        
        return cut_value
    
    @staticmethod
    def generate_random_graph(num_vertices: int, edge_probability: float = 0.5, 
                            weighted: bool = False) -> nx.Graph:
        """
        Generate random graph for testing
        
        Args:
            num_vertices: Number of vertices
            edge_probability: Probability of edge creation
            weighted: Whether to add random weights
            
        Returns:
            Random graph
        """
        G = nx.gnp_random_graph(num_vertices, edge_probability)
        
        if weighted:
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = np.random.randint(1, 10)
        
        return G
