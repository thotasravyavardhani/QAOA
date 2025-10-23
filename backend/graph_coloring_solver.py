"""
Graph Coloring Problem Solver using QAOA

Problem Definition:
Assign colors to graph vertices such that no adjacent vertices share the same color,
using minimum number of colors.

Mathematical Formulation:
Minimize number of colors k such that:
- ∀(i,j)∈E: c_i ≠ c_j (adjacent vertices have different colors)

Applications:
- Task scheduling
- Register allocation
- Frequency assignment
- Sudoku solving
"""

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from typing import Dict, List, Set
from qaoa_core import QAOAOptimizer
import logging

logger = logging.getLogger(__name__)

class GraphColoringSolver:
    """
    QAOA-based Graph Coloring Solver
    """
    
    def __init__(self, graph: nx.Graph, num_colors: int, p_layers: int = 2):
        """
        Initialize Graph Coloring solver
        
        Args:
            graph: Input graph (NetworkX)
            num_colors: Number of colors to use
            p_layers: Number of QAOA layers
        """
        self.graph = graph
        self.num_vertices = graph.number_of_nodes()
        self.num_colors = num_colors
        self.p_layers = p_layers
        
        # Encoding: n vertices × k colors = n*k qubits
        # Use log encoding for larger graphs
        if self.num_vertices * self.num_colors <= 20:
            # One-hot encoding
            self.num_qubits = self.num_vertices * self.num_colors
            self.encoding = 'one-hot'
        else:
            # Binary encoding (log_2(k) bits per vertex)
            self.num_qubits = self.num_vertices * int(np.ceil(np.log2(self.num_colors)))
            self.encoding = 'binary'
        
        # Initialize QAOA
        self.qaoa = QAOAOptimizer(self.num_qubits, p_layers)
        
        self.edges = list(graph.edges())
    
    def cost_hamiltonian(self, qc: QuantumCircuit, gamma) -> None:
        """
        Apply cost Hamiltonian for Graph Coloring
        
        H_C penalizes:
        1. Adjacent vertices with same color
        2. Invalid colorings (vertex with multiple/no colors)
        
        Args:
            qc: Quantum circuit
            gamma: Cost parameter
        """
        if self.encoding == 'one-hot':
            # One-hot encoding
            for edge in self.edges:
                u, v = edge
                # Penalize same color assignments
                for color in range(self.num_colors):
                    qubit_u = u * self.num_colors + color
                    qubit_v = v * self.num_colors + color
                    
                    if qubit_u < self.num_qubits and qubit_v < self.num_qubits:
                        # ZZ interaction: penalty if both have same color
                        qc.cx(qubit_u, qubit_v)
                        qc.rz(2 * gamma, qubit_v)
                        qc.cx(qubit_u, qubit_v)
        else:
            # Binary encoding
            bits_per_vertex = int(np.ceil(np.log2(self.num_colors)))
            
            for edge in self.edges:
                u, v = edge
                # Compare color bits
                for bit in range(bits_per_vertex):
                    qubit_u = u * bits_per_vertex + bit
                    qubit_v = v * bits_per_vertex + bit
                    
                    if qubit_u < self.num_qubits and qubit_v < self.num_qubits:
                        qc.cx(qubit_u, qubit_v)
                        qc.rz(gamma, qubit_v)
                        qc.cx(qubit_u, qubit_v)
    
    def decode_coloring(self, bitstring: str) -> Dict[int, int]:
        """
        Decode bitstring to vertex coloring
        
        Args:
            bitstring: Binary string
            
        Returns:
            Dictionary mapping vertex -> color
        """
        coloring = {}
        
        if self.encoding == 'one-hot':
            # One-hot: each vertex has k bits
            for vertex in range(self.num_vertices):
                start = vertex * self.num_colors
                end = start + self.num_colors
                
                if end <= len(bitstring):
                    vertex_bits = bitstring[start:end]
                    # Assign color of first '1' bit
                    color = vertex_bits.find('1')
                    coloring[vertex] = color if color != -1 else 0
                else:
                    coloring[vertex] = 0
        else:
            # Binary encoding
            bits_per_vertex = int(np.ceil(np.log2(self.num_colors)))
            
            for vertex in range(self.num_vertices):
                start = vertex * bits_per_vertex
                end = start + bits_per_vertex
                
                if end <= len(bitstring):
                    vertex_bits = bitstring[start:end]
                    # Convert binary to color index
                    color = int(vertex_bits, 2) % self.num_colors
                    coloring[vertex] = color
                else:
                    coloring[vertex] = 0
        
        return coloring
    
    def is_valid_coloring(self, coloring: Dict[int, int]) -> bool:
        """
        Check if coloring is valid (no adjacent vertices with same color)
        
        Args:
            coloring: Vertex -> color mapping
            
        Returns:
            True if valid
        """
        for u, v in self.edges:
            if coloring.get(u, -1) == coloring.get(v, -1):
                return False
        return True
    
    def count_conflicts(self, coloring: Dict[int, int]) -> int:
        """
        Count number of edge conflicts
        
        Args:
            coloring: Vertex -> color mapping
            
        Returns:
            Number of conflicts
        """
        conflicts = 0
        for u, v in self.edges:
            if coloring.get(u, -1) == coloring.get(v, -1):
                conflicts += 1
        return conflicts
    
    def cost_function(self, bitstring: str) -> float:
        """
        Cost function (number of conflicts)
        
        Args:
            bitstring: Binary string
            
        Returns:
            Number of conflicts
        """
        coloring = self.decode_coloring(bitstring)
        return float(self.count_conflicts(coloring))
    
    def solve(self, method: str = 'COBYLA', max_iter: int = 100) -> Dict:
        """
        Solve Graph Coloring using QAOA
        
        Args:
            method: Optimization method
            max_iter: Maximum iterations
            
        Returns:
            Solution dictionary
        """
        logger.info(f"Solving Graph Coloring: {self.num_vertices} vertices, {self.num_colors} colors")
        
        # Create QAOA circuit
        circuit = self.qaoa.create_qaoa_circuit(self.cost_hamiltonian)
        
        # Optimize
        opt_result = self.qaoa.optimize(circuit, self.cost_function, method, max_iter)
        
        # Get solution probabilities
        probs = self.qaoa.get_solution_probabilities(circuit, opt_result['optimal_params'])
        
        # Find best valid coloring
        best_coloring = None
        best_conflicts = float('inf')
        is_valid = False
        
        for bitstring, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:20]:
            coloring = self.decode_coloring(bitstring)
            conflicts = self.count_conflicts(coloring)
            
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_coloring = coloring
                is_valid = (conflicts == 0)
                
                if is_valid:
                    break
        
        # Classical comparison (greedy coloring)
        classical_coloring = self._greedy_coloring()
        classical_conflicts = self.count_conflicts(classical_coloring)
        
        # Compute chromatic number bound
        chromatic_bound = self._compute_chromatic_bound()
        
        return {
            'success': opt_result['success'] and is_valid,
            'coloring': best_coloring,
            'num_conflicts': best_conflicts,
            'is_valid': is_valid,
            'optimal_params': opt_result['optimal_params'].tolist(),
            'iterations': opt_result['iterations'],
            'classical_comparison': {
                'classical_coloring': classical_coloring,
                'classical_conflicts': classical_conflicts,
                'improvement': (classical_conflicts - best_conflicts)
            },
            'graph_metrics': {
                'num_vertices': self.num_vertices,
                'num_edges': len(self.edges),
                'num_colors': self.num_colors,
                'chromatic_bound': chromatic_bound,
                'encoding': self.encoding,
                'num_qubits': self.num_qubits,
                'p_layers': self.p_layers
            }
        }
    
    def _greedy_coloring(self) -> Dict[int, int]:
        """
        Greedy graph coloring algorithm
        
        Returns:
            Vertex -> color mapping
        """
        coloring = {}
        
        for vertex in self.graph.nodes():
            # Find colors of neighbors
            neighbor_colors = {coloring.get(neighbor) for neighbor in self.graph.neighbors(vertex)
                             if neighbor in coloring}
            
            # Assign first available color
            for color in range(self.num_colors):
                if color not in neighbor_colors:
                    coloring[vertex] = color
                    break
            else:
                # No color available, assign last color (conflict)
                coloring[vertex] = self.num_colors - 1
        
        return coloring
    
    def _compute_chromatic_bound(self) -> int:
        """
        Compute upper bound on chromatic number
        
        Returns:
            Upper bound (max degree + 1)
        """
        if self.graph.number_of_nodes() == 0:
            return 0
        
        max_degree = max(dict(self.graph.degree()).values()) if self.graph.number_of_edges() > 0 else 1
        return max_degree + 1
    
    @staticmethod
    def generate_random_graph(num_vertices: int, edge_probability: float = 0.3) -> nx.Graph:
        """
        Generate random graph for testing
        
        Args:
            num_vertices: Number of vertices
            edge_probability: Probability of edge creation
            
        Returns:
            Random graph
        """
        return nx.gnp_random_graph(num_vertices, edge_probability)
