"""
Traveling Salesman Problem (TSP) Solver using QAOA

Problem Definition:
Find the shortest route visiting all cities exactly once and returning to start.

Mathematical Formulation:
Minimize: C = Σ_{i,j} d_{ij} * x_{ij}
where x_{ij} = 1 if edge (i,j) is in tour, 0 otherwise

Constraints:
- Each city visited exactly once
- Valid tour (connected path)
"""

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from typing import Dict, List, Tuple
from qaoa_core import QAOAOptimizer
import logging
import itertools

logger = logging.getLogger(__name__)

class TSPSolver:
    """
    QAOA-based Traveling Salesman Problem Solver
    """
    
    def __init__(self, distance_matrix: np.ndarray, p_layers: int = 2):
        """
        Initialize TSP solver
        
        Args:
            distance_matrix: n x n matrix of distances between cities
            p_layers: Number of QAOA layers
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.p_layers = p_layers
        
        # Number of qubits needed (approximate encoding)
        # For n cities, we need roughly n^2 qubits for full encoding
        # Using reduced encoding for practical quantum hardware
        self.num_qubits = min(self.num_cities * (self.num_cities - 1), 20)  # Cap at 20 qubits
        
        # Initialize QAOA
        self.qaoa = QAOAOptimizer(self.num_qubits, p_layers)
        
    def cost_hamiltonian(self, qc: QuantumCircuit, gamma) -> None:
        """
        Apply cost Hamiltonian for TSP
        
        H_C encodes:
        1. Distance minimization
        2. Visit each city once constraint
        3. Path connectivity constraint
        
        Args:
            qc: Quantum circuit
            gamma: Cost parameter
        """
        # Simplified TSP Hamiltonian for demonstration
        # Full TSP requires complex encoding
        
        # Distance term: penalize longer edges
        qubit_idx = 0
        for i in range(min(self.num_cities, 5)):  # Limit for practical quantum hardware
            for j in range(i + 1, min(self.num_cities, 5)):
                if qubit_idx < self.num_qubits:
                    distance = self.distance_matrix[i][j]
                    # Apply ZZ interaction weighted by distance
                    if qubit_idx + 1 < self.num_qubits:
                        qc.cx(qubit_idx, qubit_idx + 1)
                        qc.rz(2 * gamma * distance / 10.0, qubit_idx + 1)  # Normalize
                        qc.cx(qubit_idx, qubit_idx + 1)
                    qubit_idx += 1
    
    def decode_solution(self, bitstring: str) -> List[int]:
        """
        Decode bitstring to tour
        
        Args:
            bitstring: Binary string from quantum measurement
            
        Returns:
            Tour as list of city indices
        """
        # Simplified decoding for demonstration
        # Real TSP would need sophisticated decoding
        
        tour = list(range(self.num_cities))
        
        # Use bitstring to permute tour
        for i, bit in enumerate(bitstring[:self.num_cities]):
            if bit == '1' and i + 1 < self.num_cities:
                # Swap adjacent cities
                tour[i], tour[i + 1] = tour[i + 1], tour[i]
        
        return tour
    
    def compute_tour_cost(self, tour: List[int]) -> float:
        """
        Compute total tour distance
        
        Args:
            tour: List of city indices in visit order
            
        Returns:
            Total distance
        """
        if len(tour) < 2:
            return 0.0
        
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += self.distance_matrix[tour[i]][tour[i + 1]]
        
        # Return to start
        cost += self.distance_matrix[tour[-1]][tour[0]]
        
        return cost
    
    def cost_function(self, bitstring: str) -> float:
        """
        Cost function for optimization
        
        Args:
            bitstring: Binary string
            
        Returns:
            Tour cost
        """
        tour = self.decode_solution(bitstring)
        return self.compute_tour_cost(tour)
    
    def solve(self, method: str = 'COBYLA', max_iter: int = 50) -> Dict:
        """
        Solve TSP using QAOA
        
        Args:
            method: Optimization method
            max_iter: Maximum iterations
            
        Returns:
            Solution dictionary
        """
        logger.info(f"Solving TSP for {self.num_cities} cities")
        
        # Create QAOA circuit
        circuit = self.qaoa.create_qaoa_circuit(self.cost_hamiltonian)
        
        # Optimize
        opt_result = self.qaoa.optimize(circuit, self.cost_function, method, max_iter)
        
        # Get solution probabilities
        probs = self.qaoa.get_solution_probabilities(circuit, opt_result['optimal_params'])
        
        # Find best solution
        best_bitstring = max(probs.items(), key=lambda x: x[1])[0]
        best_tour = self.decode_solution(best_bitstring)
        best_cost = self.compute_tour_cost(best_tour)
        
        # Compute classical bound (greedy nearest neighbor)
        classical_tour, classical_cost = self._greedy_nearest_neighbor()
        
        # Optimal solution (brute force for small instances)
        optimal_cost = self._compute_optimal() if self.num_cities <= 7 else classical_cost
        
        return {
            'success': opt_result['success'],
            'best_tour': best_tour,
            'tour_cost': best_cost,
            'optimal_params': opt_result['optimal_params'].tolist(),
            'iterations': opt_result['iterations'],
            'classical_comparison': {
                'classical_tour': classical_tour,
                'classical_cost': classical_cost,
                'optimal_cost': optimal_cost,
                'qaoa_approximation_ratio': best_cost / optimal_cost if optimal_cost > 0 else 1.0
            },
            'problem_info': {
                'num_cities': self.num_cities,
                'num_qubits': self.num_qubits,
                'p_layers': self.p_layers
            }
        }
    
    def _greedy_nearest_neighbor(self) -> Tuple[List[int], float]:
        """
        Greedy nearest neighbor heuristic
        
        Returns:
            (tour, cost)
        """
        unvisited = set(range(self.num_cities))
        tour = [0]  # Start from city 0
        unvisited.remove(0)
        
        while unvisited:
            current = tour[-1]
            nearest = min(unvisited, key=lambda city: self.distance_matrix[current][city])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        cost = self.compute_tour_cost(tour)
        return tour, cost
    
    def _compute_optimal(self) -> float:
        """
        Compute optimal solution via brute force (only for small instances)
        
        Returns:
            Optimal tour cost
        """
        if self.num_cities > 7:
            return float('inf')
        
        cities = list(range(self.num_cities))
        min_cost = float('inf')
        
        for perm in itertools.permutations(cities[1:]):
            tour = [0] + list(perm)
            cost = self.compute_tour_cost(tour)
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    @staticmethod
    def generate_random_instance(num_cities: int, max_distance: int = 100) -> np.ndarray:
        """
        Generate random TSP instance
        
        Args:
            num_cities: Number of cities
            max_distance: Maximum distance between cities
            
        Returns:
            Distance matrix
        """
        # Random city coordinates
        coords = np.random.rand(num_cities, 2) * max_distance
        
        # Compute Euclidean distances
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
        
        return distance_matrix
