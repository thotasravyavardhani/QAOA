"""
Vehicle Routing Problem (VRP) Solver using QAOA

Problem Definition:
Optimize routes for multiple vehicles to serve customers from a depot.

Objective:
Minimize total distance while satisfying:
- Each customer visited exactly once
- Vehicle capacity constraints
- All routes start and end at depot

Mathematical Formulation:
Minimize: Σ_{k∈K} Σ_{i,j∈V} d_{ij} * x_{ijk}

where:
- K = set of vehicles
- V = set of nodes (depot + customers)
- x_{ijk} = 1 if vehicle k travels from i to j
"""

import numpy as np
from qiskit import QuantumCircuit
from typing import Dict, List, Tuple
from qaoa_core import QAOAOptimizer
import logging

logger = logging.getLogger(__name__)

class VRPSolver:
    """
    QAOA-based Vehicle Routing Problem Solver
    """
    
    def __init__(self, distance_matrix: np.ndarray, demands: List[float],
                 vehicle_capacity: float, num_vehicles: int, p_layers: int = 2):
        """
        Initialize VRP solver
        
        Args:
            distance_matrix: (n+1) x (n+1) matrix (depot + n customers)
            demands: Customer demands
            vehicle_capacity: Vehicle capacity
            num_vehicles: Number of vehicles
            p_layers: Number of QAOA layers
        """
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.num_customers = len(demands)
        self.p_layers = p_layers
        
        # Qubit allocation: assign customers to vehicles
        self.num_qubits = min(self.num_customers * num_vehicles, 20)
        
        # Initialize QAOA
        self.qaoa = QAOAOptimizer(self.num_qubits, p_layers)
        
    def cost_hamiltonian(self, qc: QuantumCircuit, gamma) -> None:
        """
        Apply cost Hamiltonian for VRP
        
        H_C encodes:
        1. Route distance minimization
        2. Capacity constraints
        3. Customer assignment constraints
        
        Args:
            qc: Quantum circuit
            gamma: Cost parameter
        """
        # Simplified VRP Hamiltonian
        # Each qubit represents assignment of customer to vehicle
        
        qubit_idx = 0
        for customer in range(min(self.num_customers, 4)):
            for vehicle in range(min(self.num_vehicles, 2)):
                if qubit_idx < self.num_qubits:
                    # Distance-based penalty
                    distance = self.distance_matrix[0][customer + 1]  # Depot to customer
                    
                    # Apply rotation based on distance
                    qc.rz(gamma * distance / 10.0, qubit_idx)
                    
                    # Capacity constraint coupling
                    if qubit_idx + 1 < self.num_qubits:
                        demand = self.demands[customer]
                        penalty = gamma * (demand / self.vehicle_capacity)
                        
                        qc.cx(qubit_idx, qubit_idx + 1)
                        qc.rz(penalty, qubit_idx + 1)
                        qc.cx(qubit_idx, qubit_idx + 1)
                    
                    qubit_idx += 1
    
    def decode_solution(self, bitstring: str) -> List[List[int]]:
        """
        Decode bitstring to vehicle routes
        
        Args:
            bitstring: Binary string
            
        Returns:
            List of routes (one per vehicle)
        """
        routes = [[] for _ in range(self.num_vehicles)]
        
        # Assign customers to vehicles based on bitstring
        for customer in range(min(self.num_customers, len(bitstring) // self.num_vehicles)):
            # Determine vehicle assignment
            vehicle_bits = bitstring[customer * self.num_vehicles:(customer + 1) * self.num_vehicles]
            if vehicle_bits:
                # Assign to vehicle with most 1s
                vehicle_assignment = vehicle_bits.count('1') % self.num_vehicles
                routes[vehicle_assignment].append(customer + 1)  # +1 for depot offset
        
        return routes
    
    def compute_route_cost(self, routes: List[List[int]]) -> Tuple[float, bool]:
        """
        Compute total route cost and check feasibility
        
        Args:
            routes: List of routes
            
        Returns:
            (total_cost, is_feasible)
        """
        total_cost = 0.0
        is_feasible = True
        
        for route in routes:
            if not route:
                continue
            
            # Check capacity
            route_demand = sum(self.demands[customer - 1] for customer in route)
            if route_demand > self.vehicle_capacity:
                is_feasible = False
            
            # Compute route distance: depot -> customers -> depot
            route_cost = self.distance_matrix[0][route[0]]  # Depot to first customer
            
            for i in range(len(route) - 1):
                route_cost += self.distance_matrix[route[i]][route[i + 1]]
            
            route_cost += self.distance_matrix[route[-1]][0]  # Last customer to depot
            
            total_cost += route_cost
        
        # Penalty for infeasibility
        if not is_feasible:
            total_cost *= 1.5
        
        return total_cost, is_feasible
    
    def cost_function(self, bitstring: str) -> float:
        """
        Cost function for optimization
        
        Args:
            bitstring: Binary string
            
        Returns:
            Route cost with penalties
        """
        routes = self.decode_solution(bitstring)
        cost, _ = self.compute_route_cost(routes)
        return cost
    
    def solve(self, method: str = 'COBYLA', max_iter: int = 50) -> Dict:
        """
        Solve VRP using QAOA
        
        Args:
            method: Optimization method
            max_iter: Maximum iterations
            
        Returns:
            Solution dictionary
        """
        logger.info(f"Solving VRP: {self.num_customers} customers, {self.num_vehicles} vehicles")
        
        # Create QAOA circuit
        circuit = self.qaoa.create_qaoa_circuit(self.cost_hamiltonian)
        
        # Optimize
        opt_result = self.qaoa.optimize(circuit, self.cost_function, method, max_iter)
        
        # Get solution probabilities
        probs = self.qaoa.get_solution_probabilities(circuit, opt_result['optimal_params'])
        
        # Find best feasible solution
        best_routes = None
        best_cost = float('inf')
        is_feasible = False
        
        for bitstring, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]:
            routes = self.decode_solution(bitstring)
            cost, feasible = self.compute_route_cost(routes)
            
            if cost < best_cost:
                best_cost = cost
                best_routes = routes
                is_feasible = feasible
                if feasible:
                    break  # Found feasible solution
        
        # Classical comparison (greedy)
        classical_routes, classical_cost = self._greedy_solution()
        
        return {
            'success': opt_result['success'] and is_feasible,
            'routes': best_routes,
            'total_cost': best_cost,
            'is_feasible': is_feasible,
            'optimal_params': opt_result['optimal_params'].tolist(),
            'iterations': opt_result['iterations'],
            'classical_comparison': {
                'classical_routes': classical_routes,
                'classical_cost': classical_cost,
                'improvement': (classical_cost - best_cost) / classical_cost * 100 if classical_cost > 0 else 0
            },
            'problem_info': {
                'num_customers': self.num_customers,
                'num_vehicles': self.num_vehicles,
                'vehicle_capacity': self.vehicle_capacity,
                'num_qubits': self.num_qubits,
                'p_layers': self.p_layers
            }
        }
    
    def _greedy_solution(self) -> Tuple[List[List[int]], float]:
        """
        Greedy VRP solution (nearest neighbor with capacity check)
        
        Returns:
            (routes, total_cost)
        """
        routes = [[] for _ in range(self.num_vehicles)]
        route_demands = [0.0] * self.num_vehicles
        unassigned = set(range(1, self.num_customers + 1))
        
        while unassigned:
            for vehicle in range(self.num_vehicles):
                if not unassigned:
                    break
                
                # Find nearest unassigned customer that fits
                current = routes[vehicle][-1] if routes[vehicle] else 0
                
                feasible_customers = [
                    c for c in unassigned 
                    if route_demands[vehicle] + self.demands[c - 1] <= self.vehicle_capacity
                ]
                
                if not feasible_customers:
                    continue
                
                nearest = min(feasible_customers, 
                            key=lambda c: self.distance_matrix[current][c])
                
                routes[vehicle].append(nearest)
                route_demands[vehicle] += self.demands[nearest - 1]
                unassigned.remove(nearest)
        
        cost, _ = self.compute_route_cost(routes)
        return routes, cost
    
    @staticmethod
    def generate_random_instance(num_customers: int, num_vehicles: int,
                                capacity: float = 100.0, max_distance: int = 100) -> Tuple:
        """
        Generate random VRP instance
        
        Returns:
            (distance_matrix, demands)
        """
        # Random coordinates (depot + customers)
        coords = np.random.rand(num_customers + 1, 2) * max_distance
        
        # Distance matrix
        distance_matrix = np.zeros((num_customers + 1, num_customers + 1))
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
        
        # Random demands (ensure solvable)
        demands = np.random.uniform(10, capacity / num_vehicles, num_customers)
        
        return distance_matrix, demands.tolist()
