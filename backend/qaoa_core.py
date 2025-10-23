"""
QAOA Core Algorithm Implementation
Quantum Approximate Optimization Algorithm for solving NP-hard combinatorial problems

Key Features:
- Variational quantum circuit construction
- Parameterized gates (RX, RY, RZ, CNOT)
- Cost and Mixer Hamiltonians
- Multi-layer (p-depth) support
- Parameter optimization with classical feedback
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from scipy.optimize import minimize
import networkx as nx
from typing import List, Tuple, Dict, Callable
import logging

logger = logging.getLogger(__name__)

class QAOAOptimizer:
    """
    Quantum Approximate Optimization Algorithm Implementation
    
    Mathematical Foundation:
    |ψ(β, γ)⟩ = U(B, β_p)U(C, γ_p)...U(B, β_1)U(C, γ_1)|s⟩
    
    where:
    - U(C, γ) = e^(-iγC) is the cost unitary
    - U(B, β) = e^(-iβB) is the mixer unitary
    - C is the problem Hamiltonian
    - B is the mixer Hamiltonian
    """
    
    def __init__(self, num_qubits: int, p_layers: int = 1, backend_type: str = 'simulator'):
        """
        Initialize QAOA optimizer
        
        Args:
            num_qubits: Number of qubits (problem size)
            p_layers: Number of QAOA layers (depth)
            backend_type: 'simulator' or 'ibmq'
        """
        self.num_qubits = num_qubits
        self.p_layers = p_layers
        self.backend_type = backend_type
        self.backend = Aer.get_backend('qasm_simulator')
        
        # Parameters
        self.beta_params = [Parameter(f'β_{i}') for i in range(p_layers)]
        self.gamma_params = [Parameter(f'γ_{i}') for i in range(p_layers)]
        
        # Results storage
        self.optimization_history = []
        self.best_params = None
        self.best_cost = float('inf')
        
    def create_qaoa_circuit(self, cost_hamiltonian_func: Callable, 
                           initial_state: str = 'superposition') -> QuantumCircuit:
        """
        Create QAOA quantum circuit
        
        Args:
            cost_hamiltonian_func: Function that applies cost Hamiltonian
            initial_state: 'superposition' or 'custom'
            
        Returns:
            Parameterized quantum circuit
        """
        qr = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(qr)
        
        # Initial state preparation (|+⟩^⊗n)
        if initial_state == 'superposition':
            qc.h(range(self.num_qubits))
        
        # QAOA layers
        for layer in range(self.p_layers):
            # Cost Hamiltonian: U(C, γ)
            cost_hamiltonian_func(qc, self.gamma_params[layer])
            
            # Mixer Hamiltonian: U(B, β) = ∏ e^(-iβX_i)
            for qubit in range(self.num_qubits):
                qc.rx(2 * self.beta_params[layer], qubit)
        
        # Measurement
        qc.measure_all()
        
        return qc
    
    def compute_expectation(self, counts: Dict[str, int], cost_function: Callable) -> float:
        """
        Compute expectation value ⟨ψ|C|ψ⟩
        
        Args:
            counts: Measurement counts
            cost_function: Function to compute cost for each bitstring
            
        Returns:
            Expectation value
        """
        total_counts = sum(counts.values())
        expectation = 0.0
        
        for bitstring, count in counts.items():
            probability = count / total_counts
            cost = cost_function(bitstring)
            expectation += probability * cost
        
        return expectation
    
    def optimize(self, circuit: QuantumCircuit, cost_function: Callable,
                method: str = 'COBYLA', max_iter: int = 100) -> Dict:
        """
        Classical optimization of QAOA parameters
        
        Args:
            circuit: QAOA circuit
            cost_function: Cost function for expectation calculation
            method: Optimization method ('COBYLA', 'SLSQP', 'Nelder-Mead')
            max_iter: Maximum iterations
            
        Returns:
            Optimization results
        """
        self.optimization_history = []
        
        def objective_function(params):
            """Objective function to minimize"""
            # Bind parameters
            param_dict = {}
            for i in range(self.p_layers):
                param_dict[self.beta_params[i]] = params[i]
                param_dict[self.gamma_params[i]] = params[self.p_layers + i]
            
            bound_circuit = circuit.assign_parameters(param_dict)
            
            # Execute circuit
            job = self.backend.run(bound_circuit, shots=1024)
            result = job.result()
            counts = result.get_counts()
            
            # Compute expectation
            expectation = self.compute_expectation(counts, cost_function)
            
            # Store history
            self.optimization_history.append({
                'params': params.copy(),
                'cost': expectation,
                'counts': counts
            })
            
            # Update best
            if expectation < self.best_cost:
                self.best_cost = expectation
                self.best_params = params.copy()
            
            return expectation
        
        # Initial parameters (random or informed)
        initial_params = self._get_initial_params()
        
        # Optimize
        logger.info(f"Starting optimization with {method}")
        result = minimize(
            objective_function,
            initial_params,
            method=method,
            options={'maxiter': max_iter}
        )
        
        return {
            'success': result.success,
            'optimal_params': result.x,
            'optimal_cost': result.fun,
            'iterations': getattr(result, 'nit', getattr(result, 'nfev', len(self.optimization_history))),
            'history': self.optimization_history,
            'message': str(result.message) if hasattr(result, 'message') else 'Optimization complete'
        }
    
    def _get_initial_params(self, strategy: str = 'random') -> np.ndarray:
        """
        Initialize parameters
        
        Strategies:
        - random: Random initialization
        - informed: Use heuristics from literature
        - gradient: Gradient-based initialization
        """
        if strategy == 'random':
            return np.random.uniform(0, 2*np.pi, 2 * self.p_layers)
        elif strategy == 'informed':
            # Start with common good values
            beta_init = np.random.uniform(0, np.pi/2, self.p_layers)
            gamma_init = np.random.uniform(0, np.pi, self.p_layers)
            return np.concatenate([beta_init, gamma_init])
        else:
            return np.random.uniform(0, 2*np.pi, 2 * self.p_layers)
    
    def get_solution_probabilities(self, circuit: QuantumCircuit, 
                                   params: np.ndarray, shots: int = 8192) -> Dict:
        """
        Get solution probability distribution
        
        Args:
            circuit: QAOA circuit
            params: Optimized parameters
            shots: Number of measurements
            
        Returns:
            Probability distribution
        """
        # Bind parameters
        param_dict = {}
        for i in range(self.p_layers):
            param_dict[self.beta_params[i]] = params[i]
            param_dict[self.gamma_params[i]] = params[self.p_layers + i]
        
        bound_circuit = circuit.assign_parameters(param_dict)
        
        # Execute
        job = self.backend.run(bound_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Convert to probabilities
        probabilities = {k: v/shots for k, v in counts.items()}
        
        return probabilities
    
    def analyze_convergence(self) -> Dict:
        """
        Analyze optimization convergence
        
        Returns:
            Convergence metrics
        """
        if not self.optimization_history:
            return {}
        
        costs = [h['cost'] for h in self.optimization_history]
        
        return {
            'initial_cost': costs[0],
            'final_cost': costs[-1],
            'best_cost': min(costs),
            'improvement': (costs[0] - min(costs)) / costs[0] * 100,
            'iterations': len(costs),
            'convergence_rate': self._compute_convergence_rate(costs)
        }
    
    def _compute_convergence_rate(self, costs: List[float]) -> float:
        """
        Compute average convergence rate
        """
        if len(costs) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(costs)):
            if costs[i-1] != 0:
                improvement = (costs[i-1] - costs[i]) / abs(costs[i-1])
                improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
