"""Variational Quantum Eigensolver for protein energy minimization."""

import numpy as np
import torch
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import pennylane as qml
from scipy.optimize import minimize


@dataclass
class VQEResult:
    """Results from VQE optimization."""
    energy: float
    optimal_params: np.ndarray
    convergence_history: List[float]
    num_iterations: int
    success: bool


class VariationalQuantumEigensolver:
    """VQE implementation for protein folding energy minimization.
    
    Uses parameterized quantum circuits to find ground state energies
    of protein Hamiltonians constructed from symbolic rules.
    """
    
    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 3,
        backend: str = 'default.qubit',
        shots: Optional[int] = None,
        optimizer: str = 'adam'
    ):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.backend = backend
        self.shots = shots
        self.optimizer_name = optimizer
        
        # Initialize quantum device
        self.dev = qml.device(backend, wires=num_qubits, shots=shots)
        
        # Create quantum circuit
        self.circuit = self._build_circuit()
        
        # Initialize parameters
        self.params = self._init_parameters()
        
    def _build_circuit(self) -> Callable:
        """Construct parameterized quantum circuit."""
        
        @qml.qnode(self.dev, interface='torch')
        def circuit(params, hamiltonian):
            """Hardware-efficient ansatz with entangling layers."""
            num_params_per_layer = self.num_qubits * 3  # RX, RY, RZ per qubit
            
            # Initial state preparation
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
            
            # Variational layers
            for layer in range(self.num_layers):
                start_idx = layer * num_params_per_layer
                
                # Single-qubit rotations
                for i in range(self.num_qubits):
                    param_offset = start_idx + i * 3
                    qml.RX(params[param_offset], wires=i)
                    qml.RY(params[param_offset + 1], wires=i)
                    qml.RZ(params[param_offset + 2], wires=i)
                
                # Entangling layer (circular connectivity)
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.num_qubits > 1:
                    qml.CNOT(wires=[self.num_qubits - 1, 0])
            
            return qml.expval(hamiltonian)
        
        return circuit
    
    def _init_parameters(self) -> np.ndarray:
        """Initialize variational parameters."""
        num_params = self.num_qubits * 3 * self.num_layers
        # Small random initialization near identity
        return np.random.randn(num_params) * 0.1
    
    def optimize(
        self,
        hamiltonian: qml.Hamiltonian,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        learning_rate: float = 0.01
    ) -> VQEResult:
        """Optimize circuit parameters to find ground state energy.
        
        Args:
            hamiltonian: Quantum Hamiltonian to minimize
            max_iterations: Maximum optimization steps
            tolerance: Convergence threshold
            learning_rate: Step size for gradient descent
            
        Returns:
            VQEResult with optimal energy and parameters
        """
        params = torch.tensor(self.params, requires_grad=True)
        optimizer = self._get_optimizer(params, learning_rate)
        
        convergence_history = []
        prev_energy = float('inf')
        
        for iteration in range(max_iterations):
            # Forward pass
            energy = self.circuit(params, hamiltonian)
            
            # Backward pass
            energy.backward()
            
            # Optimization step
            optimizer.step()
            optimizer.zero_grad()
            
            # Track convergence
            current_energy = energy.item()
            convergence_history.append(current_energy)
            
            # Check convergence
            if abs(current_energy - prev_energy) < tolerance:
                return VQEResult(
                    energy=current_energy,
                    optimal_params=params.detach().numpy(),
                    convergence_history=convergence_history,
                    num_iterations=iteration + 1,
                    success=True
                )
            
            prev_energy = current_energy
        
        return VQEResult(
            energy=prev_energy,
            optimal_params=params.detach().numpy(),
            convergence_history=convergence_history,
            num_iterations=max_iterations,
            success=False
        )
    
    def _get_optimizer(self, params: torch.Tensor, learning_rate: float):
        """Get PyTorch optimizer."""
        if self.optimizer_name.lower() == 'adam':
            return torch.optim.Adam([params], lr=learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            return torch.optim.SGD([params], lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def compute_gradient(self, hamiltonian: qml.Hamiltonian) -> np.ndarray:
        """Compute gradient using parameter shift rule."""
        gradient = np.zeros_like(self.params)
        shift = np.pi / 2
        
        for i in range(len(self.params)):
            # Shift parameter forward
            params_plus = self.params.copy()
            params_plus[i] += shift
            
            # Shift parameter backward
            params_minus = self.params.copy()
            params_minus[i] -= shift
            
            # Parameter shift rule
            energy_plus = self.circuit(torch.tensor(params_plus), hamiltonian).item()
            energy_minus = self.circuit(torch.tensor(params_minus), hamiltonian).item()
            
            gradient[i] = (energy_plus - energy_minus) / 2
        
        return gradient


class AdaptiveVQE(VariationalQuantumEigensolver):
    """Adaptive VQE that grows circuit depth based on convergence."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_layers = 1
        self.max_layers = 10
        
    def optimize_adaptive(
        self,
        hamiltonian: qml.Hamiltonian,
        target_accuracy: float = 1e-4
    ) -> VQEResult:
        """Adaptively grow circuit depth until target accuracy achieved."""
        
        for num_layers in range(self.min_layers, self.max_layers + 1):
            self.num_layers = num_layers
            self.circuit = self._build_circuit()
            self.params = self._init_parameters()
            
            result = self.optimize(hamiltonian)
            
            if result.success and abs(result.energy) > target_accuracy:
                return result
        
        # Return best result even if target not reached
        return result


class HamiltonianBuilder:
    """Constructs quantum Hamiltonians from symbolic protein rules."""
    
    @staticmethod
    def from_distance_matrix(
        distance_matrix: np.ndarray,
        interaction_strength: float = 1.0
    ) -> qml.Hamiltonian:
        """Build Hamiltonian from pairwise distance constraints.
        
        Args:
            distance_matrix: N×N matrix of desired distances
            interaction_strength: Coupling strength
            
        Returns:
            PennyLane Hamiltonian
        """
        n = distance_matrix.shape[0]
        coeffs = []
        observables = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] > 0:
                    # Attractive interaction for residues that should be close
                    coeff = -interaction_strength / distance_matrix[i, j]
                    coeffs.append(coeff)
                    observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
        
        # Add local fields
        for i in range(n):
            coeffs.append(0.1)
            observables.append(qml.PauliX(i))
        
        return qml.Hamiltonian(coeffs, observables)
    
    @staticmethod
    def from_symbolic_rules(
        rules: List,
        num_residues: int
    ) -> qml.Hamiltonian:
        """Build Hamiltonian from symbolic constraints.
        
        Args:
            rules: List of SymbolicRule objects
            num_residues: Total number of residues
            
        Returns:
            PennyLane Hamiltonian encoding structural rules
        """
        coeffs = []
        observables = []
        
        for rule in rules:
            if rule.rule_type == 'hydrogen_bond':
                i, j = rule.residue_indices
                # Strong attractive interaction
                coeffs.append(-2.0 * rule.confidence)
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
                
            elif rule.rule_type == 'hydrophobic_core':
                # All-to-all interactions within core
                indices = rule.residue_indices
                for idx1 in range(len(indices)):
                    for idx2 in range(idx1 + 1, len(indices)):
                        i, j = indices[idx1], indices[idx2]
                        coeffs.append(-1.0 * rule.confidence)
                        observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
            
            elif rule.rule_type == 'secondary_structure':
                # Local ordering for secondary structures
                indices = rule.residue_indices
                for idx in range(len(indices) - 1):
                    i, j = indices[idx], indices[idx + 1]
                    coeffs.append(-0.5 * rule.confidence)
                    observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
        
        # Add local bias terms
        for i in range(num_residues):
            coeffs.append(0.1)
            observables.append(qml.PauliX(i))
        
        return qml.Hamiltonian(coeffs, observables)
