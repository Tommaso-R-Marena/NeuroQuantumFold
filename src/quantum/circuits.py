"""Parameterized quantum circuits for protein folding."""

import pennylane as qml
import numpy as np
from typing import List, Callable, Optional


class ProteinFoldingCircuit:
    """Base class for protein folding quantum circuits."""
    
    def __init__(self, num_qubits: int, num_layers: int = 3):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
    def get_num_parameters(self) -> int:
        """Calculate total number of variational parameters."""
        raise NotImplementedError
    
    def build(self) -> Callable:
        """Build the quantum circuit."""
        raise NotImplementedError


class HardwareEfficientAnsatz(ProteinFoldingCircuit):
    """Hardware-efficient ansatz optimized for NISQ devices."""
    
    def get_num_parameters(self) -> int:
        return self.num_qubits * 3 * self.num_layers
    
    def build(self, device: qml.Device) -> Callable:
        """Build hardware-efficient circuit."""
        
        @qml.qnode(device)
        def circuit(params, hamiltonian):
            self._apply_layers(params)
            return qml.expval(hamiltonian)
        
        return circuit
    
    def _apply_layers(self, params: np.ndarray):
        """Apply variational layers."""
        param_idx = 0
        
        for layer in range(self.num_layers):
            # Rotation layer
            for qubit in range(self.num_qubits):
                qml.RX(params[param_idx], wires=qubit)
                qml.RY(params[param_idx + 1], wires=qubit)
                qml.RZ(params[param_idx + 2], wires=qubit)
                param_idx += 3
            
            # Entanglement layer
            for i in range(0, self.num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, self.num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])


class ChemistryInspiredAnsatz(ProteinFoldingCircuit):
    """Ansatz inspired by molecular orbital theory."""
    
    def get_num_parameters(self) -> int:
        # Includes single and double excitations
        num_single = self.num_qubits * 2
        num_double = (self.num_qubits * (self.num_qubits - 1)) // 2
        return (num_single + num_double) * self.num_layers
    
    def build(self, device: qml.Device) -> Callable:
        """Build chemistry-inspired circuit."""
        
        @qml.qnode(device)
        def circuit(params, hamiltonian):
            # Initial Hartree-Fock state
            for i in range(self.num_qubits // 2):
                qml.PauliX(wires=i)
            
            # Excitation layers
            self._apply_excitations(params)
            
            return qml.expval(hamiltonian)
        
        return circuit
    
    def _apply_excitations(self, params: np.ndarray):
        """Apply single and double excitation gates."""
        param_idx = 0
        
        for layer in range(self.num_layers):
            # Single excitations
            for i in range(self.num_qubits // 2):
                for a in range(self.num_qubits // 2, self.num_qubits):
                    qml.SingleExcitation(params[param_idx], wires=[i, a])
                    param_idx += 1
            
            # Double excitations (limited for efficiency)
            for i in range(min(2, self.num_qubits // 2)):
                for j in range(i + 1, min(3, self.num_qubits // 2)):
                    for a in range(self.num_qubits // 2, min(self.num_qubits // 2 + 2, self.num_qubits)):
                        for b in range(a + 1, min(self.num_qubits // 2 + 3, self.num_qubits)):
                            if param_idx < len(params):
                                qml.DoubleExcitation(params[param_idx], wires=[i, j, a, b])
                                param_idx += 1


class SymmetryPreservingAnsatz(ProteinFoldingCircuit):
    """Ansatz that preserves protein symmetries."""
    
    def __init__(self, num_qubits: int, num_layers: int = 3, symmetry_group: Optional[List] = None):
        super().__init__(num_qubits, num_layers)
        self.symmetry_group = symmetry_group or []
    
    def get_num_parameters(self) -> int:
        return self.num_qubits * 2 * self.num_layers
    
    def build(self, device: qml.Device) -> Callable:
        """Build symmetry-preserving circuit."""
        
        @qml.qnode(device)
        def circuit(params, hamiltonian):
            self._apply_symmetric_layers(params)
            return qml.expval(hamiltonian)
        
        return circuit
    
    def _apply_symmetric_layers(self, params: np.ndarray):
        """Apply layers respecting symmetry constraints."""
        param_idx = 0
        
        for layer in range(self.num_layers):
            # Symmetric rotation layer
            for qubit in range(self.num_qubits):
                qml.RY(params[param_idx], wires=qubit)
                qml.RZ(params[param_idx + 1], wires=qubit)
                param_idx += 2
            
            # Symmetric entanglement (preserves permutation symmetry)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if self.num_qubits > 2:
                qml.CNOT(wires=[self.num_qubits - 1, 0])


class AdaptiveCircuitBuilder:
    """Builds circuits adaptively based on problem structure."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates = []
        
    def add_rotation_layer(self, axes: List[str] = ['X', 'Y', 'Z']):
        """Add single-qubit rotation layer."""
        for qubit in range(self.num_qubits):
            for axis in axes:
                self.gates.append((f'R{axis}', [qubit]))
    
    def add_entanglement_pattern(self, pattern: str = 'linear'):
        """Add entanglement gates with specified pattern."""
        if pattern == 'linear':
            for i in range(self.num_qubits - 1):
                self.gates.append(('CNOT', [i, i + 1]))
        elif pattern == 'circular':
            for i in range(self.num_qubits - 1):
                self.gates.append(('CNOT', [i, i + 1]))
            self.gates.append(('CNOT', [self.num_qubits - 1, 0]))
        elif pattern == 'all-to-all':
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    self.gates.append(('CNOT', [i, j]))
    
    def build(self, device: qml.Device) -> Callable:
        """Build the adaptive circuit."""
        gates_copy = self.gates.copy()
        
        @qml.qnode(device)
        def circuit(params, hamiltonian):
            param_idx = 0
            
            for gate_name, wires in gates_copy:
                if gate_name.startswith('R'):
                    axis = gate_name[1]
                    if axis == 'X':
                        qml.RX(params[param_idx], wires=wires[0])
                    elif axis == 'Y':
                        qml.RY(params[param_idx], wires=wires[0])
                    elif axis == 'Z':
                        qml.RZ(params[param_idx], wires=wires[0])
                    param_idx += 1
                elif gate_name == 'CNOT':
                    qml.CNOT(wires=wires)
            
            return qml.expval(hamiltonian)
        
        return circuit
    
    def get_num_parameters(self) -> int:
        """Count parameterized gates."""
        return sum(1 for gate_name, _ in self.gates if gate_name.startswith('R'))
