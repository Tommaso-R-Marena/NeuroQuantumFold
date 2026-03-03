"""Neurosymbolic rule learning for extracting interpretable constraints."""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class SymbolicRule:
    """Represents a learned symbolic rule about protein structure."""
    rule_type: str  # 'hydrogen_bond', 'hydrophobic_core', 'secondary_structure'
    residue_indices: List[int]
    confidence: float
    parameters: Dict[str, float]
    logic_form: str  # Prolog-like representation


class AttentionRuleExtractor(nn.Module):
    """Extracts symbolic rules from neural attention patterns."""
    
    def __init__(self, d_model: int = 512, num_rule_types: int = 10):
        super().__init__()
        self.d_model = d_model
        self.num_rule_types = num_rule_types
        
        # Rule-specific attention heads
        self.rule_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_rule_types,
            batch_first=True
        )
        
        # Rule type classifier
        self.rule_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_rule_types)
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract rules from neural embeddings.
        
        Args:
            embeddings: Neural embeddings [batch, seq_len, d_model]
            
        Returns:
            rule_types: Predicted rule types [batch, seq_len, num_rule_types]
            attention_weights: Rule attention patterns [batch, seq_len, seq_len]
            confidences: Rule confidence scores [batch, seq_len, 1]
        """
        # Apply rule-specific attention
        attended, attention_weights = self.rule_attention(
            embeddings, embeddings, embeddings
        )
        
        # Classify rule types
        rule_types = self.rule_classifier(attended)
        
        # Estimate confidence
        confidences = self.confidence_head(attended)
        
        return rule_types, attention_weights, confidences


class NeuralSymbolicRuleLearner(nn.Module):
    """Main neurosymbolic module for learning interpretable structural rules."""
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        
        # Rule extraction components
        self.rule_extractor = AttentionRuleExtractor(d_model)
        
        # Domain knowledge encoders
        self.hbond_detector = HydrogenBondDetector(d_model)
        self.hydrophobic_analyzer = HydrophobicCoreAnalyzer(d_model)
        self.secondary_structure_predictor = SecondaryStructurePredictor(d_model)
        
        # Logic program generator
        self.logic_generator = LogicProgramGenerator()
        
    def forward(
        self,
        embeddings: torch.Tensor,
        sequence: str
    ) -> Tuple[List[SymbolicRule], str]:
        """Learn symbolic rules from neural embeddings.
        
        Args:
            embeddings: Neural sequence embeddings
            sequence: Original amino acid sequence
            
        Returns:
            rules: List of extracted symbolic rules
            logic_program: Prolog-like logic program
        """
        # Extract rule candidates
        rule_types, attention_weights, confidences = self.rule_extractor(embeddings)
        
        # Apply domain-specific detectors
        hbond_rules = self.hbond_detector(embeddings, attention_weights, sequence)
        hydrophobic_rules = self.hydrophobic_analyzer(embeddings, sequence)
        ss_rules = self.secondary_structure_predictor(embeddings)
        
        # Combine all rules
        all_rules = hbond_rules + hydrophobic_rules + ss_rules
        
        # Filter by confidence threshold
        filtered_rules = [r for r in all_rules if r.confidence > 0.7]
        
        # Generate logic program
        logic_program = self.logic_generator.generate(filtered_rules, sequence)
        
        return filtered_rules, logic_program


class HydrogenBondDetector(nn.Module):
    """Detects potential hydrogen bonds from neural representations."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.distance_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_weights: torch.Tensor,
        sequence: str
    ) -> List[SymbolicRule]:
        """Detect hydrogen bond candidates."""
        rules = []
        seq_len = embeddings.shape[1]
        
        # Identify donor-acceptor pairs
        donors = self._find_donors(sequence)
        acceptors = self._find_acceptors(sequence)
        
        for i in donors:
            for j in acceptors:
                if abs(i - j) > 3:  # Non-local interactions
                    # Predict interaction strength
                    pair_embedding = torch.cat([
                        embeddings[0, i],
                        embeddings[0, j]
                    ])
                    strength = self.distance_predictor(pair_embedding).item()
                    
                    if strength > 0.5:
                        rule = SymbolicRule(
                            rule_type='hydrogen_bond',
                            residue_indices=[i, j],
                            confidence=strength,
                            parameters={'distance': 3.0, 'angle': 180.0},
                            logic_form=f"hbond({i}, {j}, {strength:.3f})."
                        )
                        rules.append(rule)
        
        return rules
    
    @staticmethod
    def _find_donors(sequence: str) -> List[int]:
        """Find potential H-bond donors (Ser, Thr, Tyr, Asn, Gln, His, Trp)."""
        donors = 'STYQNHW'
        return [i for i, aa in enumerate(sequence) if aa in donors]
    
    @staticmethod
    def _find_acceptors(sequence: str) -> List[int]:
        """Find potential H-bond acceptors (Asp, Glu, Asn, Gln, Ser, Thr, Tyr)."""
        acceptors = 'DENQSTY'
        return [i for i, aa in enumerate(sequence) if aa in acceptors]


class HydrophobicCoreAnalyzer(nn.Module):
    """Analyzes hydrophobic core formation patterns."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.core_predictor = nn.Linear(d_model, 1)
        
    def forward(self, embeddings: torch.Tensor, sequence: str) -> List[SymbolicRule]:
        """Identify hydrophobic core clusters."""
        rules = []
        hydrophobic = 'AILMFVPW'
        hydrophobic_indices = [i for i, aa in enumerate(sequence) if aa in hydrophobic]
        
        if len(hydrophobic_indices) > 3:
            # Predict core formation likelihood
            core_scores = torch.sigmoid(self.core_predictor(embeddings[0, hydrophobic_indices]))
            
            # Cluster high-scoring residues
            high_score_indices = [idx for idx, score in zip(hydrophobic_indices, core_scores) if score > 0.6]
            
            if len(high_score_indices) >= 3:
                avg_confidence = core_scores[core_scores > 0.6].mean().item()
                rule = SymbolicRule(
                    rule_type='hydrophobic_core',
                    residue_indices=high_score_indices,
                    confidence=avg_confidence,
                    parameters={'cluster_size': len(high_score_indices)},
                    logic_form=f"hydrophobic_core([{', '.join(map(str, high_score_indices))}])."
                )
                rules.append(rule)
        
        return rules


class SecondaryStructurePredictor(nn.Module):
    """Predicts secondary structure elements (alpha-helix, beta-sheet)."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.ss_classifier = nn.Linear(d_model, 3)  # helix, sheet, coil
        
    def forward(self, embeddings: torch.Tensor) -> List[SymbolicRule]:
        """Predict secondary structure rules."""
        rules = []
        ss_logits = self.ss_classifier(embeddings[0])
        ss_probs = torch.softmax(ss_logits, dim=-1)
        ss_pred = torch.argmax(ss_probs, dim=-1)
        
        ss_types = ['helix', 'sheet', 'coil']
        
        # Find contiguous segments
        current_type = ss_pred[0].item()
        start_idx = 0
        
        for i in range(1, len(ss_pred)):
            if ss_pred[i] != current_type:
                # End of segment
                if i - start_idx >= 3 and current_type < 2:  # Min length 3, exclude coil
                    avg_confidence = ss_probs[start_idx:i, current_type].mean().item()
                    rule = SymbolicRule(
                        rule_type='secondary_structure',
                        residue_indices=list(range(start_idx, i)),
                        confidence=avg_confidence,
                        parameters={'ss_type': ss_types[current_type], 'length': i - start_idx},
                        logic_form=f"{ss_types[current_type]}({start_idx}, {i-1})."
                    )
                    rules.append(rule)
                
                start_idx = i
                current_type = ss_pred[i].item()
        
        return rules


class LogicProgramGenerator:
    """Generates Prolog-like logic programs from symbolic rules."""
    
    def generate(self, rules: List[SymbolicRule], sequence: str) -> str:
        """Convert symbolic rules to logic program."""
        program_lines = [
            f"% Protein sequence: {sequence}",
            f"% Total residues: {len(sequence)}",
            "",
            "% Facts"
        ]
        
        # Add sequence facts
        for i, aa in enumerate(sequence):
            program_lines.append(f"residue({i}, '{aa}').")
        
        program_lines.append("\n% Structural constraints")
        
        # Add rules grouped by type
        rule_groups = {}
        for rule in rules:
            if rule.rule_type not in rule_groups:
                rule_groups[rule.rule_type] = []
            rule_groups[rule.rule_type].append(rule)
        
        for rule_type, rule_list in rule_groups.items():
            program_lines.append(f"\n% {rule_type.replace('_', ' ').title()} rules")
            for rule in rule_list:
                program_lines.append(rule.logic_form)
        
        # Add inference rules
        program_lines.extend([
            "\n% Inference rules",
            "close_contact(I, J) :- hbond(I, J, _).",
            "close_contact(I, J) :- abs(I - J) < 5.",
            "buried(I) :- hydrophobic_core(Cluster), member(I, Cluster).",
            "structured(I) :- helix(Start, End), I >= Start, I =< End.",
            "structured(I) :- sheet(Start, End), I >= Start, I =< End."
        ])
        
        return "\n".join(program_lines)
