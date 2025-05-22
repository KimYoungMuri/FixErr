from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerConfig:
    """Configuration for the transformer model."""
    
    # Model architecture
    vocab_size: int = 50000
    hidden_size: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Token IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    
    # Training
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    
    # Generation
    max_length: int = 512
    num_beams: int = 4
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    
    # Pre-training
    mlm_probability: float = 0.15
    span_masking: bool = True
    max_span_length: int = 5
    
    # Code-specific
    max_code_length: int = 512
    max_error_length: int = 128
    use_ast: bool = True
    use_data_flow: bool = True
    
    def __post_init__(self):
        """Validate the configuration."""
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size {self.hidden_size} must be divisible by "
            f"num_attention_heads {self.num_attention_heads}"
        )
        
        assert self.intermediate_size >= self.hidden_size, (
            f"intermediate_size {self.intermediate_size} must be at least "
            f"hidden_size {self.hidden_size}"
        )
        
        assert 0 <= self.dropout <= 1, f"dropout {self.dropout} must be between 0 and 1"
        assert 0 <= self.layer_norm_eps <= 1, f"layer_norm_eps {self.layer_norm_eps} must be between 0 and 1"
        
        assert self.max_position_embeddings > 0, f"max_position_embeddings {self.max_position_embeddings} must be positive"
        assert self.max_length > 0, f"max_length {self.max_length} must be positive"
        assert self.num_beams > 0, f"num_beams {self.num_beams} must be positive"
        assert self.temperature > 0, f"temperature {self.temperature} must be positive"
        assert self.top_k > 0, f"top_k {self.top_k} must be positive"
        assert 0 <= self.top_p <= 1, f"top_p {self.top_p} must be between 0 and 1"
        assert self.repetition_penalty >= 1.0, f"repetition_penalty {self.repetition_penalty} must be at least 1.0"
        
        assert 0 <= self.mlm_probability <= 1, f"mlm_probability {self.mlm_probability} must be between 0 and 1"
        assert self.max_span_length > 0, f"max_span_length {self.max_span_length} must be positive"
        
        assert self.max_code_length > 0, f"max_code_length {self.max_code_length} must be positive"
        assert self.max_error_length > 0, f"max_error_length {self.max_error_length} must be positive"
        assert self.num_epochs > 0, f"num_epochs {self.num_epochs} must be positive" 