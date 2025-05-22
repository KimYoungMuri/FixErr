from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

class CodeRepairTransformer(nn.Module):
    """
    A transformer-based model for code repair that combines encoder-decoder architecture
    with specialized components for code understanding and repair.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        
        self.config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            "dropout": dropout,
            "layer_norm_eps": layer_norm_eps,
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
        }
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Transformer layers
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(num_hidden_layers)
        ])
        
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(num_hidden_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Seq2SeqLMOutput:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask for input of shape (batch_size, seq_length)
            labels: Target token IDs of shape (batch_size, seq_length)
            decoder_attention_mask: Attention mask for decoder input of shape (batch_size, seq_length)
            
        Returns:
            Seq2SeqLMOutput containing loss and logits
        """
        # Ensure input tensors have the correct shape and dtype
        if len(input_ids.shape) != 2:
            raise ValueError(f"Expected input_ids to have shape (batch_size, seq_length), got {input_ids.shape}")
        
        batch_size, seq_length = input_ids.shape
        
        # Ensure sequence length doesn't exceed max_position_embeddings
        if seq_length > self.config["max_position_embeddings"]:
            raise ValueError(
                f"Input sequence length {seq_length} exceeds maximum position embeddings "
                f"{self.config['max_position_embeddings']}"
            )
        
        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        inputs_embeds = self.token_embeddings(input_ids)  # Shape: (batch_size, seq_length, hidden_size)
        position_embeddings = self.position_embeddings(position_ids)  # Shape: (batch_size, seq_length, hidden_size)
        
        # Combine embeddings
        hidden_states = inputs_embeds + position_embeddings  # Shape: (batch_size, seq_length, hidden_size)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Ensure attention mask has the right shape
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)  # Shape: (batch_size, seq_length)
        
        # Encoder layers
        encoder_outputs = []
        for encoder_layer in self.encoder:
            hidden_states = encoder_layer(hidden_states, attention_mask)
            encoder_outputs.append(hidden_states)
        
        # Decoder layers
        if labels is not None:
            decoder_input_ids = self._shift_right(labels)
        else:
            decoder_input_ids = input_ids  # Use input_ids as decoder input if no labels provided
        
        # Get decoder embeddings
        decoder_hidden_states = self.token_embeddings(decoder_input_ids)  # Shape: (batch_size, seq_length, hidden_size)
        decoder_position_ids = torch.arange(
            decoder_input_ids.size(1), dtype=torch.long, device=decoder_input_ids.device
        )
        decoder_position_ids = decoder_position_ids.unsqueeze(0).expand(batch_size, -1)
        decoder_position_embeddings = self.position_embeddings(decoder_position_ids)  # Shape: (batch_size, seq_length, hidden_size)
        
        # Combine decoder embeddings
        decoder_hidden_states = decoder_hidden_states + decoder_position_embeddings  # Shape: (batch_size, seq_length, hidden_size)
        decoder_hidden_states = self.layer_norm(decoder_hidden_states)
        decoder_hidden_states = self.dropout(decoder_hidden_states)
        
        # Ensure decoder attention mask has the right shape
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.view(batch_size, -1)  # Shape: (batch_size, seq_length)
        
        # Process through decoder layers
        for decoder_layer in self.decoder:
            decoder_hidden_states = decoder_layer(
                decoder_hidden_states,
                encoder_outputs[-1],
                decoder_attention_mask,
                attention_mask
            )
        
        # Output layer
        logits = self.output_layer(decoder_hidden_states)  # Shape: (batch_size, seq_length, vocab_size)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config["vocab_size"]), labels.view(-1))
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=encoder_outputs[-1],
            decoder_hidden_states=decoder_hidden_states,
        )
    
    def _shift_right(self, input_ids):
        """Shift input ids one token to the right."""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = self.config["bos_token_id"]
        return shifted_input_ids


class TransformerEncoderLayer(nn.Module):
    """A single layer of the transformer encoder."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )
        
        self.intermediate = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the encoder layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_length)
        """
        # Ensure hidden_states has the correct shape
        if len(hidden_states.shape) != 3:
            raise ValueError(
                f"Expected hidden_states to have shape (batch_size, seq_length, hidden_size), "
                f"got {hidden_states.shape}"
            )
        
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)
        attention_output = self.dropout(attention_output)
        hidden_states = self.layer_norm1(hidden_states + attention_output)
        
        # Feed-forward
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.dropout(intermediate_output)
        hidden_states = self.layer_norm2(hidden_states + intermediate_output)
        
        return hidden_states


class TransformerDecoderLayer(nn.Module):
    """A single layer of the transformer decoder."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )
        
        self.cross_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )
        
        self.intermediate = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer_norm3 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the decoder layer."""
        # Self-attention
        self_attention_output = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=decoder_attention_mask
        )
        self_attention_output = self.dropout(self_attention_output)
        hidden_states = self.layer_norm1(hidden_states + self_attention_output)
        
        # Cross-attention
        cross_attention_output = self.cross_attention(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask
        )
        cross_attention_output = self.dropout(cross_attention_output)
        hidden_states = self.layer_norm2(hidden_states + cross_attention_output)
        
        # Feed-forward
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.dropout(intermediate_output)
        hidden_states = self.layer_norm3(hidden_states + intermediate_output)
        
        return hidden_states


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float,
    ):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Ensure hidden_size is divisible by num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size} is not a multiple of the number of attention "
                f"heads {num_attention_heads}"
            )
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.output = nn.Linear(self.all_head_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose the input tensor for attention scores."""
        batch_size, seq_length, _ = x.size()
        new_x_shape = (batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the multi-head attention.
        
        Args:
            hidden_states: Query states of shape (batch_size, seq_length, hidden_size)
            key_value_states: Key/Value states of shape (batch_size, seq_length, hidden_size)
                If None, uses hidden_states for key/value (self-attention)
            attention_mask: Attention mask of shape (batch_size, seq_length)
        """
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Ensure all tensors are float
        hidden_states = hidden_states.float()
        
        # If key_value_states is None, use hidden_states (self-attention)
        if key_value_states is None:
            key_value_states = hidden_states
        else:
            key_value_states = key_value_states.float()
            # Ensure key_value_states has the same hidden_size as hidden_states
            if key_value_states.shape[-1] != hidden_size:
                raise ValueError(
                    f"key_value_states hidden size {key_value_states.shape[-1]} "
                    f"does not match hidden_states hidden size {hidden_size}"
                )
        
        # Project queries, keys, and values
        query_layer = self.transpose_for_scores(self.query(hidden_states))  # Shape: (batch_size, num_heads, seq_length, head_size)
        key_layer = self.transpose_for_scores(self.key(key_value_states))   # Shape: (batch_size, num_heads, seq_length, head_size)
        value_layer = self.transpose_for_scores(self.value(key_value_states))  # Shape: (batch_size, num_heads, seq_length, head_size)
        
        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # Shape: (batch_size, num_heads, seq_length, seq_length)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float))
        
        if attention_mask is not None:
            # Convert attention mask to float and reshape
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_length)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)  # Shape: (batch_size, num_heads, seq_length, seq_length)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)  # Shape: (batch_size, num_heads, seq_length, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, seq_length, num_heads, head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # Shape: (batch_size, seq_length, hidden_size)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Project back to hidden size
        output = self.output(context_layer)
        
        return output 