import ast
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer


class CodeProcessor:
    """Process code for the transformer model."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_code_length: int = 512,
        max_error_length: int = 128,
        use_ast: bool = True,
        use_data_flow: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_code_length = max_code_length
        self.max_error_length = max_error_length
        self.use_ast = use_ast
        self.use_data_flow = use_data_flow
    
    def process_code(
        self,
        code: str,
        error_msg: Optional[str] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Process code and error message for the model."""
        # Tokenize code
        code_tokens = self.tokenizer(
            code,
            max_length=self.max_code_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )
        
        # Process error message if provided
        if error_msg is not None:
            error_tokens = self.tokenizer(
                error_msg,
                max_length=self.max_error_length,
                padding="max_length",
                truncation=True,
                return_tensors=return_tensors,
            )
        else:
            error_tokens = {
                "input_ids": torch.zeros((1, self.max_error_length), dtype=torch.long),
                "attention_mask": torch.zeros((1, self.max_error_length), dtype=torch.long),
            }
        
        # Process AST if enabled
        if self.use_ast:
            ast_features = self._process_ast(code)
        else:
            ast_features = None
        
        # Process data flow if enabled
        if self.use_data_flow:
            data_flow_features = self._process_data_flow(code)
        else:
            data_flow_features = None
        
        # Combine all features
        features = {
            "input_ids": code_tokens["input_ids"],
            "attention_mask": code_tokens["attention_mask"],
            "error_input_ids": error_tokens["input_ids"],
            "error_attention_mask": error_tokens["attention_mask"],
        }
        
        if ast_features is not None:
            features.update(ast_features)
        
        if data_flow_features is not None:
            features.update(data_flow_features)
        
        return features
    
    def _process_ast(self, code: str) -> Dict[str, torch.Tensor]:
        """Process AST features."""
        try:
            # Parse code to AST
            tree = ast.parse(code)
            
            # Extract AST features
            node_types = []
            node_values = []
            
            for node in ast.walk(tree):
                node_types.append(type(node).__name__)
                if isinstance(node, ast.Name):
                    node_values.append(node.id)
                elif isinstance(node, ast.Constant):
                    node_values.append(str(node.value))
                else:
                    node_values.append("")
            
            # Tokenize AST features
            node_type_tokens = self.tokenizer(
                " ".join(node_types),
                max_length=self.max_code_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            node_value_tokens = self.tokenizer(
                " ".join(node_values),
                max_length=self.max_code_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            return {
                "ast_type_ids": node_type_tokens["input_ids"],
                "ast_type_mask": node_type_tokens["attention_mask"],
                "ast_value_ids": node_value_tokens["input_ids"],
                "ast_value_mask": node_value_tokens["attention_mask"],
            }
        
        except SyntaxError:
            # Return empty tensors if code is not valid Python
            return {
                "ast_type_ids": torch.zeros((1, self.max_code_length), dtype=torch.long),
                "ast_type_mask": torch.zeros((1, self.max_code_length), dtype=torch.long),
                "ast_value_ids": torch.zeros((1, self.max_code_length), dtype=torch.long),
                "ast_value_mask": torch.zeros((1, self.max_code_length), dtype=torch.long),
            }
    
    def _process_data_flow(self, code: str) -> Dict[str, torch.Tensor]:
        """Process data flow features."""
        # Extract variable definitions and uses
        var_defs = {}
        var_uses = {}
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_defs[target.id] = node.lineno
                elif isinstance(node, ast.Name):
                    if node.id not in var_defs:
                        var_uses[node.id] = node.lineno
        
        except SyntaxError:
            pass
        
        # Create data flow features
        def_lines = list(var_defs.values())
        use_lines = list(var_uses.values())
        
        # Tokenize data flow features
        def_tokens = self.tokenizer(
            " ".join(map(str, def_lines)),
            max_length=self.max_code_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        use_tokens = self.tokenizer(
            " ".join(map(str, use_lines)),
            max_length=self.max_code_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "def_ids": def_tokens["input_ids"],
            "def_mask": def_tokens["attention_mask"],
            "use_ids": use_tokens["input_ids"],
            "use_mask": use_tokens["attention_mask"],
        }
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def batch_process(
        self,
        codes: List[str],
        error_msgs: Optional[List[str]] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Process a batch of code samples."""
        if error_msgs is None:
            error_msgs = [None] * len(codes)
        
        # Process each sample
        processed_samples = [
            self.process_code(code, error_msg, return_tensors)
            for code, error_msg in zip(codes, error_msgs)
        ]
        
        # Combine features
        batch_features = {}
        for key in processed_samples[0].keys():
            batch_features[key] = torch.cat([sample[key] for sample in processed_samples])
        
        return batch_features 