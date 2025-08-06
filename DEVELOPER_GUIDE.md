# Developer Guide - Extending SCLLMManager

This guide explains how to add new single-cell language models and extend existing functionality in the SCLLMManager framework.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Adding a New Model](#adding-a-new-model)
- [Extending Existing Models](#extending-existing-models)
- [Adding New Methods](#adding-new-methods)
- [Base Classes Reference](#base-classes-reference)
- [Testing Guidelines](#testing-guidelines)
- [Contributing](#contributing)

## Architecture Overview

The SCLLMManager framework follows a modular architecture:

```
scllm/
├── base.py                 # Base classes and interfaces
├── model_factory.py        # Factory pattern for model creation
├── {model_name}_model.py   # Individual model implementations
├── utils/                  # Utility functions
│   ├── output_utils.py    # Output formatting and progress
│   └── message_standards.py  # Standard messages
└── tutorials/              # Example notebooks
```

### Key Components

1. **SCLLMBase**: Abstract base class defining the interface all models must implement
2. **ModelFactory**: Factory class for creating and registering models
3. **SCLLMManager**: High-level interface for common operations
4. **TaskConfig**: Configuration management for different tasks

## Adding a New Model

### Step 1: Create Model Implementation

Create a new file `your_model_name_model.py`:

```python
"""
YourModelName implementation for SCLLMManager.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from anndata import AnnData

from .base import SCLLMBase, ModelConfig, TaskConfig
from .utils.output_utils import SCLLMOutput, ModelProgressManager


class YourModelNameModel(SCLLMBase):
    """
    YourModelName implementation following SCLLMBase interface.
    """
    
    def __init__(self, device: Optional[str] = None, **kwargs):
        """Initialize your model with specific parameters."""
        super().__init__(model_name="YourModelName", device=device)
        
        # Model-specific initialization
        self.config = {
            'param1': kwargs.get('param1', 'default_value'),
            'param2': kwargs.get('param2', 42),
            # Add your model-specific parameters
        }
        
        # Initialize model components
        self.model = None
        self.tokenizer = None  # If needed
        self.preprocessor = None  # If needed
        
        SCLLMOutput.status(f"{self.model_name} model initialized", 'loaded')
    
    def load_model(self, model_path: Union[str, Path], **kwargs) -> None:
        """
        Load pre-trained model weights and components.
        
        Args:
            model_path: Path to model files
            **kwargs: Model-specific loading parameters
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        SCLLMOutput.status(f"Loading {self.model_name} model", 'loading')
        
        try:
            # Load your model architecture
            self.model = self._build_model(**kwargs)
            
            # Load pre-trained weights
            if (model_path / "model.pt").exists():
                state_dict = torch.load(model_path / "model.pt", map_location=self.device)
                self.model.load_state_dict(state_dict)
            
            # Load additional components (tokenizer, vocab, etc.)
            if (model_path / "vocab.json").exists():
                self.vocab = self._load_vocabulary(model_path / "vocab.json")
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            SCLLMOutput.status(f"{self.model_name} model loaded successfully", 'loaded')
            
        except Exception as e:
            SCLLMOutput.error(f"Failed to load {self.model_name} model: {e}")
            raise
    
    def preprocess(self, adata: AnnData, **kwargs) -> AnnData:
        """
        Preprocess data for model input.
        
        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters
            
        Returns:
            Preprocessed AnnData object
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before preprocessing")
        
        SCLLMOutput.status("Preprocessing data", 'processing')
        
        # Create a copy to avoid modifying original data
        adata_processed = adata.copy()
        
        # Implement your preprocessing pipeline
        # Example steps:
        
        # 1. Gene filtering/mapping
        if hasattr(self, 'vocab'):
            adata_processed = self._filter_genes_by_vocab(adata_processed)
        
        # 2. Normalization (if required)
        if kwargs.get('normalize', True):
            adata_processed = self._normalize_data(adata_processed, **kwargs)
        
        # 3. Model-specific preprocessing
        adata_processed = self._model_specific_preprocessing(adata_processed, **kwargs)
        
        SCLLMOutput.status("Preprocessing completed", 'complete')
        return adata_processed
    
    def predict(self, adata: AnnData, task: str = "annotation", **kwargs) -> Dict[str, Any]:
        """
        Make predictions using the model.
        
        Args:
            adata: Input AnnData object
            task: Task type ('annotation', 'integration', 'generation', etc.)
            **kwargs: Task-specific parameters
            
        Returns:
            Dictionary containing predictions and metadata
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before prediction")
        
        # Get task-specific configuration
        if task in ['annotation', 'integration', 'generation']:
            task_config = TaskConfig.get_task_config(task)
            # Merge with provided kwargs
            config = {**task_config, **kwargs}
        else:
            config = kwargs
        
        # Preprocess data
        adata_processed = self.preprocess(adata, **config)
        
        # Route to appropriate prediction method
        if task == "annotation":
            return self._predict_annotation(adata_processed, **config)
        elif task == "integration":
            return self._predict_integration(adata_processed, **config)
        elif task == "generation":
            return self._predict_generation(adata_processed, **config)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def fine_tune(self, train_adata: AnnData, valid_adata: Optional[AnnData] = None, **kwargs) -> Dict[str, Any]:
        """
        Fine-tune the model on new data.
        
        Args:
            train_adata: Training data
            valid_adata: Validation data (optional)
            **kwargs: Training parameters
            
        Returns:
            Training results and metrics
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before fine-tuning")
        
        SCLLMOutput.status(f"Starting {self.model_name} fine-tuning", 'training')
        
        # Set up training configuration
        config = {
            'epochs': kwargs.get('epochs', 10),
            'batch_size': kwargs.get('batch_size', 32),
            'lr': kwargs.get('lr', 1e-4),
            'task': kwargs.get('task', 'annotation'),
            **kwargs
        }
        
        # Preprocess training data
        train_processed = self.preprocess(train_adata, **config)
        valid_processed = self.preprocess(valid_adata, **config) if valid_adata is not None else None
        
        # Create data loaders
        train_loader = self._create_dataloader(train_processed, config['batch_size'], shuffle=True)
        valid_loader = self._create_dataloader(valid_processed, config['batch_size'], shuffle=False) if valid_processed else None
        
        # Set up optimizer and loss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'])
        loss_fn = self._get_loss_function(config['task'])
        
        # Training loop
        results = self._training_loop(
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=config['epochs'],
            **config
        )
        
        SCLLMOutput.status("Fine-tuning completed", 'complete')
        return results
    
    def get_embeddings(self, adata: AnnData, **kwargs) -> np.ndarray:
        """
        Extract cell embeddings from the model.
        
        Args:
            adata: Input AnnData object
            **kwargs: Embedding extraction parameters
            
        Returns:
            Cell embeddings as numpy array
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before embedding extraction")
        
        SCLLMOutput.status("Extracting embeddings", 'processing')
        
        # Preprocess data
        adata_processed = self.preprocess(adata, **kwargs)
        
        # Create data loader
        batch_size = kwargs.get('batch_size', 32)
        dataloader = self._create_dataloader(adata_processed, batch_size, shuffle=False)
        
        # Extract embeddings
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                batch_embeddings = self.model.encode(batch)  # Your model's encoding method
                embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        
        SCLLMOutput.status(f"Extracted embeddings: {embeddings.shape}", 'complete')
        return embeddings
    
    def _save_model_specific(self, save_path: Path, **kwargs) -> None:
        """Save model-specific components."""
        # Save model weights
        torch.save(self.model.state_dict(), save_path / "model.pt")
        
        # Save vocabulary/tokenizer if exists
        if hasattr(self, 'vocab'):
            import json
            with open(save_path / "vocab.json", 'w') as f:
                json.dump(self.vocab, f)
        
        # Save configuration
        import json
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    # Helper methods (implement based on your model's needs)
    
    def _build_model(self, **kwargs):
        """Build model architecture."""
        # Implement your model architecture
        pass
    
    def _load_vocabulary(self, vocab_path):
        """Load vocabulary/tokenizer."""
        # Implement vocabulary loading
        pass
    
    def _filter_genes_by_vocab(self, adata):
        """Filter genes to match model vocabulary."""
        # Implement gene filtering
        pass
    
    def _normalize_data(self, adata, **kwargs):
        """Apply normalization to data."""
        # Implement normalization
        pass
    
    def _model_specific_preprocessing(self, adata, **kwargs):
        """Apply model-specific preprocessing steps."""
        # Implement model-specific preprocessing
        pass
    
    def _predict_annotation(self, adata, **kwargs):
        """Predict cell types."""
        # Implement cell type prediction
        pass
    
    def _predict_integration(self, adata, **kwargs):
        """Perform batch integration."""
        # Implement batch integration
        pass
    
    def _predict_generation(self, adata, **kwargs):
        """Generate new cells."""
        # Implement cell generation
        pass
    
    def _create_dataloader(self, adata, batch_size, shuffle=True):
        """Create PyTorch DataLoader."""
        # Implement DataLoader creation
        pass
    
    def _get_loss_function(self, task):
        """Get appropriate loss function for task."""
        # Return appropriate loss function
        pass
    
    def _training_loop(self, train_loader, valid_loader, optimizer, loss_fn, epochs, **config):
        """Execute training loop."""
        # Implement training loop
        pass
```

### Step 2: Register the Model

Add your model to `model_factory.py`:

```python
# In model_factory.py

# Import your model
from .your_model_name_model import YourModelNameModel

class ModelFactory:
    # Add to the _models registry
    _models = {
        "scgpt": ScGPTModel,
        "scfoundation": ScFoundationModel,
        "geneformer": GeneformerModel,
        "cellplm": CellPLMModel,
        "uce": UCEModel,
        "your_model_name": YourModelNameModel,  # Add this line
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs):
        """Create model with dependency checks."""
        
        # Add dependency checks for your model
        elif model_type.lower() == "your_model_name":
            try:
                import your_required_library
            except ImportError:
                raise ImportError("your_required_library is required. Install with `pip install your_required_library`")
        
        # Rest of the method remains the same
```

### Step 3: Add Convenience Functions

Add convenient wrapper functions to `model_factory.py`:

```python
# Convenience functions for your model
def load_your_model_name(model_path: Union[str, Path], device: Optional[str] = None, **kwargs):
    """Quick function to load YourModelName."""
    return ModelFactory.create_model("your_model_name", model_path, device, **kwargs)

def get_embeddings_with_your_model_name(adata, model_path: Union[str, Path], **kwargs):
    """Quick function to get embeddings with YourModelName."""
    model = load_your_model_name(model_path, **kwargs)
    return model.get_embeddings(adata, **kwargs)

def end_to_end_your_model_name_annotation(reference_adata, query_adata, model_path, **kwargs):
    """Complete workflow for YourModelName annotation."""
    # Implement end-to-end workflow
    pass
```

## Extending Existing Models

### Adding New Methods to Existing Models

To add new functionality to existing models:

```python
# In the existing model file (e.g., scgpt_model.py)

class ScGPTModel(SCLLMBase):
    # ... existing methods ...
    
    def your_new_method(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """
        New method for specific functionality.
        
        Args:
            adata: Input data
            **kwargs: Method-specific parameters
            
        Returns:
            Results dictionary
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded")
        
        # Implement new functionality
        SCLLMOutput.status("Running new method", 'processing')
        
        # Your implementation here
        results = {}
        
        SCLLMOutput.status("New method completed", 'complete')
        return results
```

### Adding New Task Types

To support new task types, extend the `TaskConfig` class:

```python
# In base.py

class TaskConfig:
    # Add new task configuration
    YOUR_NEW_TASK_CONFIG = {
        'param1': 'value1',
        'param2': 42,
        # Add task-specific parameters
    }
    
    @classmethod
    def get_task_config(cls, task: str) -> Dict[str, Any]:
        """Get configuration for a specific task."""
        task_configs = {
            'annotation': cls.ANNOTATION_CONFIG,
            'integration': cls.INTEGRATION_CONFIG,
            'generation': cls.GENERATION_CONFIG,
            'your_new_task': cls.YOUR_NEW_TASK_CONFIG,  # Add this line
        }
        
        if task not in task_configs:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(task_configs.keys())}")
        
        return task_configs[task].copy()
```

