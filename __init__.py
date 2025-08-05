"""
scLLM: Single-cell Language Models
==================================

A unified interface for single-cell language models including scGPT, scFoundation and others.

Main classes:
- SCLLMManager: High-level interface for model operations
- ModelFactory: Factory for creating different model types
- ScGPTModel: scGPT model implementation
- ScFoundationModel: scFoundation model implementation
- CellPLMModel: CellPLM model implementation

Quick start with scGPT:
```python
import omicverse as ov

# Load and use scGPT for cell annotation
manager = ov.external.scllm.SCLLMManager(
    model_type="scgpt",
    model_path="/path/to/scgpt/model"
)

# Annotate cells
results = manager.annotate_cells(adata)

# Get embeddings
embeddings = manager.get_embeddings(adata)
```

Quick start with scFoundation:
```python
import omicverse as ov

# Load and use scFoundation for embedding extraction
manager = ov.external.scllm.SCLLMManager(
    model_type="scfoundation",
    model_path="/path/to/scfoundation/model.ckpt"
)

# Get cell embeddings
embeddings = manager.get_embeddings(adata)
```

Quick start with CellPLM:
```python
import omicverse as ov

# Load and use CellPLM for cell embedding
manager = ov.external.scllm.SCLLMManager(
    model_type="cellplm",
    model_path="/path/to/cellplm/checkpoint",
    pretrain_version="20231027_85M"
)

# Get embeddings
embeddings = manager.get_embeddings(adata)

# Fine-tune for annotation
results = manager.fine_tune(train_adata, task="annotation")

# Predict cell types
predictions = manager.annotate_cells(adata)

# Integrate batches
integration_results = manager.integrate(adata, batch_key="batch")
```
"""
