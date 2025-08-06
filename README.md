# Pantheon_llm - Single-Cell Language Model Analysis Framework

A unified framework for single-cell RNA-seq analysis using large language models including scGPT, scFoundation, Geneformer, CellPLM, and UCE.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Core Usage](#core-usage)

## Installation

```bash
pip install torch scanpy anndata
# Install model-specific dependencies as needed
```

## Quick Start

```python
import scanpy as sc
from model_factory import SCLLMManager

# Load your data
adata = sc.read_h5ad('your_data.h5ad')

# Initialize manager with a model
manager = SCLLMManager(
    model_type="scgpt",  # or "scfoundation", "geneformer", "cellplm", "uce"
    model_path="path/to/model",
    device="cuda"  # or "cpu", "auto"
)

# Get cell embeddings
embeddings = manager.get_embeddings(adata)
adata.obsm['X_scgpt'] = embeddings

```

## Supported Models

| Model | Purpose | Key Features |
|-------|---------|-------------|
| **scGPT** | General-purpose | Cell annotation, integration, generation |
| **scFoundation** | Foundation model | Large-scale pre-training, versatile |
| **Geneformer** | Gene perturbation | In silico perturbation experiments |
| **CellPLM** | Protein language model | Multi-omics integration |
| **UCE** | Universal embeddings | Cross-species, cross-platform |

## Core Usage

You can find the detailed tutorial in the folder `tutorials`

## Development

You can also read our [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) to know the model architecture

## License

This project is licensed under the MIT License.