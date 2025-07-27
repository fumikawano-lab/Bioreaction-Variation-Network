# Bioreaction-Variation Network

**Bioreaction-Variation Network** is a graph neural network (GNN)-based framework designed to infer individual-specific hidden molecular and physiological pathways that may explain interindividual variation in biological responses to physiological stimuli.

This repository provides the full implementation of the architecture, from data embedding and model training to evolutionary inference and path contribution analysis.

---

## Applications

- Mechanism inference of biological response diversity
- Visualization of individualized molecular pathways
- Research, clinical, and commercial uses (licensed under CC BY 4.0)

---

## Directory Structure
Note: The following is the recommended directory structure for using the provided code and pre-trained models in this repository.
To ensure compatibility with the scripts, please organize your files accordingly.
```
Main directory/
├── data/
│   ├── clean_data/
│   ├── evolution_output/
│   ├── explor_data/
│   ├── gnn_data/
│   ├── input_data/
│   ├── pt_data/
│   └── temp_json/
├── embedding_for_input_data.py
├── embedding_for_training_data.py
├── learning.py
├── network_inference.py
└── temp_json_to_pt_conversion.py
```

---

## Pipeline Overview

### 1. Embedding of Training Data
`embedding_for_training_data.py`  
→ Creates `.json` files in `temp_json/`.

### 2. Embedding of Input Data
`embedding_for_input_data.py`  
→ Creates identical `.json` files in `input_data/input_temp_json/`.

### 3. JSON to PT Conversion
`temp_json_to_pt_conversion.py`  
→ Converts JSON files to `.pt` files under `pt_data/`.

### 4. Model Training
`learning.py`  
→ Trains the GNN and saves model files and attention visualizations in `gnn_data/`.

### 5. Network Inference
`network_inference.py`  
→ Performs path inference and evolutionary optimization. Outputs are saved in `explor_data/` and `evolution_output/`.

---

## Requirements

- Python ≥ 3.8
- Recommended packages:
  - `torch`
  - `torch_geometric`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `pandas`
  - `networkx`
  - `deap`

> Use `pip install -r requirements.txt` if the list is provided.

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.  
You are free to use, adapt, and distribute this work, with appropriate credit.

---

## Pre-trained Model
A pre-trained GNN model developed from previously constructed datasets is provided under the following folder in this repository:
/skeletal_muscle/
These folders contain the complete set of files required to perform the domain-specific inference.

### Included Files
- /gnn_model/
  - gnn_model_final.pt – Final trained GNN model

- /pt_data/
  - edge_attr.pt
  - edge_index.pt
  - edge_weight_with_nodes.pt
  - model_features.pt
  - model_features_with_nodes.pt
  - target_edge_attr.pt
  - target_edge_dict.pt
  - target_edge_index.pt
  - target_features.pt

    *These .pt files contain the embedding and structural information necessary for executing inference using the pre-trained model.

- /metadata/
  - model_id_frequency.csv – Frequency of all model nodes
  - target_id_frequency.csv – Frequency of all target nodes

- /reverse_dict/
  - reverse_model_dict.json – Maps model node indices to their original names
  - reverse_target_dict.json – Maps target node indices to their original names

    *These dictionaries are essential for interpreting inference results.

### Current version
| Domain          | Version  | Upload Date | Description                                                                                                           | Reference    |
|-----------------|----------|-------------|-----------------------------------------------------------------------------------------------------------------------|--------------|
| Skeletal muscle | sm_v1.0  | July 2025   | Pre-trained model based on 65,096 literature-derived biological response datasets focused on skeletal muscle          | Unpublished  |

*If you use this pre-trained model in your research, please cite the associated article (once published) or refer to this repository.

---

## Citation

If you use this code in your research, please cite the following article:

Under preparation...

Once published, the DOI and full citation will be provided.

---

## Contact

For questions or collaborations, please contact:  
Fumi Kawano
kawano@t.matsu.ac.jp
