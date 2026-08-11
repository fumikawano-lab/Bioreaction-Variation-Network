import os
import json
import torch
import gc
import numpy as np
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"LOG: Using device: {device}")

DATA_DIR = "./data"

try:
    model_name = "."
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    print(f"LOG: BERT Model ('{model_name}') and Tokenizer loaded successfully!")
except Exception as e:
    print(f"ERROR: Failed to load BERT model ('{model_name}'). {e}")
    exit(1)

def get_embedding(text):
    if isinstance(text, list):
        text = " ".join(map(str, text))
        
    if not text or text.strip().lower() == "none":
        return torch.zeros(768, device=device)

    try:
        tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokens)
        return outputs.last_hidden_state[:, 0, :].squeeze().to(device)
    except Exception as e:
        print(f"ERROR: Failed to generate embedding for text: {text[:50]}... {e}")
        return torch.zeros(768, device=device)
        
input_file = os.path.join(DATA_DIR, "./input_data/YOUR_FILE_NAME_HERE.json")

if not os.path.exists(input_file):
    print(f"ERROR: Dataset file not found at {input_file}")
    exit(1)

with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(f"LOG: Loaded {len(dataset)} records from dataset")

output_dir = "./data/input_data/input_temp_json"
os.makedirs(output_dir, exist_ok=True)

model_node = []
model_index = []
model_features = {}
model_features_with_nodes = {}
target_node = []
target_index = []
edge_index = []
total_edge_index = []
edge_attr = []
target_features = {}
edge_weight_with_nodes = {}
target_edges = []
target_edges_with_weights = {}
model_dict = {}
reverse_model_dict = {}
target_dict = {}
reverse_target_dict = {}
edges_dict = {}
count_model = {}
count_target = {}
target_edge_dict = {}

input_edge_index = []
input_edge_attr = []
input_target_edge_index = []
input_target_edge_attr = []

for index, data in enumerate(dataset):
    paper_id = data.get("id", f"paper_{index}")
    if (index + 1) % 100 == 0:
        print(f"LOG: Processing record {index + 1}/{len(dataset)}")
        print(f"LOG: Current model_id count: {len(model_dict)}")
        print(f"LOG: Current target_id count: {len(target_dict)}")
        print(f"LOG: Current edge count: {len(edges_dict)}")
        print(f"LOG: Current target_edge count: {len(target_edges)}")

        top_models = sorted(count_model.items(), key=lambda x: x[1], reverse=True)[:5]
        print("LOG: Top 5 most frequent model_ids:")
        for model_id, count in top_models:
            model_main_text = reverse_model_dict.get(model_id, "Unknown")
            print(f"  - {model_id} ({model_main_text}): {count} occurrences")

        top_targets = sorted(count_target.items(), key=lambda x: x[1], reverse=True)[:5]
        print("LOG: Top 5 most frequent target_ids:")
        for target_id, count in top_targets:
            target_text = reverse_target_dict.get(target_id, "Unknown")
            print(f"  - {target_id} ({target_text}): {count} occurrences")

        torch.cuda.empty_cache()
        gc.collect()
 
    model_main_text = data.get("model_main", "").strip().replace("none", "")
    if not model_main_text:
        print(f"WARNING: Skipping entry with empty model_main")
        continue

    model_dict_file = os.path.join(DATA_DIR, "./temp_json/model_dict.json")
    if not os.path.exists(model_dict_file):
        print(f"ERROR: model_dict.json not found at {model_dict_file}")
        exit(1)

    with open(model_dict_file, "r", encoding="utf-8") as f:
        model_dict = json.load(f)

    if model_main_text in model_dict:
        model_id = model_dict[model_main_text]
        count_model[model_id] = count_model.get(model_id, 0) + 1
    else:
        print(f"WARNING: model_main_text not found in model_dict -> {model_main_text}")
        continue

    model_main_embedding = get_embedding(model_main_text)

    def rodrigues_rotation(vector, angle_deg, axis):
        device = vector.device
        angle_rad = torch.tensor(np.radians(angle_deg), dtype=torch.float32, device=device)
    
        axis = axis.to(device) / torch.norm(axis.to(device))
        identity = torch.eye(vector.shape[0], device=device)
        axis_outer = torch.ger(axis, axis).to(device)

        cos_theta = torch.cos(angle_rad)
        sin_theta = torch.sin(angle_rad)

        rotation_matrix = cos_theta * identity + (1 - cos_theta) * axis_outer + sin_theta * torch.diag(axis)
        rotated_vector = torch.matmul(rotation_matrix, vector)

        return rotated_vector

    common_rotation_axis = torch.ones(768)
    common_rotation_axis = common_rotation_axis / torch.norm(common_rotation_axis)

    model_features_text = " ".join([
        data.get("species", ""), data.get("age", ""), data.get("sex", ""),
        data.get("biosample_main", ""), data.get("biosample_detail", ""),
        data.get("experiment_type", ""), data.get("model_main", ""),
        data.get("model_detail1", ""), data.get("model_detail2", ""),
        data.get("model_detail3", ""), data.get("timepoint", "")
    ]).strip().replace("none", "")

    model_features_embedding = get_embedding(model_features_text)

    species = data.get("species", "").strip().lower()
    species_rotation_params = {
        "human": (174.20, 2.5152),
        "mouse": (176.36, 5.7104),
        "rat": (173.81, 5.7201)
    }

    if species in species_rotation_params:
        angle, norm_factor = species_rotation_params[species]
        rotation_axis = common_rotation_axis.to(model_features_embedding.device)
    
        model_features_embedding = rodrigues_rotation(model_features_embedding, angle, rotation_axis)
        model_features_embedding = model_features_embedding * norm_factor

    model_node.append(model_main_embedding)
    model_index.append(model_id)
    
    if model_id not in model_features:
        model_features[model_id] = []

    model_features[model_id].append(model_features_embedding)
   
    if model_id not in model_features_with_nodes:
        model_features_with_nodes[model_id] = {"model": model_main_embedding, "features": []}

    model_features_with_nodes[model_id]["features"].append(model_features_embedding)

    targets = data.get("targets", [])

    if isinstance(targets, dict):
        print(f"WARNING: `targets` is a dictionary! Converting to list: {targets}")
        targets = [targets]

    if not isinstance(targets, list):
        print(f"ERROR: Unexpected format for `targets`: {targets}")
        targets = []

    target_ids = []

    edge_index = []
    edge_attr = [] 
    
    for target in targets:
        target_text = target.get("target", "").strip().replace("none", "")

        if not target_text:
            print(f"WARNING: Skipping target with empty name -> {target}")
            continue

        target_dict_file = os.path.join(DATA_DIR, "./temp_json/target_dict.json")
        if not os.path.exists(target_dict_file):
            print(f"ERROR: reverse_target_dict.json not found at {target_dict_file}")
            exit(1)

        with open(target_dict_file, "r", encoding="utf-8") as f:
            target_dict = json.load(f)

        if target_text in target_dict:
            target_id = target_dict[target_text]
            count_target[target_id] = count_target.get(target_id, 0) + 1
        else:
            print(f"WARNING: target_text not found in target_dict -> {target_text}")
            continue

        target_ids.append(target_id)
        target_embedding = get_embedding(target_text)

        if isinstance(target, dict):
            target = [target]
    
        if isinstance(target, list):
            target_node.append(target_embedding)
        else:
            print(f"WARNING: Unexpected type for target: {type(target)}. Skipping append.")

        edge_weight_text = " ".join([
            target[0].get("target", ""), target[0].get("molecule_type", ""),
            target[0].get("analysis_main", ""), target[0].get("analysis_detail", ""),
            target[0].get("relation", ""), target[0].get("change", ""),
            target[0].get("significance", ""), target[0].get("control", "")
        ]).strip().replace("none", "")

        relation = target[0].get("relation", "").strip().lower()
        edge_weight_embedding = get_embedding(edge_weight_text)

        if relation == "increase":
            edge_weight_embedding = rodrigues_rotation(edge_weight_embedding, 90, common_rotation_axis)
        elif relation == "decrease":
            edge_weight_embedding = rodrigues_rotation(edge_weight_embedding, -90, common_rotation_axis)

        if target_id not in target_features:
            target_features[target_id] = []

        target_features[target_id].append(edge_weight_embedding)

        if target_id not in edge_weight_with_nodes:
            edge_weight_with_nodes[target_id] = {"target": target_embedding, "edge_weight": []} 

        edge_weight_with_nodes[target_id]["edge_weight"].append(edge_weight_embedding)

        if (model_id, target_id) not in edges_dict:
            edges_dict[(model_id, target_id)] = []

        edge_feature = torch.cat((model_features_embedding, edge_weight_embedding), dim=0)  # `model_features_embedding` + `edge_weight_embedding`

        new_edge_entry = {
            "model_features": model_features_embedding,
            "edge_weight": edge_weight_embedding
        }
        edges_dict[(model_id, target_id)].append(new_edge_entry)

        edge_index.append([model_id, target_id])
        total_edge_index.append([model_id, target_id])
        edge_attr.append(edge_feature)

    target_edge_index = []
    target_edge_attr = []
    target_edges_with_weights = {}

    for i in range(len(target_ids)):
        for j in range(i + 1, len(target_ids)):
            target_id_i, target_id_j = target_ids[i], target_ids[j]

            if (target_id_i, target_id_j) in target_edges_with_weights or (target_id_j, target_id_i) in target_edges_with_weights:
                continue

            edge_info_i = edges_dict.get((model_id, target_id_i), [])
            edge_info_j = edges_dict.get((model_id, target_id_j), [])

            if isinstance(edge_info_i, list) and len(edge_info_i) > 0:
                edge_info_i = edge_info_i[-1]

            if isinstance(edge_info_j, list) and len(edge_info_j) > 0:
                edge_info_j = edge_info_j[-1]

            if isinstance(edge_info_i, dict) and isinstance(edge_info_j, dict):
                edge_weights_i = edge_info_i["edge_weight"]
                edge_weights_j = edge_info_j["edge_weight"]
            else:
                print(f"WARNING: Skipping edge between {target_id_i} and {target_id_j} due to missing edge_weight.")
                continue

            target_edges.append((target_id_i, target_id_j))
            target_edges.append((target_id_j, target_id_i))

            if (target_id_i, target_id_j) not in target_edges_with_weights:
                target_edges_with_weights[(target_id_i, target_id_j)] = []
            if (target_id_j, target_id_i) not in target_edges_with_weights:
                target_edges_with_weights[(target_id_j, target_id_i)] = []

            target_edges_with_weights[(target_id_i, target_id_j)].append({
                "edge_weight_source": edge_weights_i,
                "edge_weight_target": edge_weights_j
            })
            target_edges_with_weights[(target_id_j, target_id_i)].append({
                "edge_weight_source": edge_weights_j,
                "edge_weight_target": edge_weights_i
            })

            if (target_id_i, target_id_j) not in target_edge_dict:
                target_edge_dict[(target_id_i, target_id_j)] = []
            if (target_id_j, target_id_i) not in target_edge_dict:
                target_edge_dict[(target_id_j, target_id_i)] = []

            target_edge_dict[(target_id_i, target_id_j)].append([edge_weights_i, edge_weights_j])
            target_edge_dict[(target_id_j, target_id_i)].append([edge_weights_j, edge_weights_i])

    for (target_id_i, target_id_j), weight_list in target_edges_with_weights.items():
            for weight in weight_list:
                target_edge_index.append([target_id_i, target_id_j])
                target_edge_attr.append(torch.cat((weight["edge_weight_source"], weight["edge_weight_target"]), dim=0))

    input_edge_index.extend(edge_index)
    input_edge_attr.extend(edge_attr)
    input_target_edge_index.extend(target_edge_index)
    input_target_edge_attr.extend(target_edge_attr)

    print(f"LOG: Saved processed record {index + 1}/{len(dataset)}")

json_files = {
    "input_edge_index": os.path.join(output_dir, "input_edge_index.json"),
    "input_edge_attr": os.path.join(output_dir, "input_edge_attr.json"),
    "input_target_edge_index": os.path.join(output_dir, "input_target_edge_index.json"),
    "input_target_edge_attr": os.path.join(output_dir, "input_target_edge_attr.json"),
}

with open(json_files["input_edge_index"], "w", encoding="utf-8") as f:
    json.dump(input_edge_index, f, indent=4)
with open(json_files["input_edge_attr"], "w", encoding="utf-8") as f:
    json.dump([t.tolist() for t in input_edge_attr], f, indent=4)
with open(json_files["input_target_edge_index"], "w", encoding="utf-8") as f:
    json.dump(input_target_edge_index, f, indent=4)
with open(json_files["input_target_edge_attr"], "w", encoding="utf-8") as f:
    json.dump([t.tolist() for t in input_target_edge_attr], f, indent=4)

model_features_with_nodes_file = os.path.join(output_dir, "input_model_features_with_nodes.json")
edge_weight_with_nodes_file = os.path.join(output_dir, "input_edge_weight_with_nodes.json")

with open(model_features_with_nodes_file, "w", encoding="utf-8") as f:
    json.dump(
        {str(k): {
            "model": v["model"].tolist() if isinstance(v["model"], torch.Tensor) else v["model"], 
            "features": [f.tolist() if isinstance(f, torch.Tensor) else f for f in v["features"]]
        }
        for k, v in model_features_with_nodes.items()},
        f, indent=4
    )

with open(edge_weight_with_nodes_file, "w", encoding="utf-8") as f:
    json.dump(
        {str(k): {  
            "target": v["target"].tolist() if isinstance(v["target"], torch.Tensor) else v["target"], 
            "edge_weight": [ew.tolist() if isinstance(ew, torch.Tensor) else ew for ew in v["edge_weight"]]
        }
        for k, v in edge_weight_with_nodes.items()},
        f, indent=4
    )

model_features_file = os.path.join(output_dir, "input_model_features.json")
target_features_file = os.path.join(output_dir, "input_target_features.json")

with open(model_features_file, "w", encoding="utf-8") as f:
    json.dump(
        {key: [tensor.tolist() for tensor in value]
         for key, value in model_features.items()}, 
        f, ensure_ascii=False, indent=4
    )

with open(target_features_file, "w", encoding="utf-8") as f:
    json.dump(
        {key: [tensor.tolist() for tensor in value]
         for key, value in target_features.items()}, 
        f, ensure_ascii=False, indent=4
    )

target_edge_dict_file = os.path.join(output_dir, "input_target_edge_dict.json")

target_edge_dict_serializable = {
    str(key): [[tensor_i.tolist(), tensor_j.tolist()] for tensor_i, tensor_j in value]
    for key, value in target_edge_dict.items()
}

with open(target_edge_dict_file, "w", encoding="utf-8") as f:
    json.dump(target_edge_dict_serializable, f, ensure_ascii=False, indent=4)

model_dict_file = os.path.join(output_dir, "input_model_dict.json")
target_dict_file = os.path.join(output_dir, "input_target_dict.json")

with open(model_dict_file, "w", encoding="utf-8") as f:
    json.dump(model_dict, f, ensure_ascii=False, indent=4)

with open(target_dict_file, "w", encoding="utf-8") as f:
    json.dump(target_dict, f, ensure_ascii=False, indent=4)

print("LOG: Data processing completed.")
