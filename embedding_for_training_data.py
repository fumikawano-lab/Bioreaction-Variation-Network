import os
import json
import torch
import gc
import csv
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

input_file = os.path.join(DATA_DIR, "./clean_data/YOUR_FILE_NAME_HERE.json")

if not os.path.exists(input_file):
    print(f"ERROR: Dataset file not found at {input_file}")
    exit(1)

with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(f"LOG: Loaded {len(dataset)} records from dataset")

output_dir = "./data/temp_json"
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

batch_size = 100
batch_edge_index = []
batch_edge_attr = []
batch_target_edge_index = []
batch_target_edge_attr = []
batch_count = 0

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

    if model_main_text not in model_dict:
        model_id = len(model_dict)
        model_dict[model_main_text] = model_id
        reverse_model_dict[model_id] = model_main_text
        count_model[model_id] = 1
    else:
        model_id = model_dict[model_main_text]
        count_model[model_id] += 1

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

        if target_text not in target_dict:
            target_id = len(target_dict)
            target_dict[target_text] = target_id
            reverse_target_dict[target_id] = target_text
            count_target[target_id] = 1
        else:
            target_id = target_dict[target_text]
            count_target[target_id] += 1

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

        edge_feature = torch.cat((model_features_embedding, edge_weight_embedding), dim=0)

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

    batch_edge_index.extend(edge_index)
    batch_edge_attr.extend(edge_attr)
    batch_target_edge_index.extend(target_edge_index)
    batch_target_edge_attr.extend(target_edge_attr)
    batch_count += 1

    if batch_count >= batch_size:
        batch_id = (index + 1) // batch_size
        json_files = {
            "edge_index": os.path.join(output_dir, f"batch_{batch_id}_edge_index.json"),
            "edge_attr": os.path.join(output_dir, f"batch_{batch_id}_edge_attr.json"),
            "target_edge_index": os.path.join(output_dir, f"batch_{batch_id}_target_edge_index.json"),
            "target_edge_attr": os.path.join(output_dir, f"batch_{batch_id}_target_edge_attr.json"),
        }

        with open(json_files["edge_index"], "w", encoding="utf-8") as f:
            json.dump(batch_edge_index, f, indent=4)
        with open(json_files["edge_attr"], "w", encoding="utf-8") as f:
            json.dump([t.tolist() for t in batch_edge_attr], f, indent=4)
        with open(json_files["target_edge_index"], "w", encoding="utf-8") as f:
            json.dump(batch_target_edge_index, f, indent=4)
        with open(json_files["target_edge_attr"], "w", encoding="utf-8") as f:
            json.dump([t.tolist() for t in batch_target_edge_attr], f, indent=4)

        print(f"📄 Batch {batch_id} has been saved")
        batch_edge_index.clear()
        batch_edge_attr.clear()
        batch_target_edge_index.clear()
        batch_target_edge_attr.clear()
        batch_count = 0

if batch_count > 0:
    batch_id += 1
    json_files = {
        "edge_index": os.path.join(output_dir, f"batch_{batch_id}_edge_index.json"),
        "edge_attr": os.path.join(output_dir, f"batch_{batch_id}_edge_attr.json"),
        "target_edge_index": os.path.join(output_dir, f"batch_{batch_id}_target_edge_index.json"),
        "target_edge_attr": os.path.join(output_dir, f"batch_{batch_id}_target_edge_attr.json"),
    }

    with open(json_files["edge_index"], "w", encoding="utf-8") as f:
        json.dump(batch_edge_index, f, indent=4)
    with open(json_files["edge_attr"], "w", encoding="utf-8") as f:
        json.dump([t.tolist() for t in batch_edge_attr], f, indent=4)
    with open(json_files["target_edge_index"], "w", encoding="utf-8") as f:
        json.dump(batch_target_edge_index, f, indent=4)
    with open(json_files["target_edge_attr"], "w", encoding="utf-8") as f:
        json.dump([t.tolist() for t in batch_target_edge_attr], f, indent=4)

    print(f"📄 Final batch {batch_id} has been saved")

    batch_edge_index.clear()
    batch_edge_attr.clear()
    batch_target_edge_index.clear()
    batch_target_edge_attr.clear()
    batch_count = 0

if index == len(dataset) - 1:
    print(f"LOG: Processing record {index + 1}/{len(dataset)}")
    print(f"LOG: Current model_id count: {len(model_dict)}")
    print(f"LOG: Current target_id count: {len(target_dict)}")
    print(f"LOG: Current edge count: {len(total_edge_index)}")
    print(f"LOG: Current target_edge count: {len(target_edges)}")
    print(f"LOG: Current target_edge_dict count: {len(target_edge_dict)}")

    top_models = sorted(count_model.items(), key=lambda x: x[1], reverse=True)[:10]
    print("LOG: Top 10 most frequent model_ids:")
    for model_id, count in top_models:
        model_main_text = reverse_model_dict.get(model_id, "Unknown")
        print(f"  - {model_id} ({model_main_text}): {count} occurrences")

    top_targets = sorted(count_target.items(), key=lambda x: x[1], reverse=True)[:10]
    print("LOG: Top 10 most frequent target_ids:")
    for target_id, count in top_targets:
        target_text = reverse_target_dict.get(target_id, "Unknown")
        print(f"  - {target_id} ({target_text}): {count} occurrences")

model_features_with_nodes_file = os.path.join(output_dir, "model_features_with_nodes.json")
edge_weight_with_nodes_file = os.path.join(output_dir, "edge_weight_with_nodes.json")

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

model_features_file = os.path.join(output_dir, "model_features.json")
target_features_file = os.path.join(output_dir, "target_features.json")

with open(model_features_file, "w", encoding="utf-8") as f:
    json.dump(
        {key: [tensor.tolist() for tensor in value]  # 変更点: value["features"] → value
         for key, value in model_features.items()}, 
        f, ensure_ascii=False, indent=4
    )

with open(target_features_file, "w", encoding="utf-8") as f:
    json.dump(
        {key: [tensor.tolist() for tensor in value]  # 変更点: value["features"] → value
         for key, value in target_features.items()}, 
        f, ensure_ascii=False, indent=4
    )

target_edge_dict_file = os.path.join(output_dir, "target_edge_dict.json")

target_edge_dict_serializable = {
    str(key): [[tensor_i.tolist(), tensor_j.tolist()] for tensor_i, tensor_j in value]
    for key, value in target_edge_dict.items()
}

with open(target_edge_dict_file, "w", encoding="utf-8") as f:
    json.dump(target_edge_dict_serializable, f, ensure_ascii=False, indent=4)

model_dict_file = os.path.join(output_dir, "model_dict.json")
target_dict_file = os.path.join(output_dir, "target_dict.json")

with open(model_dict_file, "w", encoding="utf-8") as f:
    json.dump(model_dict, f, ensure_ascii=False, indent=4)

with open(target_dict_file, "w", encoding="utf-8") as f:
    json.dump(target_dict, f, ensure_ascii=False, indent=4)

reverse_model_dict_file = os.path.join(output_dir, "reverse_model_dict.json")
reverse_target_dict_file = os.path.join(output_dir, "reverse_target_dict.json")

with open(reverse_model_dict_file, "w", encoding="utf-8") as f:
    json.dump(reverse_model_dict, f, ensure_ascii=False, indent=4)

with open(reverse_target_dict_file, "w", encoding="utf-8") as f:
    json.dump(reverse_target_dict, f, ensure_ascii=False, indent=4)

model_output_file = os.path.join(output_dir, "model_id_frequency.csv")
sorted_models = sorted(count_model.items(), key=lambda x: x[1], reverse=True)

with open(model_output_file, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model ID", "Model Main Text", "Count"])
    for model_id, count in sorted_models:
        model_main_text = reverse_model_dict.get(model_id, "Unknown")
        writer.writerow([model_id, model_main_text, count])

target_output_file = os.path.join(output_dir, "target_id_frequency.csv")
sorted_targets = sorted(count_target.items(), key=lambda x: x[1], reverse=True)

with open(target_output_file, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Target ID", "Target Text", "Count"])
    for target_id, count in sorted_targets:
        target_text = reverse_target_dict.get(target_id, "Unknown")
        writer.writerow([target_id, target_text, count])

merged_files = {
    "edge_index": os.path.join(output_dir, "edge_index.json"),
    "edge_attr": os.path.join(output_dir, "edge_attr.json"),
    "target_edge_index": os.path.join(output_dir, "target_edge_index.json"),
    "target_edge_attr": os.path.join(output_dir, "target_edge_attr.json")
}

merged_data = {
    "edge_index": [],
    "edge_attr": [],
    "target_edge_index": [],
    "target_edge_attr": []
}

for file_name in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file_name)

    if not os.path.isfile(file_path) or not file_name.endswith(".json"):
        continue

    if file_name.startswith("batch_") and "_edge_index.json" in file_name and "target_edge_index.json" not in file_name:
        key = "edge_index"
    elif file_name.startswith("batch_") and "_edge_attr.json" in file_name and "target_edge_attr.json" not in file_name:
        key = "edge_attr"
    elif file_name.startswith("batch_") and "_target_edge_index.json" in file_name:
        key = "target_edge_index"
    elif file_name.startswith("batch_") and "_target_edge_attr.json" in file_name:
        key = "target_edge_attr"
    else:
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        if key in ["edge_index", "edge_attr", "target_edge_index", "target_edge_attr"]:
            merged_data[key].extend(data)

for key, file_path in merged_files.items():
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(merged_data[key], f, indent=4)

print("LOG: Data processing completed.")
