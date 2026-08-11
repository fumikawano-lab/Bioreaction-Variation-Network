import os
import json
import torch
import ijson
import gc

DATA_DIR = "./data/temp_json"  # ./data/input_data/input_temp_json for input data
OUTPUT_DIR = "./data/pt_data"  # ./data/input_data/input_pt_data for input data
os.makedirs(OUTPUT_DIR, exist_ok=True)

edge_index_file = os.path.join(DATA_DIR, "edge_index.json")
edge_attr_file = os.path.join(DATA_DIR, "edge_attr.json")
model_features_with_nodes_file = os.path.join(DATA_DIR, "model_features_with_nodes.json")
target_edge_index_file = os.path.join(DATA_DIR, "target_edge_index.json")
target_edge_attr_file = os.path.join(DATA_DIR, "target_edge_attr.json")
edge_weight_with_nodes_file = os.path.join(DATA_DIR, "edge_weight_with_nodes.json")
model_features_file = os.path.join(DATA_DIR, "model_features.json")
target_features_file = os.path.join(DATA_DIR, "target_features.json")
target_edge_dict_file = os.path.join(DATA_DIR, "target_edge_dict.json")

edge_index_pt = os.path.join(OUTPUT_DIR, "edge_index.pt")
edge_attr_pt = os.path.join(OUTPUT_DIR, "edge_attr.pt")
model_features_with_nodes_pt = os.path.join(OUTPUT_DIR, "model_features_with_nodes.pt")
target_edge_index_pt = os.path.join(OUTPUT_DIR, "target_edge_index.pt")
target_edge_attr_pt = os.path.join(OUTPUT_DIR, "target_edge_attr.pt")
edge_weight_with_nodes_pt = os.path.join(OUTPUT_DIR, "edge_weight_with_nodes.pt")
model_features_pt = os.path.join(OUTPUT_DIR, "model_features.pt")
target_features_pt = os.path.join(OUTPUT_DIR, "target_features.pt")
target_edge_dict_pt = os.path.join(OUTPUT_DIR, "target_edge_dict.pt")

target_edge_attr_parts_dir = os.path.join(OUTPUT_DIR, "target_edge_attr_parts")
os.makedirs(target_edge_attr_parts_dir, exist_ok=True)

target_edge_dict_parts_dir = os.path.join(OUTPUT_DIR, "target_edge_dict_parts")
os.makedirs(target_edge_dict_parts_dir, exist_ok=True)

def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

edge_index = load_json(edge_index_file)
if edge_index:
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).T
    torch.save(edge_index_tensor, edge_index_pt)
    print(f"Saved: {edge_index_pt}")

edge_attr = load_json(edge_attr_file)
if edge_attr:
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    torch.save(edge_attr_tensor, edge_attr_pt)
    print(f"Saved: {edge_attr_pt}")

model_features_with_nodes = load_json(model_features_with_nodes_file)

if model_features_with_nodes:    
    feature_counts = {
        int(model_id): len(v["features"]) for model_id, v in model_features_with_nodes.items()
    }

    top_10_models = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    for model_id, num_features in top_10_models:
        print(f"  - Model ID: {model_id} | Features: {num_features}")

    converted_feature_data = {}
    for model_id, v in model_features_with_nodes.items():
        model_id = int(model_id)

        model_tensor = torch.tensor(v["model"], dtype=torch.float32) if "model" in v else torch.zeros(768)

        feature_tensors = [torch.tensor(f, dtype=torch.float32) for f in v["features"]]
        num_features = len(feature_tensors)

        if num_features > 0:
            avg_feature_tensor = torch.stack(feature_tensors).mean(dim=0)
        else:
            avg_feature_tensor = torch.zeros(768)

        converted_feature_data[model_id] = {
            "model": model_tensor,
            "features": avg_feature_tensor
        }

    torch.save(converted_feature_data, model_features_with_nodes_pt)
    print(f"\nSaved: {model_features_with_nodes_pt}")
else:
    print("No data found in model_features_with_nodes.json")

target_edge_index = load_json(target_edge_index_file)
if target_edge_index:
    target_edge_index_tensor = torch.tensor(target_edge_index, dtype=torch.long).T
    torch.save(target_edge_index_tensor, target_edge_index_pt)
    print(f"Saved: {target_edge_index_pt}")

if os.path.exists(target_edge_attr_file):
    batch_size = 100_000
    part_id = 0
    attr_list = []
    with open(target_edge_attr_file, "r", encoding="utf-8") as f:
        for i, vec in enumerate(ijson.items(f, "item")):
            attr_list.append(torch.tensor(vec, dtype=torch.float32))
            if (i + 1) % batch_size == 0:
                path = os.path.join(target_edge_attr_parts_dir, f"part_{part_id}.pt")
                torch.save(torch.stack(attr_list), path)
                print(f"Saved: {path} ({i+1} entries)")
                attr_list.clear()
                gc.collect()
                part_id += 1
        if attr_list:
            path = os.path.join(target_edge_attr_parts_dir, f"part_{part_id}.pt")
            torch.save(torch.stack(attr_list), path)
            print(f"Saved final: {path}")
            attr_list.clear()
            gc.collect()
    all_parts = [torch.load(os.path.join(target_edge_attr_parts_dir, f))
                 for f in sorted(os.listdir(target_edge_attr_parts_dir)) if f.endswith(".pt")]
    merged_tensor = torch.cat(all_parts, dim=0)
    torch.save(merged_tensor, target_edge_attr_pt)
    print(f"Merged and saved: {target_edge_attr_pt}")
else:
    print("target_edge_attr.json not found.")

edge_weight_with_nodes = load_json(edge_weight_with_nodes_file)

if edge_weight_with_nodes:    
    edge_weight_counts = {
        int(target_id): len(v["edge_weight"]) for target_id, v in edge_weight_with_nodes.items()
    }

    top_10_edges = sorted(edge_weight_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    for target_id, num_edge_weight in top_10_edges:
        print(f"  - Target ID: {target_id} | Edge weight: {num_edge_weight}")

    converted_edge_weight_data = {}
    for target_id, v in edge_weight_with_nodes.items():
        target_id = int(target_id)

        target_tensor = torch.tensor(v["target"], dtype=torch.float32) if "target" in v else torch.zeros(768)

        edge_weight_tensors = [torch.tensor(f, dtype=torch.float32) for f in v["edge_weight"]]
        num_edge_weight = len(edge_weight_tensors)

        if num_edge_weight > 0:
            avg_edge_weight_tensor = torch.stack(edge_weight_tensors).mean(dim=0)
        else:
            avg_edge_weight_tensor = torch.zeros(768)

        converted_edge_weight_data[target_id] = {
            "target": target_tensor,
            "edge_weight": avg_edge_weight_tensor
        }

    torch.save(converted_edge_weight_data, edge_weight_with_nodes_pt)
    print(f"\nSaved: {edge_weight_with_nodes_pt}")
else:
    print("No data found in edge_weight_with_nodes.json")

model_features = load_json(model_features_file)
if model_features:
    converted_model_features = {
        int(model_id): torch.mean(torch.stack([torch.tensor(f, dtype=torch.float32) for f in v]), dim=0) 
        for model_id, v in model_features.items()
    }
    torch.save(converted_model_features, model_features_pt)
    print(f"Saved: {model_features_pt}")
else:
    print("No data found in model_features.json")

target_features = load_json(target_features_file)
if target_features:
    converted_target_features = {
        int(target_id): torch.mean(torch.stack([torch.tensor(f, dtype=torch.float32) for f in v]), dim=0) 
        for target_id, v in target_features.items()
    }
    torch.save(converted_target_features, target_features_pt)
    print(f"Saved: {target_features_pt}")
else:
    print("No data found in target_features.json")

if os.path.exists(target_edge_dict_file):
    print("\nConverting target_edge_dict.json in streaming mode...")
    batch_size = 5000
    part_id = 0
    count = 0
    converted_dict = {}

    with open(target_edge_dict_file, "r", encoding="utf-8") as f:
        for key, value in ijson.kvitems(f, ""):
            key_tuple = eval(key)
            tensor_pairs = [(torch.tensor(i, dtype=torch.float32), torch.tensor(j, dtype=torch.float32))
                            for i, j in value]
            src_avg = torch.stack([t[0] for t in tensor_pairs]).mean(dim=0)
            tgt_avg = torch.stack([t[1] for t in tensor_pairs]).mean(dim=0)
            converted_dict[key_tuple] = (src_avg, tgt_avg)
            count += 1

            if count % batch_size == 0:
                path = os.path.join(target_edge_dict_parts_dir, f"part_{part_id}.pt")
                torch.save(converted_dict, path)
                print(f"Saved: {path} ({count} entries)")
                converted_dict.clear()
                gc.collect()
                part_id += 1

        if converted_dict:
            path = os.path.join(target_edge_dict_parts_dir, f"part_{part_id}.pt")
            torch.save(converted_dict, path)
            print(f"Saved final: {path}")
            converted_dict.clear()
            gc.collect()

    merged = {}
    for f in sorted(os.listdir(target_edge_dict_parts_dir)):
        if f.endswith(".pt"):
            part = torch.load(os.path.join(target_edge_dict_parts_dir, f))
            merged.update(part)
    torch.save(merged, target_edge_dict_pt)
    print(f"Merged and saved: {target_edge_dict_pt}")
else:
    print("target_edge_dict.json not found.")

print("All JSON files converted to .pt format successfully.")
