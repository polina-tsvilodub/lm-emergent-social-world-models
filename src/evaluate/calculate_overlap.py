# import numpy as np
# import os
# # network_1, network_2: the two tasks to compare, example_no_1, example_no_2: the number of samples used for
# # localization, e.g. example_no_1="_samples=2000", if you have different versions
# def calculate_overlap(network_1, network_2, percentage, example_no_1="", example_no_2=""):
#     # data1 = np.load(f'./cache/Llama-3.2-3B-Instruct_network={network_1}_pooling=mean_pretrained=True_pvalues{example_no_1}.npy')
#     data = np.load(f'./cache/{model_name}_network={network}_pooling={pooling}_pretrained=True_dataset=tom_pvalues.npy')
#     data2 = np.load(f'./cache/Llama-3.2-3B-Instruct_network={network_2}_pooling=mean_pretrained=True_pvalues{example_no_2}.npy')

#     all_layers = []
#     for i in range(len(data1)):
#         all_layers = all_layers + list(data1[i])
#     threshold_1 = np.sort(all_layers)[int((len(all_layers) * percentage) // 100)]

#     all_layers = []
#     for i in range(len(data2)):
#         all_layers = all_layers + list(data2[i])
#     threshold_2 = np.sort(all_layers)[int((len(all_layers) * percentage) // 100)]
#     total = int((len(all_layers) * percentage) // 100)

#     mask1 = data1 < threshold_1
#     mask2 = data2 < threshold_2

#     overlap = [[1 if mask1[l][i] and mask2[l][i] else 0 for i in range(len(data1[l]))] for l in range(len(data1))]
#     return overlap, sum([sum(overlap[i]) for i in range(len(data1))])/total

# def get_overlaps(network_1, network_2, example_no_1="", example_no_2="", lower=0.5, upper=10, stepsize=0.5):
#     return [calculate_overlap(network_1, network_2, p, example_no_1, example_no_2)[1] for p in np.arange(lower, upper, stepsize)]

# percentage = 1
# # threshold = 0.05
# network = "theory-of-mind" #"multiple-demand" 

# model_list = [
#     "meta-llama/Llama-3.1-8B-Instruct", 
#     # "meta-llama/Llama-3.2-3B-Instruct", 
#     "mistralai/Mistral-7B-Instruct-v0.3", 
#     # "mistralai/Mistral-7B-v0.3", 
#     # "meta-llama/Llama-2-7b-hf", 
#     # "meta-llama/Llama-2-7b-chat-hf"
#     "meta-llama/Llama-2-13b-chat-hf",
#     # "openai-community/gpt2-large",
#     # "tiiuae/falcon-7b",
#     # "tiiuae/falcon-7b-instruct",
#     # "google/gemma-2b",
#     # "google/gemma-7b",
#     "google/gemma-1.1-7b-it",
# ]

# localizer_list = [
#     "tom_matched", "photograph_synthetic", "tom_matched_synthetic", "tom_synthetic", "tomi_fo", "tomi_so", "fauxpas" # 
# ]

# for model_name in model_list:
#     for ds in localizer_list:
#         plot_data = {"selectivity": [], "layer_num": [], "model_name": []}  

#         model_name = os.path.basename(model_name)
#         # num_layers = get_num_blocks(model_name)
#         # hidden_dim = get_hidden_dim(model_name)

#         print(f"Model: {model_name}")

#         pooling = "mean" #  "mean" if network != "lang" else
#         # model_loc_path = f"{model_name}_network={network}_pooling={pooling}_range=100-100_perc={percentage}_nunits=None_pretrained=True.npy"
#         model_loc_path = f"{model_name}_network={network}_pooling={pooling}_pretrained=True_dataset={ds}_pvalues.npy"

#         cache_dir = "../cache"
#         lang_mask_path = f"{cache_dir}/{model_loc_path}"
#         if not os.path.exists(lang_mask_path):
#             print(f"Path does not exist: {lang_mask_path}")
#             continue
        
#         lang_mask_p_values = np.load(lang_mask_path)

# from matplotlib import pyplot as plt

# plt.plot(np.arange(0.5, 10, 0.5), get_overlaps("tomi-fo","tomi-so","_samples=20","_samples=20"), color="pink", label="tomi-fo:tomi-so,n=20")
# plt.plot(np.arange(0.5, 10, 0.5), get_overlaps("tomi-fo","tomi-so","_samples=200","_samples=200"), color="red", label="tomi-fo:tomi-so,n=200")
# plt.plot(np.arange(0.5, 10, 0.5), get_overlaps("tomi-fo","tomi-so","_samples=2000","_samples=2000"), color="darkred", label="tomi-fo:tomi-so,n=2000")
# plt.plot(np.arange(0.5, 10, 0.5), get_overlaps("tomi-fo","theory-of-mind","_samples=20",""), color="lightgreen", label="tomi-fo:tom,n=20")
# plt.plot(np.arange(0.5, 10, 0.5), get_overlaps("tomi-fo","theory-of-mind","_samples=200",""), color="green", label="tomi-fo:tom,n=200")
# plt.plot(np.arange(0.5, 10, 0.5), get_overlaps("tomi-fo","theory-of-mind","_samples=2000",""), color="darkgreen", label="tomi-fo:tom,n=2000")
# plt.plot(np.arange(0.5, 10, 0.5), get_overlaps("tomi-so","theory-of-mind","_samples=20",""), color="lightblue", label="tomi-so:tom,n=20")
# plt.plot(np.arange(0.5, 10, 0.5), get_overlaps("tomi-so","theory-of-mind","_samples=200",""), color="blue", label="tomi-so:tom,n=200")
# plt.plot(np.arange(0.5, 10, 0.5), get_overlaps("tomi-so","theory-of-mind","_samples=2000",""), color="darkblue", label="tomi-so:tom,n=2000")
# plt.plot(np.arange(0.5,10,0.5), np.arange(0.005,0.1,0.005), color="grey")

# plt.show()
import numpy as np
import os
import pandas as pd

def load_pvalues(model_name, network, pooling, dataset):
    path = f"./cache/{model_name}_network={network}_pooling={pooling}_pretrained=True_dataset={dataset}_pvalues.npy"
    if os.path.exists(path):
        return np.load(path), path
    return None, path


def calculate_overlap(data1, data2, percentage, ds_path):
    # flatten both
    flat1 = np.concatenate([layer for layer in data1])
    flat2 = np.concatenate([layer for layer in data2])

    # thresholds
    t1 = np.sort(flat1)[int(len(flat1) * percentage / 100)]
    t2 = np.sort(flat2)[int(len(flat2) * percentage / 100)]
    total = int(len(flat1) * percentage / 100)

    # boolean masks
    mask1 = data1 < t1
    if "tomi_" in ds_path:
        mask2 = data2 <= t2
    else:
        mask2 = data2 < t2

    overlap = (mask1 & mask2).sum()
    return overlap / total if total > 0 else np.nan


# -----------------------------------------------------------------------------------------
# Main section
# -----------------------------------------------------------------------------------------

percentage = 1
network = "theory-of-mind"
pooling = "mean"

model_list = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-2-13b-chat-hf",
    "google/gemma-1.1-7b-it",
]

localizer_list = [
    "tom_matched", "photograph_synthetic", "tom_matched_synthetic",
    "tom_synthetic", "tomi_fo", "tomi_so", "fauxpas"
]

results = []

for model in model_list:
    model_name = os.path.basename(model)

    # load reference TOM dataset
    tom_data, tom_path = load_pvalues(model_name, network, pooling, "tom")
    if tom_data is None:
        print(f"[SKIP] TOM file missing: {tom_path}")
        continue

    print(f"\nModel: {model_name}")
    print("TOM pvalues loaded.")

    for ds in localizer_list:
        ds_data, ds_path = load_pvalues(model_name, network, pooling, ds)

        if ds_data is None:
            print(f"[MISSING] {ds_path}")
            continue

        ov = calculate_overlap(tom_data, ds_data, percentage, ds_path)

        results.append({
            "model": model_name,
            "reference_dataset": "tom",
            "comparison_dataset": ds,
            "percentage": percentage,
            "overlap": ov
        })

        print(f"Overlap tom vs {ds}: {ov:.4f}")


# Convert to DataFrame
df = pd.DataFrame(results)
print("\n=== Overlap Table ===")
print(df)
# save to csv
df.to_csv(f"./cache/2025-11-17/overlap_results_network={network}_pooling={pooling}_percentage={percentage})_updated.csv", index=False)