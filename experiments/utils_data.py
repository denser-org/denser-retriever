import os

import numpy as np


def standardize_normalize(data):
    # Convert list to numpy array for easier manipulation
    arr = np.array(data)

    # Calculate mean and standard deviation of the array
    mean = np.mean(arr)
    std_dev = np.std(arr)

    # Standardize the array
    if std_dev != 0:
        standardized_arr = (arr - mean) / std_dev
    else:
        standardized_arr = arr - mean
        print("Warning: std_dev is zero")

    # Convert back to list if necessary
    return standardized_arr.tolist()


def min_max_normalize(data):
    # Convert list to numpy array for easier manipulation
    arr = np.array(data)

    # Find the minimum and maximum values of the array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Apply min-max normalization
    if max_val - min_val != 0:
        normalized_arr = (arr - min_val) / (max_val - min_val)
    else:
        normalized_arr = arr - min_val
        print("Warning: max_val - min_val is zero")

    # Convert back to list if needed
    return normalized_arr.tolist()


def save_data(group_data, output_feature, output_group, features, features_to_normalize):
    if len(group_data) == 0:
        return

    output_group.write(str(len(group_data)) + "\n")
    # collect feature values
    if features_to_normalize:
        features_raw = {f: [] for f in features_to_normalize}
        for data in group_data:
            for p in data[2:]:
                f_name, f_value = p.split(":")
                if f_name in features_to_normalize:
                    features_raw[f_name].append(float(f_value))

        # normalized features_raw
        features_standardize = {}
        features_min_max = {}
        for f in features_to_normalize:
            features_standardize[f] = standardize_normalize(features_raw[f])
            features_min_max[f] = min_max_normalize(features_raw[f])

    for i, data in enumerate(group_data):
        # only include nonzero features
        feats = []

        for p in data[2:]:
            f_name, f_value = p.split(":")
            if features and f_name not in features:
                continue
            if float(f_value) != 0.0:
                feats.append(p)

        if features_to_normalize:
            f_id = len(data[2:]) + 1
            feats_normalized = []
            for j, f in enumerate(features_to_normalize):
                if features_standardize[f][i] != 0.0:
                    feats_normalized.append(f"{f_id + 2 * j}:{features_standardize[f][i]}")
                if features_min_max[f][i] != 0.0:
                    feats_normalized.append(f"{f_id + 2 * j + 1}:{features_min_max[f][i]}")
            output_feature.write(data[0] + " " + " ".join(feats) + " " + " ".join(feats_normalized) + "\n")
        else:
            output_feature.write(data[0] + " " + " ".join(feats) + "\n")


def prepare_xgbdata(exp_dir, out_file, out_group_file, features_to_use, features_to_normalize):
    fi = open(os.path.join(exp_dir, "features.svmlight"))
    output_feature = open(os.path.join(exp_dir, out_file), "w")
    output_group = open(os.path.join(exp_dir, out_group_file), "w")
    if features_to_use:
        features_to_use = features_to_use.split(",")
    if features_to_normalize:
        features_to_normalize = features_to_normalize.split(",")

    group_data = []
    group = ""
    for line in fi:
        if not line:
            break
        if "#" in line:
            line = line[: line.index("#")]
        splits = line.strip().split(" ")
        if splits[1] != group:
            # print(f"Processing group {group}")
            save_data(group_data, output_feature, output_group, features_to_use, features_to_normalize)
            group_data = []
        group = splits[1]
        group_data.append(splits)

    save_data(group_data, output_feature, output_group, features_to_use, features_to_normalize)

    fi.close()
    output_feature.close()
    output_group.close()
