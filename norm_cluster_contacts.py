import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, coo_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import igraph as ig
import leidenalg

# Load data
script_dir = '/Users/shaynewang/Documents/0_hi_c'
normalized_contact_matrix_path = os.path.join(script_dir, 'normalized_contact_matrix.npz')
metadata_path = os.path.join(script_dir, 'contig_info_yeast.csv')
normalized_contact_matrix = load_npz(normalized_contact_matrix_path).tocoo()
metadata = pd.read_csv(metadata_path)

# Extract neighbors and edge weights
def extract_neighbors_and_weights(contact_matrix):
    neighbors = {}
    row, col, data = contact_matrix.row, contact_matrix.col, contact_matrix.data
    for i, j, weight in zip(row, col, data):
        if i not in neighbors:
            neighbors[i] = {}
        if j not in neighbors:
            neighbors[j] = {}
        neighbors[i][j] = float(weight)
        neighbors[j][i] = float(weight)
    return neighbors

# Calculate the 5% percentile cut-off
def get_cutoff(scores, percentile=5):
    scores_sorted = sorted(scores, key=lambda x: x[2])
    cutoff_index = int(len(scores_sorted) * percentile / 100)
    return scores_sorted[cutoff_index][2]

# Filter scores by cutoff
def filter_scores_by_cutoff(scores, cutoff):
    return [(i, j) for i, j, score in scores if score >= cutoff]

# Calculate Salton Index
def salton_index(neighbors):
    salton_scores = []
    for i in neighbors:
        for j in neighbors[i]:
            neighbors_i = set(neighbors[i].keys())
            neighbors_j = set(neighbors[j].keys())
            common_neighbors = neighbors_i & neighbors_j
            numerator = sum(neighbors[i][k] * neighbors[j][k] for k in common_neighbors)
            denominator = np.sqrt(sum(neighbors[i][k]**2 for k in neighbors_i) * sum(neighbors[j][k]**2 for k in neighbors_j))
            salton_scores.append((i, j, numerator / denominator if denominator != 0 else 0))
    return salton_scores

# Calculate Sørensen Index
def sorensen_index(neighbors):
    sorensen_scores = []
    for i in neighbors:
        for j in neighbors[i]:
            neighbors_i = set(neighbors[i].keys())
            neighbors_j = set(neighbors[j].keys())
            common_neighbors = neighbors_i & neighbors_j
            numerator = 2 * sum(min(neighbors[i][k], neighbors[j][k]) for k in common_neighbors)
            denominator = sum(neighbors[i][k] for k in neighbors_i) + sum(neighbors[j][k] for k in neighbors_j)
            sorensen_scores.append((i, j, numerator / denominator if denominator != 0 else 0))
    return sorensen_scores

# Calculate Jaccard Index
def jaccard_index(neighbors):
    jaccard_scores = []
    for i in neighbors:
        for j in neighbors[i]:
            neighbors_i = set(neighbors[i].keys())
            neighbors_j = set(neighbors[j].keys())
            common_neighbors = neighbors_i & neighbors_j
            union_neighbors = neighbors_i | neighbors_j
            numerator = sum(min(neighbors[i][k], neighbors[j][k]) for k in common_neighbors)
            denominator = sum(max(neighbors[i].get(k, 0), neighbors[j].get(k, 0)) for k in union_neighbors)
            jaccard_scores.append((i, j, numerator / denominator if denominator != 0 else 0))
    return jaccard_scores

# Calculate Leicht-Holme-Newman Index
def lhn_index(neighbors):
    lhn_scores = []
    for i in neighbors:
        for j in neighbors[i]:
            neighbors_i = set(neighbors[i].keys())
            neighbors_j = set(neighbors[j].keys())
            common_neighbors = neighbors_i & neighbors_j
            ku = sum(neighbors[i][k] for k in neighbors_i)
            kv = sum(neighbors[j][k] for k in neighbors_j)
            numerator = sum(neighbors[i][k] * neighbors[j][k] for k in common_neighbors)
            lhn_scores.append((i, j, numerator / (ku * kv) if ku * kv != 0 else 0))
    return lhn_scores

# Calculate the local similarity metric scores
def calculate_similarity_scores(contact_matrix, metric_function):
    neighbors = extract_neighbors_and_weights(contact_matrix)
    return metric_function(neighbors)

# Perform Leiden clustering 
def perform_metacc_leiden_clustering(contact_matrix, filtered_pairs, random_seed):
    filtered_pairs_set = set(filtered_pairs)
    row, col, data = contact_matrix.row, contact_matrix.col, contact_matrix.data
    filtered_data = [data[i] for i in range(len(row)) if (row[i], col[i]) in filtered_pairs_set]
    filtered_row = [row[i] for i in range(len(row)) if (row[i], col[i]) in filtered_pairs_set]
    filtered_col = [col[i] for i in range(len(col)) if (row[i], col[i]) in filtered_pairs_set]
    _vcount = max(max(filtered_row), max(filtered_col)) + 1
    g = ig.Graph(_vcount, list(zip(filtered_row, filtered_col)), edge_attrs={'weight': filtered_data})
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', seed=random_seed, n_iterations=-1)
    dist_cluster = {v: part.membership[v] for v in range(_vcount)}
    predicted_labels = [dist_cluster.get(i, -1) for i in range(_vcount)]
    return predicted_labels

# Aggregate contig lengths into bins
def aggregate_contig_lengths(predicted_labels, contig_len):
    bin_lengths = {}
    for i, bin_id in enumerate(predicted_labels):
        if bin_id not in bin_lengths:
            bin_lengths[bin_id] = 0
        bin_lengths[bin_id] += contig_len[i]
    return bin_lengths

# Evaluate clustering performance
def evaluate_clustering(true_labels, predicted_labels, contig_len, bin_size_threshold):
    bin_lengths = aggregate_contig_lengths(predicted_labels, contig_len)
    valid_bins = {bin_id for bin_id, length in bin_lengths.items() if length >= bin_size_threshold}
    valid_indices = [i for i, bin_id in enumerate(predicted_labels) if bin_id in valid_bins]
    true_labels_filtered = true_labels[valid_indices]
    predicted_labels_filtered = np.array(predicted_labels)[valid_indices]
    ari = adjusted_rand_score(true_labels_filtered, predicted_labels_filtered)
    nmi = normalized_mutual_info_score(true_labels_filtered, predicted_labels_filtered)
    return ari, nmi, len(valid_bins)

def main():
    contact_matrix = load_npz(normalized_contact_matrix_path).tocoo()
    metadata = pd.read_csv(metadata_path)
    mask = contact_matrix.row != contact_matrix.col
    norm_contact_matrix_without_self_loops = coo_matrix((contact_matrix.data[mask], 
                                                        (contact_matrix.row[mask], contact_matrix.col[mask])), 
                                                       shape=contact_matrix.shape)
    true_labels = metadata['True identity'].astype('category').cat.codes.values
    contig_len = metadata['Contig length'].values
    random_seed = 42  
    bin_size_threshold = 150000
    similarity_metrics = {
        'Salton': salton_index,
        'Sørensen': sorensen_index,
        'Jaccard': jaccard_index,
        'LHN': lhn_index
    }
    results = []
    for metric_name, metric_function in similarity_metrics.items():
        print(f"Calculating {metric_name} scores without self-loops")
        scores = calculate_similarity_scores(norm_contact_matrix_without_self_loops, metric_function)
        cutoff = get_cutoff(scores, 10)
        filtered_pairs = filter_scores_by_cutoff(scores, cutoff)
        predicted_labels = perform_metacc_leiden_clustering(norm_contact_matrix_without_self_loops, filtered_pairs, random_seed)
        ari, nmi, num_valid_bins = evaluate_clustering(true_labels, predicted_labels, contig_len, bin_size_threshold)
        results.append({
            'Metric': metric_name,
            'Self Loops': False,
            'ARI': ari,
            'NMI': nmi,
            'Number of Valid Bins': num_valid_bins
        })

    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == "__main__":
    main()
