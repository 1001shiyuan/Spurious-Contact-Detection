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

# Perform Leiden clustering
def perform_metacc_leiden_clustering(contact_matrix, random_seed):
    row, col, data = contact_matrix.row, contact_matrix.col, contact_matrix.data
    _vcount = max(max(row), max(col)) + 1
    g = ig.Graph(_vcount, list(zip(row, col)), edge_attrs={'weight': data})
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
    print("Performing clustering using normalized contacts without self-loops")
    predicted_labels = perform_metacc_leiden_clustering(norm_contact_matrix_without_self_loops, random_seed)
    ari, nmi, num_valid_bins = evaluate_clustering(true_labels, predicted_labels, contig_len, bin_size_threshold)
    results = {
        'ARI': ari,
        'NMI': nmi,
        'Number of Valid Bins': num_valid_bins
    }
    results_df = pd.DataFrame([results])
    print(results_df)

if __name__ == "__main__":
    main()
