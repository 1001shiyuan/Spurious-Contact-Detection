import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, coo_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import igraph as ig
import leidenalg
import statsmodels.api as sm

# Load data
script_dir = '/Users/shaynewang/Documents/0_hi_c'
contact_matrix_path = os.path.join(script_dir, 'raw_contact_matrix.npz')
metadata_path = os.path.join(script_dir, 'corrected_contig_info_combine.csv')
contact_matrix = load_npz(contact_matrix_path).tocoo()
metadata = pd.read_csv(metadata_path)

# NormCC normalization 
def normcc(df):
    df['log_site'] = np.log(df['Number of restriction sites'])
    df['log_length'] = np.log(df['Contig length'])
    df['log_covcc'] = np.log(df['Contig coverage'])
    df['signal'] = df['Hi-C contacts mapped to the same contigs']
    df['log_site'] = np.where(df['log_site'] <= 0, 0.0001, df['log_site'])
    df['log_length'] = np.where(df['log_length'] <= 0, 0.0001, df['log_length'])
    df['log_covcc'] = np.where(df['log_covcc'] <= 0, 0.0001, df['log_covcc'])
    exog = df[['log_site', 'log_length', 'log_covcc']]
    endog = df[['signal']]
    exog = sm.add_constant(exog)
    glm_nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(alpha=1))
    res = glm_nb.fit(method="lbfgs")
    return res.params

# Normalize the contact matrix
def normalize_contact_matrix(contact_matrix, metadata, norm_params):
    norm_data = []
    mu = np.exp(metadata['log_site'] * norm_params['log_site'] +
                metadata['log_length'] * norm_params['log_length'] +
                metadata['log_covcc'] * norm_params['log_covcc'] +
                norm_params['const'])
    for idx, value in enumerate(contact_matrix.data):
        i, j = contact_matrix.row[idx], contact_matrix.col[idx]
        norm_value = value / np.sqrt(mu[i] * mu[j])
        norm_data.append(norm_value)
    return coo_matrix((norm_data, (contact_matrix.row, contact_matrix.col)), shape=contact_matrix.shape)

# Exclude self-loops
def exclude_self_loops(contact_matrix):
    mask = contact_matrix.row != contact_matrix.col
    return coo_matrix((contact_matrix.data[mask], 
                       (contact_matrix.row[mask], contact_matrix.col[mask])), 
                      shape=contact_matrix.shape)

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

# Calculate the 10% percentile cut-off
def get_cutoff(scores, percentile=10):
    scores_sorted = sorted(scores, key=lambda x: x[2])
    cutoff_index = int(len(scores_sorted) * percentile / 100)
    return scores_sorted[cutoff_index][2]

# Filter scores by cutoff
def filter_scores_by_cutoff(scores, cutoff):
    return [(i, j, score) for i, j, score in scores if score >= cutoff]

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

# Perform Leiden clustering
def perform_leiden_clustering(filtered_contact_matrix, random_seed):
    sources, targets = filtered_contact_matrix.row, filtered_contact_matrix.col
    weights = filtered_contact_matrix.data
    g = ig.Graph(len(set(sources) | set(targets)), list(zip(sources, targets)), edge_attrs={'weight': weights})
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', seed=random_seed)
    predicted_labels = part.membership
    return predicted_labels

# Evaluate clustering performance
def evaluate_clustering(true_labels, predicted_labels):
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return ari, nmi

# Main function
def main():
    contact_matrix = load_npz(contact_matrix_path).tocoo()
    metadata = pd.read_csv(metadata_path)
    norm_params = normcc(metadata)
    norm_contact_matrix = normalize_contact_matrix(contact_matrix, metadata, norm_params)
    norm_contact_matrix = exclude_self_loops(norm_contact_matrix)  # Exclude self-loops after normalization
    true_labels = metadata['True identity'].astype('category').cat.codes.values
    similarity_metrics = {
        'Salton': salton_index,
        'Sørensen': sorensen_index,
        'Jaccard': jaccard_index,
        'LHN': lhn_index
    }
    neighbors = extract_neighbors_and_weights(norm_contact_matrix)
    results = []
    for metric_name, metric_function in similarity_metrics.items():
        print(f"Calculating {metric_name} scores")
        scores = metric_function(neighbors)
        cutoff = get_cutoff(scores, 10)
        filtered_scores = filter_scores_by_cutoff(scores, cutoff)
        filtered_indices = set((i, j) for i, j, _ in filtered_scores)
        mask = np.array([index in filtered_indices for index in zip(norm_contact_matrix.row, norm_contact_matrix.col)])
        filtered_contact_matrix = coo_matrix(
            (norm_contact_matrix.data[mask], (norm_contact_matrix.row[mask], norm_contact_matrix.col[mask])),
            shape=norm_contact_matrix.shape
        )
        random_seed = 42
        predicted_labels = perform_leiden_clustering(filtered_contact_matrix, random_seed)
        ari, nmi = evaluate_clustering(true_labels, predicted_labels)

        results.append({
            'Metric': metric_name,
            'ARI': ari,
            'NMI': nmi
        })
    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == "__main__":
    main()
