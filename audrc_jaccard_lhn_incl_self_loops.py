import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, coo_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import statsmodels.api as sm

# Load data
script_dir = '/Users/shaynewang/Documents/0_hi_c'
contact_matrix_path = os.path.join(script_dir, 'raw_contact_matrix.npz')
metadata_path = os.path.join(script_dir, 'corrected_contig_info_combine.csv')
contact_matrix = load_npz(contact_matrix_path).tocoo()
metadata = pd.read_csv(metadata_path)

# Function to extract neighbors and edge weights
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

# Function to calculate true positives
def calculate_true_positives(contact_matrix, metadata):
    species_map = dict(zip(metadata['Contig name'], metadata['True identity']))
    true_positives = set()
    for i, j in zip(contact_matrix.row, contact_matrix.col):
        contig1 = metadata.loc[i, 'Contig name']
        contig2 = metadata.loc[j, 'Contig name']
        if species_map.get(contig1) == species_map.get(contig2):
            true_positives.add((i, j))
    return true_positives

# Function to calculate Jaccard Index
def jaccard_index(contact_matrix):
    neighbors = extract_neighbors_and_weights(contact_matrix)
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

# Function to calculate Leicht-Holme-Newman Index
def lhn_index(contact_matrix):
    neighbors = extract_neighbors_and_weights(contact_matrix)
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

# Function to calculate AUDRC
def calculate_audrc(scores):
    scores.sort(key=lambda x: x[2], reverse=True)
    true_positive_count = sum(1 for _, _, _, is_tp in scores if is_tp)
    spurious_contact_count = len(scores) - true_positive_count
    tpr = []
    discard_proportion = []
    thresholds = np.percentile([x[2] for x in scores], np.arange(0, 100, 5))
    for threshold in thresholds:
        tp_cumsum = sum(1 for _, _, score, is_tp in scores if score >= threshold and is_tp)
        spurious_cumsum = sum(1 for _, _, score, is_tp in scores if score < threshold and not is_tp)
        tpr.append(tp_cumsum / true_positive_count)
        discard_proportion.append(spurious_cumsum / spurious_contact_count)
    audrc = auc(discard_proportion, tpr)
    return audrc, tpr, discard_proportion

# Function to plot the discard-retain curve
def plot_discard_retain_curve(discard_proportion, tpr, audrc, method_label, color):
    plt.plot(discard_proportion, tpr, label=f'{method_label}: {audrc:.3f}', color=color)

# NormCC normalization function 
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

# Main function
def main():
    true_positives = calculate_true_positives(contact_matrix, metadata)
    # NormCC normalization
    norm_params = normcc(metadata)
    norm_data = []
    mu = np.exp(metadata['log_site'] * norm_params['log_site'] +
                metadata['log_length'] * norm_params['log_length'] +
                metadata['log_covcc'] * norm_params['log_covcc'] +
                norm_params['const'])
    for idx, value in enumerate(contact_matrix.data):
        i, j = contact_matrix.row[idx], contact_matrix.col[idx]
        norm_value = value / np.sqrt(mu[i] * mu[j])
        norm_data.append(norm_value)
    norm_contact_matrix = coo_matrix((norm_data, (contact_matrix.row, contact_matrix.col)), shape=contact_matrix.shape)
    # Calculate Jaccard Index scores and AUDRC for raw and normalized data
    jaccard_scores_raw = [(i, j, score, (i, j) in true_positives) for (i, j, score) in jaccard_index(contact_matrix)]
    audrc_jaccard_raw, tpr_jaccard_raw, discard_proportion_jaccard_raw = calculate_audrc(jaccard_scores_raw)
    jaccard_scores_norm = [(i, j, score, (i, j) in true_positives) for (i, j, score) in jaccard_index(norm_contact_matrix)]
    audrc_jaccard_norm, tpr_jaccard_norm, discard_proportion_jaccard_norm = calculate_audrc(jaccard_scores_norm)
    # Calculate Leicht-Holme-Newman (LHN) Index scores and AUDRC for raw and normalized data
    lhn_scores_raw = [(i, j, score, (i, j) in true_positives) for (i, j, score) in lhn_index(contact_matrix)]
    audrc_lhn_raw, tpr_lhn_raw, discard_proportion_lhn_raw = calculate_audrc(lhn_scores_raw)
    lhn_scores_norm = [(i, j, score, (i, j) in true_positives) for (i, j, score) in lhn_index(norm_contact_matrix)]
    audrc_lhn_norm, tpr_lhn_norm, discard_proportion_lhn_norm = calculate_audrc(lhn_scores_norm)
    # Plot results
    plt.figure(figsize=(8, 6))
    plot_discard_retain_curve(discard_proportion_jaccard_raw, tpr_jaccard_raw, audrc_jaccard_raw, 'Jaccard Index Raw', 'blue')
    plot_discard_retain_curve(discard_proportion_jaccard_norm, tpr_jaccard_norm, audrc_jaccard_norm, 'Jaccard Index Norm', 'orange')
    plot_discard_retain_curve(discard_proportion_lhn_raw, tpr_lhn_raw, audrc_lhn_raw, 'LHN Index Raw', 'green')
    plot_discard_retain_curve(discard_proportion_lhn_norm, tpr_lhn_norm, audrc_lhn_norm, 'LHN Index Norm', 'red')
    plt.xlabel('Proportion of discarded spurious contacts')
    plt.ylabel('Proportion of retained intra-species contacts')
    plt.title('Discard-Retain Curve Including Self-Loops')
    plt.legend()
    plt.grid(True)
    plt.savefig('audrc_jaccard_lhn_incl_self_loops.png')
    plt.show()

if __name__ == "__main__":
    main()
