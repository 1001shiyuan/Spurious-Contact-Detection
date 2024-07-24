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

# Function to calculate Katz Index
def katz_index(contact_matrix, beta):
    identity_matrix = np.eye(contact_matrix.shape[0])
    adjacency_matrix = contact_matrix.toarray()
    katz_matrix = np.linalg.inv(identity_matrix - beta * adjacency_matrix) - identity_matrix
    return katz_matrix

# Function to calculate scores for each contact pair
def calculate_scores(contact_matrix, katz_matrix, true_positives):
    scores = []
    for i, j, value in zip(contact_matrix.row, contact_matrix.col, contact_matrix.data):
        score = katz_matrix[i, j]
        scores.append((i, j, score, (i, j) in true_positives))
    return scores

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

# Function to exclude self-loops from a contact matrix
def exclude_self_loops(matrix):
    matrix = matrix.tocoo()
    mask = matrix.row != matrix.col
    return coo_matrix((matrix.data[mask], (matrix.row[mask], matrix.col[mask])), shape=matrix.shape)

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
def main(beta=0.01):
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
    # Exclude self-loops for Katz Index scores and AUDRC calculation
    filtered_contact_matrix = exclude_self_loops(contact_matrix)
    filtered_norm_contact_matrix = exclude_self_loops(norm_contact_matrix)
    # Calculate Katz Index scores and AUDRC for raw and normalized data
    katz_matrix_raw = katz_index(filtered_contact_matrix, beta)
    katz_scores_raw = calculate_scores(filtered_contact_matrix, katz_matrix_raw, true_positives)
    audrc_katz_raw, tpr_katz_raw, discard_proportion_katz_raw = calculate_audrc(katz_scores_raw)
    katz_matrix_norm = katz_index(filtered_norm_contact_matrix, beta)
    katz_scores_norm = calculate_scores(filtered_norm_contact_matrix, katz_matrix_norm, true_positives)
    audrc_katz_norm, tpr_katz_norm, discard_proportion_katz_norm = calculate_audrc(katz_scores_norm)
    # Plot results
    plt.figure(figsize=(8, 6))
    plot_discard_retain_curve(discard_proportion_katz_raw, tpr_katz_raw, audrc_katz_raw, 'Katz Index Raw', 'blue')
    plot_discard_retain_curve(discard_proportion_katz_norm, tpr_katz_norm, audrc_katz_norm, 'Katz Index Norm', 'orange')
    plt.xlabel('Proportion of discarded spurious contacts')
    plt.ylabel('Proportion of retained intra-species contacts')
    plt.title('Discard-Retain Curve Excluding Self-Loops')
    plt.legend()
    plt.grid(True)
    plt.savefig('audrc_katz_excl_self_loops.png')
    plt.show()

if __name__ == "__main__":
    main(beta=0.01)
