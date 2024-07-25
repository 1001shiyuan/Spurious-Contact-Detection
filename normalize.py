import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, coo_matrix, save_npz
import statsmodels.api as sm

# Load data
script_dir = '/Users/shaynewang/Documents/0_hi_c'
contact_matrix_path = os.path.join(script_dir, 'Raw_contact_matrix.npz')
metadata_path = os.path.join(script_dir, 'contig_info_yeast.csv')
contact_matrix = load_npz(contact_matrix_path).tocoo()
metadata = pd.read_csv(metadata_path)

# NormCC normalization
def normcc(df):
    df['log_site'] = np.log(df['Number of restriction sites'])
    df['log_length'] = np.log(df['Contig length'])
    df['log_covcc'] = np.log(df['Within-contig Hi-C contacts '])  # within-contig Hi-C contacts
    df['signal'] = df['Across-contig Hi-C contacts']  # across-contig Hi-C contacts
    
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

# Main function to normalize contacts
def main():
    # Load data
    contact_matrix = load_npz(contact_matrix_path).tocoo()
    metadata = pd.read_csv(metadata_path)
    
    # Perform NormCC normalization
    norm_params = normcc(metadata)
    
    # Normalize the contact matrix
    normalized_contacts = normalize_contact_matrix(contact_matrix, metadata, norm_params)
    
    return normalized_contacts

# Execute the main function and store the result
normalized_contacts = main()

# Saving the normalized contact matrix to NPZ file (sparse format)
save_path_npz = os.path.join(script_dir, 'normalized_contact_matrix.npz')
save_npz(save_path_npz, normalized_contacts)
