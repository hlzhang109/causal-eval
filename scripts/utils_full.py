import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
from matplotlib.colors import FuncNorm
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.impute import KNNImputer
from scipy.optimize import minimize
import seaborn as sns
import copy
import itertools
import os

def left_inv(A):
    # Compute the left inverse of A. For A (m x n) with full row–rank,
    # A^+ = A^T (A A^T)^{-1} is the unique left inverse.
    A_inv = A.T @ np.linalg.inv(A @ A.T)
    return A_inv

def right_inv(A):
    A_inv = np.linalg.inv(A.T @ A) @ A.T
    return A_inv

def leaderboard_augment(df, cols_to_transform, base_model_dict):
    for col in cols_to_transform:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except KeyError as e:
            print(f"Warning: Column '{e}' not found in DataFrame.")
    
    df = df.dropna(subset=cols_to_transform)

    df['fullname'] = df['fullname'].str.lower()
    df['Tokens'] = 0  # Initialize the 'Tokens' column with 0
    df['Tokens'] = df['Tokens'].astype(float)

    df['base_model'] = ''  # Initialize the 'base_model' column with empty strings
    
    # Apply the rules to impute the values based on the 'fullname' column
    for index, row in df.iterrows():
        row['fullname'] = str(row['fullname']).lower()  # Convert to lowercase for case-insensitive comparison
        for key, value in base_model_dict.items():
            if key in row['fullname']:
                df.loc[index, 'base_model'] = key
                df.loc[index, 'Tokens'] = value

    # Create the 'pretrain' covariate
    df['pretrain'] = df['Type'].astype(str).str.contains('pretrain', case=False).astype(int)
    
    df['#Params (B)'] = pd.to_numeric(df['#Params (B)'], errors='coerce')
    df['Tokens'] = pd.to_numeric(df['Tokens'], errors='coerce')

    df_filtered = df[(df['Tokens'] != 0) & (df['#Params (B)'] != 0)]

    df_filtered['Pretraining compute'] = np.log(df_filtered['#Params (B)'] * df_filtered['Tokens'])
    
    
    return df, df_filtered

def component_analysis(df_with_compute, cols_to_use, frequent_base_models_dict, num_cols = 5, num_rows = 2, n_components = 3):

    # Dictionary to store the PCA basis (principal directions) for each model.
    pca_components = {}

    # Dictionary to store the orthonormalized ICA basis for each model.
    ica_components = {}
    
    fig_ica, axes_ica = plt.subplots(num_rows, num_cols, figsize=(15 * num_rows, 6))

    # Flatten the axes array for easier iteration
    axes_ica = axes_ica.flatten()
    
    fig_pca, axes_pca = plt.subplots(num_rows, num_cols, figsize=(15 * num_rows, 6))
    
    # Flatten the axes array for easier iteration
    axes_pca = axes_pca.flatten()


    #This contains the transpose of the individual ICA mixing matrices
    unmixing_list = []


    pca_index = 0
    
    # Perform ICA for each frequent compute value
    for model, compute in frequent_base_models_dict.items():
        # Extract data for the current compute value
        subset_df = df_with_compute[df_with_compute['Identified base model'].str.contains(model, case = False, na = False)]
        data = subset_df[cols_to_use].values  
    
        # Apply PCA
        pca = PCA(n_components=n_components) # Initialize PCA with desired number of components
        pca_result = pca.fit_transform(data)
    
        # Get eigenvalues
        eigenvalues = pca.explained_variance_

        # pca.components_ is shape (n_components, n_features).
        # Transpose so that columns are the eigenvectors (basis for the subspace).
        pca_components[model] = pca.components_.T # shape: (n_features, n_components)
    
    
        ax = axes_pca[pca_index]  # Subplot for explained variance
        # subplot for the mixing matrix
        mixing_matrix = pca.components_.T
        
        abs_max = np.abs(mixing_matrix).max()
        norm = FuncNorm((np.abs, lambda x: x), vmin=0, vmax=abs_max)
        ax.imshow(mixing_matrix, cmap='Greens', norm = norm, aspect = 'auto')
        ax.set_xticks([])
        ax.set_yticks(np.arange(mixing_matrix.shape[0]), cols_to_use)
        ax.set_ylabel('Original Feature')
        ax.set_title(f'PCA Mixing Matrix for \n base model {model}', fontsize=12)
        fig_pca.colorbar(ax.images[0], ax=ax)
    
        for j in range(mixing_matrix.shape[0]):
            for k in range(mixing_matrix.shape[1]):
                text = ax.text(k, j, f"{mixing_matrix[j, k]:.2f}",
                              ha="center", va="center", color="w", fontsize = 10)
    
        ax = axes_pca[pca_index + len(frequent_base_models_dict)]  # Subplot for eigenvalues
        ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
        ax.set_title(f'Leading Eigenvalues for \n base model {model}', fontsize=12)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Eigenvalue')
        ax.grid(True)
    
        
    
        # Apply ICA
        ica = FastICA(n_components=n_components, random_state=0, max_iter=1000) # Use min of rows and cols to avoid errors
        ica_result = ica.fit_transform(data)
    
        # Visualize the mixing matrix
        mixing_matrix = ica.mixing_

        # 1) Compute standard deviations of each component:
        stds = ica_result.std(axis=0, ddof=1)  # ddof=1 for sample std
        
        # 2) Rescale ICA result so each component has variance 1:
        ica_result_unit_var = ica_result / stds  # divides each column j by stds[j]
        
        # 3) Update the mixing matrix accordingly:
        mixing_matrix = mixing_matrix * stds[np.newaxis, :]

        # To compare subspaces, orthonormalize the mixing matrix using QR decomposition.
        Q, _ = np.linalg.qr(mixing_matrix)
        ica_components[model] = Q  # Q has shape (n_features, n_components)
    
        inv_mixing_matrix = right_inv(mixing_matrix)
    
        unmixing_list.append(inv_mixing_matrix)
    
        # Visualize the mixing and unmixing matrices
        ax1 = axes_ica[pca_index]  # Subplot for mixing matrix
        ax2 = axes_ica[pca_index + len(frequent_base_models_dict)]  # Subplot for unmixing matrix

        #Plot mixing matrix
        abs_max = np.abs(mixing_matrix).max()
        norm = FuncNorm((np.abs, lambda x: x), vmin=0, vmax=abs_max)
        im1 = ax1.imshow(mixing_matrix, cmap='Greens', norm = norm, aspect = 'auto')
        fig_ica.colorbar(im1, ax=ax1)
        
        for j in range(mixing_matrix.shape[0]):
            for k in range(mixing_matrix.shape[1]):
                text = ax1.text(k, j, f"{mixing_matrix[j, k]:.2f}",
                              ha="center", va="center", color="w", fontsize = 10)
                
        ax1.set_title(f'ICA Mixing Matrix for \n base model {model}', fontsize=12)
        ax1.set_xticks([])
        ax1.set_yticks(np.arange(mixing_matrix.shape[0]), cols_to_use)
        ax1.set_xlabel('ICA Component')
        ax1.set_ylabel('Original Feature')
        ax1.grid(False)
    
        # Plot unmixing matrix
        abs_max = np.abs(inv_mixing_matrix).max()
        norm = FuncNorm((np.abs, lambda x: x), vmin=0, vmax=abs_max)
        im2 = ax2.imshow(inv_mixing_matrix, cmap='Greens', norm = norm, aspect = 'auto')
        fig_ica.colorbar(im2, ax=ax2, label='Unmixing Matrix Value')
        
        for j in range(inv_mixing_matrix.shape[0]):
            for k in range(inv_mixing_matrix.shape[1]):
                text = ax2.text(k, j, f"{inv_mixing_matrix[j, k]:.2f}",
                              ha="center", va="center", color="w")
                
        ax2.set_title(f'ICA Unmixing Matrix for \n base model {model}', fontsize=12)
        ax2.set_xticks(np.arange(inv_mixing_matrix.shape[1]))
        ax2.set_xticklabels(cols_to_use, rotation=45)
        ax2.set_yticks([])
        ax2.set_xlabel('Original Feature')
        ax2.set_ylabel('ICA Component')
        ax2.grid(False)

        pca_index += 1
    
    fig_pca.tight_layout()
    fig_ica.tight_layout()

    '''
    Now we compute a cosine-similarity matrix between different PCA subspaces.
    '''

    fig_pca_sim, ax_pca_sim = plt.subplots(figsize=(len(frequent_base_models_dict)+2,len(frequent_base_models_dict)))
    
    # List of model names
    models = list(pca_components.keys())
    num_models = len(models)
    
    # Initialize a similarity matrix.
    similarity_matrix = np.zeros((num_models, num_models))
    
    # Compute cosine distance between subspaces for each pair of models.
    for i in range(num_models):
        for j in range(num_models):
            U = pca_components[models[i]]  # Basis for model i
            V = pca_components[models[j]]  # Basis for model j
            
            # Compute the matrix of dot products: shape (n_components, n_components)
            M = np.dot(U.T, V)
            # Compute singular values; these are the cosines of the principal angles.
            singular_vals = np.linalg.svd(M, compute_uv=False)
            # Average cosine similarity across the principal angles.
            avg_cos_sim = np.mean(singular_vals)
            # Cosine distance defined as 1 minus the average cosine similarity.
            cosine_distance = 1 - avg_cos_sim
            similarity_matrix[i, j] = cosine_distance

    # Plot the similarity matrix as a heatmap.
    im_pca_sim = ax_pca_sim.imshow(similarity_matrix, cmap='Blues', aspect = 'auto')

    for j in range(similarity_matrix.shape[0]):
            for k in range(similarity_matrix.shape[1]):
                text = ax_pca_sim.text(k, j, f"{similarity_matrix[j, k]:.2f}",
                              ha="center", va="center", color="w", fontsize = 15)

    ax_pca_sim.set_xticks(np.arange(num_models))
    ax_pca_sim.set_yticks(np.arange(num_models))
    ax_pca_sim.set_xticklabels(models, fontsize=20, rotation=30)
    ax_pca_sim.set_yticklabels(models, fontsize=20)
    fig_pca_sim.colorbar(im_pca_sim, ax=ax_pca_sim)
    # ax_pca_sim.set_title("Cosine Distance between PCA Subspaces")
    
    fig_pca_sim.tight_layout()


    '''
    Now we compute a cosine-similarity matrix between different ICA subspaces.
    '''

    fig_ica_sim, ax_ica_sim = plt.subplots(figsize=(len(frequent_base_models_dict)+2,len(frequent_base_models_dict)))
    
    # List of models and initialize the similarity matrix.
    models = list(ica_components.keys())
    num_models = len(models)
    similarity_matrix_ica = np.zeros((num_models, num_models))
    
    # Compute the cosine distance between ICA subspaces for each pair.
    for i in range(num_models):
        for j in range(num_models):
            U = ica_components[models[i]]  # Orthonormal basis for model i
            V = ica_components[models[j]]  # Orthonormal basis for model j
            
            # Compute the dot product matrix (shape: [n_components, n_components])
            M = np.dot(U.T, V)
            # The singular values of M are the cosines of the principal angles.
            singular_vals = np.linalg.svd(M, compute_uv=False)
            avg_cos_sim = np.mean(singular_vals)
            # Define cosine distance as 1 minus the average cosine similarity.
            cosine_distance = 1 - avg_cos_sim
            similarity_matrix_ica[i, j] = cosine_distance

    # Plot the similarity matrix as a heatmap.
    im_ica_sim = ax_ica_sim.imshow(similarity_matrix_ica, cmap='Blues', aspect = 'auto')
    fig_ica_sim.colorbar(im_ica_sim, ax=ax_ica_sim)

    for j in range(similarity_matrix_ica.shape[0]):
            for k in range(similarity_matrix_ica.shape[1]):
                text = ax_ica_sim.text(k, j, f"{similarity_matrix_ica[j, k]:.2f}",
                              ha="center", va="center", color="w", fontsize = 15)
    
    ax_ica_sim.set_xticks(np.arange(num_models))
    ax_ica_sim.set_yticks(np.arange(num_models))
    ax_ica_sim.set_xticklabels(models, fontsize=15, rotation=30)
    ax_ica_sim.set_yticklabels(models, fontsize=15)
    # ax_ica_sim.set_title("Cosine Distance between ica Subspaces")
    
    fig_ica_sim.tight_layout()

    for ax in [ax_pca_sim, ax_ica_sim]:
        ax.grid(False)

    return fig_pca, fig_ica, fig_pca_sim, fig_ica_sim, unmixing_list


def choose_subset_indices(df, model_names):
    subset_indices = []
    for model_name in model_names:
      subset_indices.extend(df[df['fullname'].str.lower().str.contains(model_name)].index)
    return subset_indices

def select_data(df, cols, model_names):
    subset_indices = choose_subset_indices(df, model_names)
    return df.loc[subset_indices, cols].values



def plot_combined_pca(X, X_subset, cols_to_transform_new):
    # Set a clean, professional style with larger default font size
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14  # Increased base font size
    
    # Define a colorblind-friendly color palette
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    
    # 1. All data
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(X.T)
    # Normalize rows of principal_components
    col_norms = np.linalg.norm(principal_components[:, :3], axis=0, keepdims=True)
    normalized_principal_components = principal_components[:, :3] / col_norms
    
    # 2. Data subset
    pca = PCA(n_components=3)
    principal_components_subset = pca.fit_transform(X_subset.T)
    # Normalize rows of principal_components_subset
    col_norms_subset = np.linalg.norm(principal_components_subset[:, :3], axis=0, keepdims=True)
    normalized_principal_components_subset = principal_components_subset[:, :3] / col_norms_subset
    
    # Create figure with improved sizing and spacing
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.3)  # Add more space between subplots
    
    # Improved colormap with better divergence
    cmap = plt.cm.RdBu_r  # Reversed RdBu for better distinction
    
    # Plot for All Data
    im1 = axes[0].imshow(normalized_principal_components.T, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # Add annotations with color-adaptive text for all data plot - LARGER NUMBERS
    for i in range(normalized_principal_components.T.shape[0]):
        for j in range(normalized_principal_components.T.shape[1]):
            val = normalized_principal_components.T[i, j]
            # Choose text color based on background intensity for better visibility
            color = "black" if abs(val) < 0.6 else "white"
            text = axes[0].text(j, i, f"{val:.2f}",
                         ha="center", va="center", color=color, fontsize=18, fontweight='bold')
    
    # Improved title and labels
    axes[0].set_title('Principal Components (All Data)', fontsize=18, fontweight='bold', pad=15)
    
    # Format x and y labels - NO ROTATION
    axes[0].set_xticks(np.arange(len(cols_to_transform_new)))
    axes[0].set_xticklabels(cols_to_transform_new, fontsize=14)  # Removed rotation
    axes[0].set_yticks(np.arange(3))
    axes[0].set_yticklabels(['PC-1', 'PC-2', 'PC-3'], fontsize=16, fontweight='bold')
    
    # Add grid for better readability
    axes[0].grid(False)  # Remove default grid for heatmap
    
    # Improved colorbar
    cbar1 = fig.colorbar(im1, ax=axes[0], pad=0.01)
    cbar1.ax.tick_params(labelsize=14)
    cbar1.set_label('Normalized Component Value', fontsize=16, fontweight='bold')
    
    # Plot for Subset Data
    im2 = axes[1].imshow(normalized_principal_components_subset.T, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # Add annotations with color-adaptive text for subset plot - LARGER NUMBERS
    for i in range(normalized_principal_components_subset.T.shape[0]):
        for j in range(normalized_principal_components_subset.T.shape[1]):
            val = normalized_principal_components_subset.T[i, j]
            # Choose text color based on background intensity
            color = "black" if abs(val) < 0.6 else "white"
            text = axes[1].text(j, i, f"{val:.2f}",
                         ha="center", va="center", color=color, fontsize=18, fontweight='bold')
    
    # Improved title and labels for subset
    axes[1].set_title('Principal Components (Selected Data)', fontsize=18, fontweight='bold', pad=15)
    
    # Format x and y labels - NO ROTATION
    axes[1].set_xticks(np.arange(len(cols_to_transform_new)))
    axes[1].set_xticklabels(cols_to_transform_new, fontsize=14)  # Removed rotation
    axes[1].set_yticks(np.arange(3))
    axes[1].set_yticklabels(['PC-1', 'PC-2', 'PC-3'], fontsize=16, fontweight='bold')
    
    # Add grid for better readability
    axes[1].grid(False)  # Remove default grid for heatmap
    
    # Improved colorbar
    cbar2 = fig.colorbar(im2, ax=axes[1], pad=0.01)
    cbar2.ax.tick_params(labelsize=14)
    cbar2.set_label('Normalized Component Value', fontsize=16, fontweight='bold')
    
    # Better layout adjustment
    fig.tight_layout()
    
    # --- Singular value plotting with enhanced styling ---
    # Calculate SVD
    U_all, S_all, V_all = np.linalg.svd(X)
    U_subset, S_subset, V_subset = np.linalg.svd(X_subset)
    
    # Create singular values plot with improved styling
    fig_sv = plt.figure(figsize=(10, 8))
    ax_sv = fig_sv.add_subplot(111)
    
    # Plot singular values with better styling
    ax_sv.plot(S_all, marker='o', markersize=10, linewidth=2.5, color=colors[0],
               markeredgecolor='black', markeredgewidth=1.0, label='All Data')
    ax_sv.plot(S_subset, marker='s', markersize=10, linewidth=2.5, color=colors[1],
               markeredgecolor='black', markeredgewidth=1.0, label='Selected Data')
    
    # Add a logarithmic plot inset for better visualization of smaller values
    # Create inset axes for log scale plot
    inset_ax = fig_sv.add_axes([0.6, 0.2, 0.3, 0.3])  # [left, bottom, width, height]
    inset_ax.semilogy(S_all, marker='o', markersize=6, linewidth=1.5, color=colors[0],
                    markeredgecolor='black', markeredgewidth=0.5, label='All Data')
    inset_ax.semilogy(S_subset, marker='s', markersize=6, linewidth=1.5, color=colors[1],
                    markeredgecolor='black', markeredgewidth=0.5, label='Selected Data')
    inset_ax.set_title('Log Scale', fontsize=12)
    inset_ax.grid(True, linestyle='--', alpha=0.6)
    
    # Improve main plot styling
    ax_sv.set_xlabel('Singular Value Index', fontsize=18, fontweight='bold')
    ax_sv.set_ylabel('Singular Value', fontsize=18, fontweight='bold')
    
    # Add grid for better readability
    ax_sv.grid(True, linestyle='--', alpha=0.4)
    
    # Set axis limits and ticks with larger font
    ax_sv.set_xlim(-0.5, len(S_all) - 0.5)
    ax_sv.set_xticks(np.arange(0, len(S_all), 1))
    ax_sv.set_xticklabels(np.arange(1, len(S_all) + 1), fontsize=14)
    ax_sv.tick_params(axis='y', labelsize=14)
    
    # Improved legend
    ax_sv.legend(fontsize=14, frameon=True, fancybox=True, framealpha=0.9, 
                edgecolor='black', title="Data Source", title_fontsize=16)
    
    # Add title
    ax_sv.set_title('Singular Values Comparison', fontsize=20, fontweight='bold', pad=15)
    
    # Annotate significant singular values
    for i in [0, 1, 2]:  # Annotate first three singular values
        ax_sv.annotate(f'{S_all[i]:.2f}', 
                      xy=(i, S_all[i]), 
                      xytext=(10, 10),
                      textcoords='offset points',
                      fontsize=12,
                      fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color='black'))
    
    # Better layout adjustment
    fig_sv.tight_layout()
    
    return fig, fig_sv


def swap_rows(matrices, row_indices):
    """
    Swaps rows in a list of matrices based on provided row indices.

    Args:
        matrices: A list of NumPy matrices.
        row_indices: A list of row indices corresponding to each matrix.
    
    Returns:
        A list of matrices with swapped rows, or None if input is invalid.
    """

    if not isinstance(matrices, list) or not all(isinstance(matrix, np.ndarray) for matrix in matrices):
        print("Error: 'matrices' must be a list of NumPy arrays.")
        return None

    if not isinstance(row_indices, list) or len(matrices) != len(row_indices):
        print("Error: 'row_indices' must be a list of the same length as 'matrices'.")
        return None

    modified_matrices = []
    for k in range(len(matrices)):
        matrix = matrices[k]
        i1 = row_indices[0]  # Always swap with the first row index provided
        ik = row_indices[k]
      
        # Check if the indices are valid for the current matrix.
        rows, cols = matrix.shape
        if not (0 <= i1 < rows and 0 <= ik < rows):
            print(f"Error: Invalid row indices for matrix {k+1}: {i1}, {ik}")
            return None

        # Create a copy to avoid modifying the original matrices
        new_matrix = matrix.copy()
        
        # Swap the rows
        new_matrix[[i1, ik]] = new_matrix[[ik, i1]]

        modified_matrices.append(new_matrix)

    return modified_matrices

def intersect(W, d):
    m = len(W)
    M = np.zeros((d,d), dtype=float)
    for i in range(m):
        P = np.identity(d) - W[i].T @ np.linalg.inv(W[i] @ W[i].T) @ W[i]
        M += P.T @ P
    U_, S_, Vh_ = np.linalg.svd(M)
    return Vh_[-1], S_[-1]

def process_matrices(matrices):
    m = len(matrices)
    n = matrices[0].shape[0]  # Assuming all matrices have the same number of rows
    visited_rows = []
    min_eigenvalue_ratio_list = []

    while len(visited_rows) < n:
        min_eigenvalue_ratio = float('inf')
        best_i0 = -1
        best_row_indices = []

        # Create a list of allowed indices (all indices in range(d) that are not in S)
        allowed = [i for i in range(n) if i not in visited_rows]
            
        for indices in itertools.product(allowed, repeat=m):
            row_indices = list(indices)
            # Residual calculation
            residual_matrix = []
            for k in range(m):
                ik = row_indices[k]
                
                visited_rows_k = list(visited_rows) # Convert visited_rows to a list
                
                # If there are no visited rows, skip the projection
                if len(visited_rows_k) == 0:
                  residual_matrix.append(matrices[k][ik])  # Use original row as residual
                  continue

                # Project ik-th row onto visited rows in matrix k
                visited_data_k = matrices[k][visited_rows_k]
                projection= np.zeros(visited_data_k.shape[1])
                for r in range(len(visited_rows_k)):
                    projection += np.dot(matrices[k][ik], visited_data_k[r]) / np.linalg.norm(visited_data_k[r])**2 * visited_data_k[r]
                
                
                # Calculate residual
                residual = matrices[k][ik] - projection 

                residual_matrix.append(residual)

            residual_matrix = np.array(residual_matrix) # Convert to NumPy array
            residual_matrix /= np.linalg.norm(residual_matrix)
            
            # Calculate singular values
            U, singular_values, Vt = np.linalg.svd(residual_matrix)
            
            # Sort singular values in descending order
            sorted_singular_values = sorted(singular_values, reverse=True)
            
            # Get the second largest singular value
            # second_largest_singular_value = sorted_singular_values[1] 
            singular_value_ratio = np.sum(sorted_singular_values[1:]) / np.sum(sorted_singular_values)
            
            if singular_value_ratio < min_eigenvalue_ratio:
                min_eigenvalue_ratio = singular_value_ratio
                best_i0 = indices[0]
                best_row_indices = row_indices

        
        # Swap rows
        matrices = swap_rows(matrices, best_row_indices)
        min_eigenvalue_ratio_list.append(min_eigenvalue_ratio)
        visited_rows.append(best_i0)

    num_matrices = len(matrices)
    for i in range(num_matrices):
        # Access the matrix using its index
        original_matrix = matrices[i]
        # Create the modified matrix
        modified_matrix = original_matrix[visited_rows[::-1], :]
        # Assign the modified matrix back to the list at the correct index
        matrices[i] = modified_matrix


    return visited_rows, min_eigenvalue_ratio_list, matrices



def compute_residual_for_row(A, H, i):
    """
    Given an m×n matrix A and a partially computed H (m×n),
    compute the residual for row i after projecting A[i,:] onto
    the subspace spanned by H[i+1,:], ..., H[m-1,:].

    That is, compute:
        r = A[i, :] - sum_{j=i+1}^{m-1} (A[i, :] · H[j, :]) * H[j, :].
    """
    r = A[i, :].copy()
    m = H.shape[0]
    for j in range(i + 1, m):
        # H[j, :] is assumed already computed.
        r -= np.dot(A[i, :], H[j, :]) * H[j, :]
    return r

def recover_H(A_list, verbose=False):
    """
    Recover an m×n matrix H (with orthonormal rows) from a list of A_k (each m×n)
    so that there exist (approximately) upper–triangular B_k satisfying A_k = B_k * H.
    
    The algorithm proceeds from the last row to the first. For each row i (0-indexed),
    we compute, for each A_k, the residual r_{i,k} by projecting A_k[i,:] onto the
    orthogonal complement of the span of {H[i+1,:],...,H[m-1,:]}. We then stack the
    residuals (one per k) into a matrix R and extract its principal right singular
    vector (via SVD). That vector is taken as H[i,:].
    
    Parameters:
      A_list : list of numpy arrays, each of shape (m, n).
      verbose: if True, prints intermediate diagnostic information.
      
    Returns:
      H : numpy array of shape (m, n), whose rows are recovered from bottom to top.
    """
    K = len(A_list)
    m, n = A_list[0].shape
    H = np.zeros((m, n))  # to store the rows of H

    rank1_err = []

    # Recover rows in backward order: last row has no projection.
    for i in reversed(range(m)):
        # Compute residuals for the i-th row from all A_k.
        R = []
        for A in A_list:
            r = compute_residual_for_row(A, H, i)
            R.append(r)
        R = np.array(R)  # shape (K, n)

        # print("Rows combined:\n", R)

        # Compute SVD of the residual matrix.
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        # The principal component is given by the first row of Vt.
        principal_component = Vt[0, :]
        principal_component /= np.linalg.norm(principal_component)  # ensure unit norm

        ratio = 1 - S[0] ** 2 / np.sum(S ** 2)
        rank1_err.append(ratio)
        
        if verbose:
            print(f"Row {i}: dominant singular value ratio = {ratio:.4f}")

        # Set the i-th row of H.
        H[i, :] = principal_component

    return H, rank1_err

def vector_to_upper_triangular(x, m):
    """
    Convert a vector x of length m*(m+1)/2 into an m x m upper–triangular matrix.
    The entries of x are filled row–by–row into the upper–triangular part.
    """
    B = np.zeros((m, m))
    idx = 0
    for i in range(m):
        for j in range(i, m):
            B[i, j] = x[idx]
            idx += 1
    return B

def upper_triangular_to_vector(B):
    """
    Given an m x m matrix B, extract the upper–triangular elements (including the diagonal)
    into a vector.
    """
    m = B.shape[0]
    vec = []
    for i in range(m):
        for j in range(i, m):
            vec.append(B[i, j])
    return np.array(vec)

def cost_B(x, A, H, m):
    """
    Given a vector x parameterizing an m x m upper–triangular matrix B, compute the cost

         f(B) = || A^{-T} (B H)^T (B H) A^{-1} - I ||_F^2

    where A^{-1} is the left inverse of A (i.e. A^{+} = A^T (A A^T)^{-1})
    and A^{-T} is its transpose.

    Parameters:
      x : vector of length m*(m+1)/2 representing the free parameters of B.
      A : m x n matrix (assumed full row–rank)
      H : m x n matrix (with orthonormal rows, for example)
      m : number of rows of A (and size of B)

    Returns:
      cost : the squared Frobenius norm of the difference.
    """
    # Reconstruct B from x (B is forced to be upper triangular)
    B = vector_to_upper_triangular(x, m)
    BH = B @ H  # shape m x n

    diff = A - BH
    cost = np.linalg.norm(diff, ord='fro')**2
    return cost

def recover_Bk(A, H):
    """
    Given A (m x n) and H (m x n), find the upper–triangular matrix B (m x m)
    that minimizes
         || A^{-T} (B H)^T (B H) A^{-1} - I ||_F^2.
    
    We parameterize B by its upper–triangular entries.
    
    Parameters:
      A : numpy array of shape (m, n)
      H : numpy array of shape (m, n)
      
    Returns:
      B_opt : numpy array of shape (m, m) (upper–triangular) that minimizes the cost.
      res   : the optimization result from scipy.optimize.minimize.
    """
    m, n = A.shape
    # Initial guess: use the unconstrained least-squares solution B0 = A * H^T,
    # and then take its upper-triangular part.
    B0_full = A @ H.T
    B0 = np.triu(B0_full)
    x0 = upper_triangular_to_vector(B0)
    
    # Run the optimization.
    res = minimize(cost_B, x0, args=(A, H, m), method='BFGS')
    
    # Reconstruct the optimal B from the solution vector.
    B_opt = vector_to_upper_triangular(res.x, m)
    return B_opt

def solve_for_H(A_list, maxiter=5000, verbose=False):
    """
    Given a list of matrices A_k (each of shape m x n) satisfying A_k = B_k * H,
    where B_k is ideally upper-triangular, find an m x n matrix H (with orthonormal rows)
    that minimizes the loss function:
    
        J(H) = sum_{k=1}^K ||tril(A_k * H.T, k=-1)||_F^2
    
    subject to H * H.T = I.
    
    Parameters:
        A_list : list of numpy arrays, each with shape (m, n)
        maxiter: maximum number of iterations for the optimizer
        verbose: if True, prints optimization details
        
    Returns:
        H_opt : an m x n numpy array (with H_opt * H_opt.T = I) that minimizes the loss.
        res   : the result object returned by scipy.optimize.minimize.
    """
    # Determine dimensions from the first A_k.
    m, n = A_list[0].shape

    # Create an initial guess for H: random m x n matrix with orthonormal rows.
    np.random.seed(2025)
    H0 = np.random.randn(m, n)
    # Orthonormalize rows: perform QR on the transpose so that H0 @ H0.T = I.
    Q, _ = np.linalg.qr(H0.T)
    H0 = Q.T  # shape (m, n)

    # Define the objective function.
    def objective(x):
        H = x.reshape(m, n)
        loss = 0.0
        for A in A_list:
            # Compute B_k = A_k * H.T
            B = A @ H.T  # shape (m, m)
            # Get strictly lower triangular entries (i > j)
            tril_indices = np.tril_indices(m, k=-1)
            loss += np.sum(B[tril_indices] ** 2)
        return loss

    # Define constraints for orthonormality: H * H.T == I.
    # For each i, we need (H*H.T)[i,i] - 1 = 0.
    def constraint_diag(x):
        H = x.reshape(m, n)
        HHt = H @ H.T
        return np.array([HHt[i, i] - 1 for i in range(m)])
    
    # For each off-diagonal element (i < j), we need (H*H.T)[i,j] = 0.
    def constraint_offdiag(x):
        H = x.reshape(m, n)
        HHt = H @ H.T
        cons = []
        for i in range(m):
            for j in range(i+1, m):
                cons.append(HHt[i, j])
        return np.array(cons)
    
    constraints = [
        {'type': 'eq', 'fun': constraint_diag},
        {'type': 'eq', 'fun': constraint_offdiag}
    ]

    # Use SLSQP to solve the constrained optimization problem.
    res = minimize(
        objective,
        H0.flatten(),
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-9, 'disp': verbose}
    )

    H_opt = res.x.reshape(m, n)
    return H_opt, res


def plot_covariance_matrices(cov_noise, model_name_list, n_components=3):
    """
    Plots covariance matrices as individual heatmaps with cell values annotated.
    
    Parameters:
    -----------
    cov_noise : list of numpy arrays
        List of covariance matrices to plot
    model_name_list : list of str
        Names of models corresponding to each covariance matrix
    n_components : int
        Number of components in each matrix
        
    Returns:
    --------
    list
        List of matplotlib figure objects
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import FuncNorm
    
    noise_var_list = [f"$\\epsilon_{{{i}}}$" for i in range(1, n_components+1)]
    figures = []
    
    for i, cov_matrix in enumerate(cov_noise):
        # Create a new figure for each matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Find the absolute maximum value for consistent color scaling
        abs_max = np.abs(cov_matrix).max()
        norm = FuncNorm((np.abs, lambda x: x), vmin=0, vmax=abs_max)
        
        # Plot heatmap using imshow with improved styling
        im = ax.imshow(cov_matrix, cmap='Blues', norm=norm, aspect='equal')
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=12)
        
        # Annotate each cell with the numeric value and adaptive text color
        for (j, k), value in np.ndenumerate(cov_matrix):
            # Use black text for light cells, white text for dark cells
            cell_intensity = abs(value)/abs_max
            color = "white" if cell_intensity > 0.5 else "black"
            ax.text(k, j, f"{value:.2f}", ha="center", va="center", 
                    color=color, fontsize=13, weight='bold')
        
        # Set tick positions and labels
        ax.set_xticks(np.arange(len(noise_var_list)))
        ax.set_xticklabels(noise_var_list, fontsize=14)
        ax.set_yticks(np.arange(len(noise_var_list)))
        ax.set_yticklabels(noise_var_list, fontsize=14)
        
        # Add title with better styling
        ax.set_title(model_name_list[i], fontsize=16, pad=10, fontweight='bold')
        
        # Add thin grid lines for better readability
        ax.set_xticks(np.arange(-.5, len(noise_var_list), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(noise_var_list), 1), minor=True)
        ax.grid(False)
        
        # Adjust layout and append to figures list
        fig.tight_layout()
        figures.append(fig)
    
    return figures


def plot_3d(data, model, labels):
    fig = plt.figure(figsize=(5, 5),constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 2], data[:, 1], data[:, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel(labels[0], fontsize=20)
    ax.set_ylabel(labels[1], fontsize=20)
    ax.set_zlabel(labels[2], fontsize=20)
    return fig

def total_cov(M): # Return the sum of cov(eps_i,eps_j)
    res = 0
    a = M.shape[0]
    for i in range(a):
        for j in range(i+1,a):
            res += np.abs(M[i,j] / (M[i,i] * M[j,j])**0.5)
    return res


def plot_crl_results(M_opt_norm, cols_to_use, unmixing_matrices_sorted, base_models, benchmarks, WEIGHT_DIR): 
    # Set a clean, professional style with larger default font size
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    
    # Define a better colormap option
    cmap = 'Greens'  
    
    fig_unmixing_ica, fig_weight, unexplained_var, cov_noise = [], [], [], []
    
    for i, um in enumerate(unmixing_matrices_sorted):
        # Original matrix visualization with enhanced styling
        fig, ax = plt.subplots(figsize=(10, 5))
        abs_max = np.abs(um).max()
        norm = FuncNorm((np.abs, lambda x: x), vmin=0, vmax=abs_max)
        im = ax.imshow(um, cmap=cmap, norm=norm, aspect='auto')  
        
        # Add title with model information
        ax.set_title(f"ICA Unmixing Matrix - Model {base_models[i]}", 
                    fontsize=18, fontweight='bold', pad=15)
        
        # Improve tick labels
        ax.set_xticks(np.arange(unmixing_matrices_sorted[0].shape[1]))
        ax.set_xticklabels(cols_to_use, rotation=45, ha='right', fontsize=12)
        ax.set_yticks(np.arange(um.shape[0]))
        ax.set_yticklabels([f"Component {j+1}" for j in range(um.shape[0])], fontsize=12)
        
        # Enhanced colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.ax.set_ylabel('Coefficient Value', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
    
        # Add text annotations with adaptive color for better visibility
        for (j, k), value in np.ndenumerate(um):
            # Choose text color based on background intensity for better visibility
            color = "black" if abs(value) < abs_max * 0.7 else "white"
            ax.text(k, j, f'{value:.2f}', ha='center', va='center', 
                   color=color, fontsize=12, fontweight='bold')
        
        fig.tight_layout()
        fig_unmixing_ica.append(fig)
        
        # Weight matrix visualization with enhanced styling
        fig, ax = plt.subplots(figsize=(6, 6))
        # Inverse of near-orthogonal product visualization
        transformed_matrix = np.linalg.inv(recover_Bk(um, M_opt_norm))
        # print(f"Check recovery error for base model {base_models[i]}: ", np.linalg.norm(um - np.linalg.inv(transformed_matrix) @  M_opt_norm, ord = 'fro') / np.linalg.norm(um, ord = 'fro'))
        cov_sqt = np.linalg.inv(transformed_matrix) @ M_opt_norm @ left_inv(um)
        
        # Compute metrics for covariance analysis
        abs_cov = np.abs(cov_sqt**2)
        diag = np.diag(abs_cov)
        off_diag_sum = np.sum(abs_cov, axis=1) - diag
        ratios = off_diag_sum / np.sum(abs_cov, axis=1)
        total_sum = np.sum(ratios)
        
        cov = cov_sqt.T @ cov_sqt
        cov_noise.append(cov)
        P = np.tril(transformed_matrix, k=-1)
        unexplained_var.append(total_sum/cov.shape[0])
        
        # Enhanced visualization of transformed matrix
        abs_max = np.abs(transformed_matrix).max()
        norm = FuncNorm((np.abs, lambda x: x), vmin=0, vmax=abs_max)
        transformed_matrix_reversed = transformed_matrix[::-1, ::-1]

        save_path = os.path.join(WEIGHT_DIR, f"{base_models[i]}_weights.npy")
        np.save(save_path, transformed_matrix_reversed)

        # Plot the updated matrix
        im = ax.imshow(transformed_matrix_reversed, cmap=cmap, norm=norm, aspect='auto', origin='upper')
        
        ax.set_xticks([])
        ax.set_yticks([])

        # Add informative title
        ax.set_title(f"Weight Matrix - {base_models[i]}\nInexactness coefficient: {total_sum/cov.shape[0]:.4f}", 
                    fontsize=16, fontweight='bold', pad=15)
        
        # Add axis labels
        # ax.set_xlabel("Input Components", fontsize=14, fontweight='bold')
        # ax.set_ylabel("Output Components", fontsize=14, fontweight='bold')
        
        # Enhanced colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.ax.set_ylabel('Weight Value', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)

        # Add x-axis labels in LaTeX format
        x_labels = ['$\epsilon_1$', '$\epsilon_2$', '$\epsilon_3$']
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=14, fontweight='bold')

        # Add y-axis labels in LaTeX format
        y_labels = ['$z_1$', '$z_2$', '$z_3$']
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=14, fontweight='bold')
    
        # Add text annotations with adaptive color
        for (j, k), value in np.ndenumerate(transformed_matrix_reversed):
            color = "black" if abs(value) < abs_max * 0.7 else "white"
            ax.text(k, j, f'{value:.2f}', ha='center', va='center', 
                   color=color, fontsize=14, fontweight='bold')
        
        fig.tight_layout()
        fig_weight.append(fig)

    n_components = M_opt_norm.shape[0]


    # Clear any previous plots to prevent overlap
    plt.close('all')  # Close all existing figures

    # Enhanced M_opt_norm visualization
    fig_M_opt, ax_M_opt = plt.subplots(figsize=(10, 8))  # Increased figure size for better spacing
    plt.figure(fig_M_opt.number)  # Ensure we're working with the correct figure

    # Set up the heatmap
    abs_max = np.abs(M_opt_norm).max()
    norm = FuncNorm((np.abs, lambda x: x), vmin=0, vmax=abs_max)
    M_opt_norm_reversed = M_opt_norm.T[:, ::-1]
    im = ax_M_opt.imshow(M_opt_norm_reversed, cmap=cmap, norm=norm, aspect='auto', origin='upper')

    # Enhanced colorbar
    cbar = fig_M_opt.colorbar(im, ax=ax_M_opt, pad=0.01)
    cbar.ax.set_ylabel('Coefficient Value', fontsize=14, fontweight='bold', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=12)

    # Add axis labels
    ax_M_opt.set_xlabel("Latent Factors", fontsize=16, fontweight='bold', labelpad=10)
    ax_M_opt.set_ylabel("Benchmark Accuracies", fontsize=16, fontweight='bold', labelpad=10)

    # Add x-axis labels in LaTeX format
    x_labels = ['$z_1$', '$z_2$', '$z_3$']
    ax_M_opt.set_xticks(np.arange(len(x_labels)))
    ax_M_opt.set_xticklabels(x_labels, fontsize=14, fontweight='bold')

    # Add y-axis labels
    y_labels = benchmarks
    ax_M_opt.set_yticks(np.arange(len(y_labels)))
    ax_M_opt.set_yticklabels(y_labels, fontsize=14, fontweight='bold')

    # Add text annotations with adaptive color
    for (j, k), value in np.ndenumerate(M_opt_norm_reversed):
        color = "black" if abs(value) < abs_max * 0.7 else "white"
        ax_M_opt.text(k, j, f'{value:.2f}', ha='center', va='center', 
                  color=color, fontsize=14, fontweight='bold')

    ax_M_opt.set_xticks(np.arange(-.5, len(x_labels), 1), minor=True)
    ax_M_opt.set_yticks(np.arange(-.5, len(y_labels), 1), minor=True)
    ax_M_opt.grid(False)

    # Ensure proper layout with enough margins
    fig_M_opt.tight_layout(pad=1.5)

    # Save with high DPI to maintain quality (optional)
    # plt.savefig('benchmark_latent_factors.png', dpi=300, bbox_inches='tight')

    
    # Get covariance plot
    fig_cov = plot_covariance_matrices(cov_noise, base_models)
    
    # Add summary information
    print(f"Summary of Inexactness Coefficients:")
    for i, uv in enumerate(unexplained_var):
        print(f"  Model {base_models[i]}: {uv:.4f}")
    print(f"  Average: {np.mean(unexplained_var):.4f}")
    
    return fig_unmixing_ica, fig_weight, fig_M_opt, unexplained_var, fig_cov

def return_crl_results(M_opt_norm, cols_to_use, unmixing_matrices_sorted, base_models):

    unexplained_var = []

    # print("Check orthonormality: ", np.linalg.norm(M_opt_norm @ M_opt_norm.T - np.identity(M_opt_norm.shape[0])))
    fig_unmixing_ica, fig_weight, unexplained_var = [], [], []
    
    for i, um in enumerate(unmixing_matrices_sorted):
        
        transformed_matrix = np.linalg.inv(recover_Bk(um, M_opt_norm)) # This is (B_k * H * G)^{-1}

        cov_sqt = np.linalg.inv(transformed_matrix) @ M_opt_norm @ left_inv(um)

        # Compute the absolute values of the matrix
        abs_cov = np.abs(cov_sqt**2)
        
        # Extract the diagonal elements
        diag = np.diag(abs_cov)
        
        # Sum the absolute values of each row and subtract the diagonal elements
        off_diag_sum = np.sum(abs_cov, axis=1) - diag
        
        # Compute the ratio for each row
        ratios = off_diag_sum / np.sum(abs_cov, axis=1)
        
        # Sum all the ratios to get the final value
        total_sum = np.sum(ratios)

        cov = cov_sqt.T @ cov_sqt

        P = np.tril(transformed_matrix, k=-1)
        
        unexplained_var.append(total_sum/cov.shape[0])       
    
    return unexplained_var