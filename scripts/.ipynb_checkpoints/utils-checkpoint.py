import pandas as pd

def leaderboard_augment(df, cols_to_transform, base_model_dict):
    for col in cols_to_transform:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except KeyError as e:
            print(f"Warning: Column '{e}' not found in DataFrame.")
    
    df = df.dropna(subset=cols_to_transform)

    df['Tokens'] = 0  # Initialize the 'Tokens' column with 0
    df['Tokens'] = df['Tokens'].astype(float)

    df['base_model'] = ''  # Initialize the 'base_model' column with empty strings
    
    # Apply the rules to impute the values based on the 'fullname' column
    for index, row in df.iterrows():
        fullname = str(row['fullname']).lower()  # Convert to lowercase for case-insensitive comparison
        for key, value in base_model_dict.items():
            if key in fullname:
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
    fig_ica, axes_ica = plt.subplots(num_rows, num_cols, figsize=(15 * num_rows, 8))

    # Flatten the axes array for easier iteration
    axes_ica = axes_ica.flatten()
    
    fig_pca, axes_pca = plt.subplots(num_rows, num_cols, figsize=(15 * num_rows, 8))
    
    # Flatten the axes array for easier iteration
    axes_pca = axes_pca.flatten()
    
    unmixing_list = []

    pca_index = 0
    
    # Perform ICA for each frequent compute value
    for model, compute in frequent_computes.items():
        # Extract data for the current compute value
        subset_df = df_with_compute[df_with_compute['Pretraining compute'] == compute]
        data = subset_df[cols_to_use].values  
    
        # Apply PCA
        pca = PCA(n_components=n_components) # Initialize PCA with desired number of components
        pca_result = pca.fit_transform(data)
    
        # Get eigenvalues
        eigenvalues = pca.explained_variance_
    
    
        ax = axes_pca[pca_index]  # Subplot for explained variance
        # subplot for the mixing matrix
        mixing_matrix = pca.components_.T
        # mixing_matrix = mixing_matrix / np.linalg.norm(mixing_matrix, axis=1, keepdims=True)
        ax.imshow(mixing_matrix, cmap='viridis', aspect='auto')
        ax.set_xticks([])
        ax.set_yticks(np.arange(mixing_matrix.shape[0]), cols_to_use)
        ax.set_ylabel('Original Feature')
        ax.set_title(f'PCA Mixing Matrix for base model {frequent_base_models[pca_index]}')
        fig_pca.colorbar(ax.images[0], ax=ax, label='Mixing Matrix Value')
    
        for j in range(mixing_matrix.shape[0]):
            for k in range(mixing_matrix.shape[1]):
                text = ax.text(k, j, f"{mixing_matrix[j, k]:.2f}",
                              ha="center", va="center", color="w")
    
        ax = axes_pca[pca_index + len(frequent_computes)]  # Subplot for eigenvalues
        ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
        ax.set_title(f'Leading Eigenvalues for base model {frequent_base_models[pca_index]}')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Eigenvalue')
        ax.grid(True)
    
        
    
        # Apply ICA
        ica = FastICA(n_components=n_components, random_state=0) # Use min of rows and cols to avoid errors
        ica_result = ica.fit_transform(data)
    
        # Visualize the mixing matrix
        mixing_matrix = ica.mixing_
        # mixing_matrix = mixing_matrix / np.linalg.norm(mixing_matrix, axis=1, keepdims=True)
    
        inv_mixing_matrix = np.linalg.pinv(mixing_matrix)
    
        unmixing_list.append(inv_mixing_matrix)
    
        # Visualize the mixing and unmixing matrices
        ax1 = axes_ica[pca_index]  # Subplot for mixing matrix
        ax2 = axes_ica[pca_index + len(frequent_computes)]  # Subplot for unmixing matrix
        
        #Plot mixing matrix
        im1 = ax1.imshow(mixing_matrix, cmap='viridis', aspect='auto')
        fig_ica.colorbar(im1, ax=ax1, label='Mixing Matrix Value')
        
        for j in range(mixing_matrix.shape[0]):
            for k in range(mixing_matrix.shape[1]):
                text = ax1.text(k, j, f"{mixing_matrix[j, k]:.2f}",
                              ha="center", va="center", color="w")
                
        ax1.set_title(f'ICA Mixing Matrix for base model {frequent_base_models[pca_index]}')
        ax1.set_xticks([])
        ax1.set_yticks(np.arange(mixing_matrix.shape[0]), cols_to_use)
        ax1.set_xlabel('ICA Component')
        ax1.set_ylabel('Original Feature')
    
        # Plot unmixing matrix
        im2 = ax2.imshow(inv_mixing_matrix, cmap='viridis', aspect='auto')
        fig_ica.colorbar(im2, ax=ax2, label='Unmixing Matrix Value')
        
        for j in range(inv_mixing_matrix.shape[0]):
            for k in range(inv_mixing_matrix.shape[1]):
                text = ax2.text(k, j, f"{inv_mixing_matrix[j, k]:.2f}",
                              ha="center", va="center", color="w")
                
        ax2.set_title(f'ICA Unmixing Matrix for base model {frequent_base_models[pca_index]}')
        ax2.set_xticks(np.arange(inv_mixing_matrix.shape[1]), cols_to_use)
        ax2.set_yticks([])
        ax2.set_xlabel('Original Feature')
        ax2.set_ylabel('ICA Component')

        pca_index += 1
    
    fig_pca.tight_layout()
    fig_ica.tight_layout()

    return fig_pca, fig_ica