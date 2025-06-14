import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import seaborn as sns

def plot_explained_variance(df_new, cols_to_transform_new, FIG_DIR = '/content/drive/MyDrive/LLM causality/Figures/', base_model = None):
    if base_model != None:
        df_new = df_new[df_new['Identified base model'] == base_model].copy()
    
    # Extract the columns for PCA
    df_pca = df_new[cols_to_transform_new]
    imputer = KNNImputer(n_neighbors=5)  # You can adjust n_neighbors
    df_imputed = pd.DataFrame(imputer.fit_transform(df_pca), columns=df_pca.columns)
    
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 18  # Increased base font size
    
    pca = PCA()
    pca_result = pca.fit_transform(df_imputed)
    plt.figure(figsize=(8, 6))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    bars = plt.bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        color='#4C72B0',
        alpha=0.7,
        width=0.7,
        edgecolor='black',
        linewidth=0.8
    )
    
    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        color='#DD8452',
        marker='o',
        markersize=6,
        markeredgecolor='black',
        markeredgewidth=1,
        linewidth=2,
        label='Cumulative Explained Variance'
    )
    
    threshold = 0.95
    components_95pct = np.where(cumulative_variance >= threshold)[0][0] + 1
    
    plt.axhline(
        y=threshold,
        color='#C44E52',
        linestyle='--',
        linewidth=2,
        label=f'{threshold*100:.0f}% Explained Variance Threshold'
    )
    
    plt.annotate(
        f'{components_95pct} components',
        xy=(components_95pct, threshold),
        xytext=(components_95pct + 1, threshold - 0.05),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=20,
        fontweight='bold'
    )
    
    plt.xlabel('Principal Component', fontsize=24, fontweight='bold')
    plt.ylabel('Explained Variance Ratio', fontsize=24, fontweight='bold')
    
    plt.xlim(0.5, len(pca.explained_variance_ratio_) + 0.5)  
    plt.ylim(0, 1.05)  
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1), fontsize=16)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    for i, bar in enumerate(bars[:5]):  # Only label first 5 bars to avoid clutter
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f'{height:.2f}',
            ha='center',
            fontsize=18,
            fontweight='bold'
        )
    
    plt.legend(
        bbox_to_anchor=(1.0, 0.6),  
        loc='center right',        
        frameon=True,
        framealpha=0.9,
        fontsize=18,
        edgecolor='black',
        fancybox=True
    )
    
    plt.tight_layout()
    if base_model == None:
        plt.savefig(FIG_DIR + 'explained_variance_ratio.pdf', dpi=300, bbox_inches='tight')

    else:
        plt.savefig(FIG_DIR + base_model + '_explained_variance_ratio.pdf', dpi=300, bbox_inches='tight')


def plot_pca_distance(df_new, cols_to_transform_new, model_names = ['Qwen2.5', 'Qwen2.5-14B'], FIG_DIR = '/content/drive/MyDrive/LLM causality/Figures/'):
    X = df_new[cols_to_transform_new].values
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    residuals = X - pca.inverse_transform(X_pca)
    
    # Calculate residual lengths and ratios for all data
    residual_lengths = np.linalg.norm(residuals, axis=1)
    original_lengths = np.linalg.norm(X, axis=1)
    ratios = residual_lengths / original_lengths

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3']
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 18  # Increased base font size
    
    plt.figure(figsize=(10, 6))
    
    plt.grid(True, linestyle='--', alpha=0.4, axis='both')
    
    n, bins, patches = plt.hist(ratios, bins=25, color=colors[0], edgecolor='black',
                               alpha=0.6, label='All Data', density=True, range=(0, 0.2),
                               linewidth=0.8)

    def residual_subset(model_name):
        """Extract residual ratios for a specific model."""
        df = df_new[df_new['Identified base model'].str.contains(model_name, case=False, na=False)]
        X = df[cols_to_transform_new].values
    
        indices = df.index
    
        # Calculate residual lengths for filtered rows only
        residual_lengths = np.linalg.norm(residuals[indices], axis=1)
        original_lengths = np.linalg.norm(X, axis=1)
        ratios_model = residual_lengths / original_lengths
    
        return ratios_model
    
    for i, model_name in enumerate(model_names):
        model_ratios = residual_subset(model_name.lower())
        plt.hist(model_ratios, bins=25, color=colors[i+1], edgecolor='black',
                 alpha=0.6, label=f'{model_name}', density=True, range=(0, 0.2),
                 linewidth=0.8)
    
        mean_ratio = np.mean(model_ratios)
        plt.axvline(mean_ratio, color=colors[i+1], linestyle='--', linewidth=2.5,
                    label=f'{model_name} Mean: {mean_ratio:.4f}')
    
        plt.plot([mean_ratio], [0], marker='o', markersize=14, color=colors[i+1],
                 markeredgecolor='black', markeredgewidth=1.5)
    
    mean_all = np.mean(ratios)
    plt.axvline(mean_all, color=colors[0], linestyle='--', linewidth=2.5,
                label=f'All Data Mean: {mean_all:.4f}')
    
    plt.plot([mean_all], [0], marker='o', markersize=14, color=colors[0],
             markeredgecolor='black', markeredgewidth=1.5)
    plt.xlabel('Ratio of Residual Length to Original Length', fontsize=24, fontweight='bold')
    plt.ylabel('Probability Density', fontsize=24, fontweight='bold')
    
    plt.xlim(0, 0.2)
    plt.ylim(bottom=0)  
    plt.xticks(np.arange(0, 0.21, 0.02), fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=20,
               title_fontsize=16,
               edgecolor='black', fancybox=True)
    
    total_samples = len(ratios)
    qwen_samples = len(residual_subset('qwen2.5'))
    qwen_14b_samples = len(residual_subset('qwen2.5-14b'))
    
    stats_text = (f"Sample Sizes: All Data = {total_samples:,}  |  "
                  f"{model_names[0]} = {qwen_samples:,}  |  "
                  f"{model_names[1]} = {qwen_14b_samples:,}")
    
    plt.figtext(0.5, 0.01, stats_text, fontsize=18, ha='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                         alpha=0.8, edgecolor='lightgray'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  
    plt.savefig(FIG_DIR + 'residual_ratio.pdf', dpi=300, bbox_inches='tight')
