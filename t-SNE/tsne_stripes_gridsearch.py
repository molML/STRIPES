# %%
import argparse
import pandas as pd
import numpy as np
import json
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import entropy
import os
from itertools import product
import warnings
import matplotlib as mpl

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description="Grid search for optimal t-SNE hyperparameters on STRIPES embeddings."
)
parser.add_argument(
    "--vocab_path", required=True,
    help="Path to the token-to-label JSON vocabulary file (stripes_token2label.json)."
)
parser.add_argument(
    "--data_path", required=True,
    help="Path to the MISATO protein features CSV file (MISATO_protein_feat.csv)."
)
parser.add_argument(
    "--output_dir", default="results/tsne_grid_search",
    help="Directory where grid-search results are saved (default: results/tsne_grid_search)."
)
parser.add_argument(
    "--svg_save_dir", default="results/figures",
    help="Directory where final SVG figures for the best configuration are saved (default: results/figures)."
)
args = parser.parse_args()

vocab_path   = args.vocab_path
data_path    = args.data_path
output_dir   = args.output_dir
svg_save_dir = args.svg_save_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

if not os.path.exists(vocab_path):
    print(f"ERROR: Vocabulary file not found: {vocab_path}")
    exit(1)

if not os.path.exists(data_path):
    print(f"ERROR: Data file not found: {data_path}")
    exit(1)

try:
    with open(vocab_path, 'r') as f:
        token2label = json.load(f)
    print(f"Vocabulary loaded: {len(token2label)} tokens")
except Exception as e:
    print(f"Error loading vocabulary: {e}")
    exit(1)

def parse_stripes_string(stripes_str, max_tokens: int = 1200):  
    """
    Converts a STRIPES string into a list of atoms (token sequences),
    skipping entries with more than max_tokens tokens and removing subsequences
    that contain exactly 11 tokens with the '-' character.
    """
    if pd.isna(stripes_str) or stripes_str == '':
        return []
    
    try:
        atoms = stripes_str.split(';')
        parsed_atoms = []
        total_tokens = 0
        
        for atom in atoms:
            if atom.strip():  
                tokens = atom.strip().split('.')
                total_tokens += len(tokens)
                if total_tokens > max_tokens:
                    return []  
                if tokens.count('-') == 11:
                    continue
                
                parsed_atoms.append(tokens)
        
        return parsed_atoms
    except Exception as e:
        print(f"Error parsing STRIPES string: {e}")
        return []

def create_stripes_embedding(stripes_str, token2label):
    """
    Creates an embedding for a ligand based on its STRIPES tokens.
    Each token is converted to a one-hot vector and the mean is computed.
    """
    atoms = parse_stripes_string(stripes_str)
    
    if not atoms:
        return np.zeros(len(token2label))
    
    vocab_size = len(token2label)
    all_embeddings = []
    
    for atom in atoms:
        for token in atom:
            if token in token2label:
                token_embedding = np.zeros(vocab_size)
                label = token2label[token]
                if isinstance(label, int) and 0 <= label < vocab_size:
                    token_embedding[label] = 1.0
                elif not isinstance(label, int):
                    idx = abs(hash(str(label))) % vocab_size
                    token_embedding[idx] = 1.0
                
                all_embeddings.append(token_embedding)
    
    if not all_embeddings:
        return np.zeros(vocab_size)
    
    return np.mean(all_embeddings, axis=0)

def calculate_entropy_features(stripes_str, token2label):
    """
    Computes entropy features for a STRIPES ligand.
    """
    atoms = parse_stripes_string(stripes_str)
    
    if not atoms:
        return [0, 0]
    
    all_tokens = []
    atom_entropies = []
    
    for atom in atoms:
        valid_tokens = [token for token in atom if token in token2label]
        all_tokens.extend(valid_tokens)
        
        if valid_tokens:
            token_counts = Counter(valid_tokens)
            total_tokens = len(valid_tokens)
            probs = [count/total_tokens for count in token_counts.values()]
            atom_entropy = entropy(probs, base=2)
            atom_entropies.append(atom_entropy)
        else:
            atom_entropies.append(0)
    
    if all_tokens:
        overall_token_counts = Counter(all_tokens)
        total_overall_tokens = len(all_tokens)
        overall_probs = [count/total_overall_tokens for count in overall_token_counts.values()]
        overall_entropy = entropy(overall_probs, base=2)
    else:
        overall_entropy = 0
    
    avg_atom_entropy = np.mean(atom_entropies) if atom_entropies else 0
    
    return [overall_entropy, avg_atom_entropy]

def get_num_atoms(stripes_str):
    """
    Counts the number of atoms in the ligand.
    """
    atoms = parse_stripes_string(stripes_str)
    return len(atoms)

def count_h_bonds(stripes_str):
    """
    Count hydrogen bonds from STRIPES string.
    """
    atoms = parse_stripes_string(stripes_str)
    
    if not atoms:
        return 0.0
    
    h_token_count = 0
    for atom in atoms:
        for token in atom:
            h_token_count += token.count('H(')
    
    return h_token_count / 11.0

def calculate_trustworthiness(X_high, X_low, k=7):
    """
    Computes the trustworthiness of the t-SNE projection.
    """
    n = X_high.shape[0]
    
    dist_high = pairwise_distances(X_high)
    
    dist_low = pairwise_distances(X_low)
    
    nn_low = np.argsort(dist_low, axis=1)[:, 1:k+1]
    
    rank_high = np.argsort(np.argsort(dist_high, axis=1), axis=1)
    
    trustworthiness = 0
    for i in range(n):
        for j in nn_low[i]:
            if rank_high[i, j] > k:
                trustworthiness += (rank_high[i, j] - k)
    
    trustworthiness = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * trustworthiness
    return trustworthiness

def calculate_continuity(X_high, X_low, k=7):
    """
    Computes the continuity of the t-SNE projection.
    """
    n = X_high.shape[0]
    
    dist_high = pairwise_distances(X_high)
    
    dist_low = pairwise_distances(X_low)
    
    nn_high = np.argsort(dist_high, axis=1)[:, 1:k+1]
    
    rank_low = np.argsort(np.argsort(dist_low, axis=1), axis=1)
    
    continuity = 0
    for i in range(n):
        for j in nn_high[i]:
            if rank_low[i, j] > k:
                continuity += (rank_low[i, j] - k)
    
    continuity = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * continuity
    return continuity


def create_tsne_plot(tsne_results, df, params, config_name, output_dir, svg_save_path=None):
    """
    Creates separate t-SNE plots for each property: one square figure per plot, saved as SVG.
    """
    properties_to_plot = ['lig_logP', 'lig_TPSA', 'lig_H_donor', 'count_h_bonds', 'hydrophobic_atoms', 'net_charge', 'frac_hydrophobic', 'frac_polar', 'frac_positive', 'frac_negative', 'mean_hydropathy', 'mean_sasa']
    display_names = {
        'lig_logP': 'logP',
        'lig_TPSA': 'TPSA',
        'lig_H_donor': 'H Donor',
        'count_h_bonds': 'H-bond Interactions',
        'hydrophobic_atoms': 'Hydrophobic Atoms',
        'net_charge': 'Net Charge',
        'frac_hydrophobic': 'Fraction Hydrophobic',
        'frac_polar': 'Fraction Polar',
        'frac_positive': 'Fraction Positive',
        'frac_negative': 'Fraction Negative',
        'mean_hydropathy': 'Mean Hydropathy',
        'mean_sasa': 'Mean SASA'
    }

     # Outliers beyond these percentiles are clamped to the boundary color.
    CLIP_LOW  = 5   # lower percentile
    CLIP_HIGH = 95  # upper percentile

    save_dir = svg_save_path if svg_save_path is not None else output_dir
    os.makedirs(save_dir, exist_ok=True)

    first_filename = None
    for prop in properties_to_plot:
        prop_values = df[prop].values
        display_name = display_names.get(prop, prop)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        if np.all(np.isnan(prop_values)):
            ax.text(0.5, 0.5, f'No data for {display_name}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='red')
        else:
            valid_values = prop_values[~np.isnan(prop_values)]
            vmin = np.percentile(valid_values, CLIP_LOW)
            vmax = np.percentile(valid_values, CLIP_HIGH)

            scatter = ax.scatter(
                tsne_results[:, 0], tsne_results[:, 1],
                c=np.clip(prop_values, vmin, vmax),
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                alpha=0.8,
                s=55,
                edgecolor='none',
                rasterized=True
            )

            cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                               fraction=0.05, pad=0.12, aspect=30)
            cbar.set_label(display_name, fontsize=34)
            cbar.ax.tick_params(labelsize=22)

            x_range = tsne_results[:, 0].max() - tsne_results[:, 0].min()
            y_range = tsne_results[:, 1].max() - tsne_results[:, 1].min()
            margin = 0.05
            ax.set_xlim(tsne_results[:, 0].min() - margin * x_range,
                         tsne_results[:, 0].max() + margin * x_range)
            ax.set_ylim(tsne_results[:, 1].min() - margin * y_range,
                         tsne_results[:, 1].max() + margin * y_range)

            ax.set_xlabel('t-SNE 1', fontsize=34)
            ax.set_ylabel('t-SNE 2', fontsize=34)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', length=0)
        ax.set_box_aspect(1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        ax.grid(False)

        svg_filename = os.path.join(save_dir, f'tsne_{config_name}_{prop}.svg')
        plt.savefig(svg_filename, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  SVG saved: {svg_filename}")
        if first_filename is None:
            first_filename = svg_filename
        plt.close(fig)

    return first_filename


def create_correlation_plot(tsne_results, df, params, config_name, output_dir):
    """
    Creates a correlation plot for a specific configuration.
    """
    analysis_df = df.copy()
    analysis_df['tsne_x'] = tsne_results[:, 0]
    analysis_df['tsne_y'] = tsne_results[:, 1]
    
    correlation_props = ['lig_MW', 'lig_logP', 'lig_TPSA', 'num_atoms', 'overall_entropy', 'avg_atom_entropy', 'count_h_bonds', 'tsne_x', 'tsne_y']
    corr_matrix = analysis_df[correlation_props].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title(f'Correlation Matrix - {config_name}\nperplexity={params["perplexity"]}, n_iter={params["n_iter"]}, learning_rate={params["learning_rate"]}\nTrustworthiness={params["trustworthiness"]:.3f}, Continuity={params["continuity"]:.3f}',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'correlation_{config_name}.png')
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    
    return filename

try:
    data_df = pd.read_csv(data_path)
    print(f"Data loaded: {len(data_df)} rows")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

required_columns = ['STRIPES', 'lig_MW', 'lig_logP', 'lig_TPSA', 'lig_H_donor', 'lig_H_acceptor',
                   'hydrophobic_atoms', 'polar_atoms', 'net_charge',
                   'frac_hydrophobic', 'frac_polar', 'frac_positive',
                   'frac_negative', 'mean_hydropathy', 'mean_sasa']

missing_columns = [col for col in required_columns if col not in data_df.columns]
if missing_columns:
    print(f"ERROR: Missing columns in dataset: {missing_columns}")
    exit(1)

df = data_df[required_columns].copy()
df = df.dropna(subset=['STRIPES'])
print(f"After NaN removal: {len(df)} rows")

if len(df) == 0:
    print("ERROR: No valid data after cleaning")
    exit(1)

print("Creating STRIPES embeddings for each ligand...")

embeddings_list = []
entropy_features_list = []
num_atoms_list = []
h_bond_list = []

for i, stripes_str in enumerate(df['STRIPES']):
    if i % 100 == 0:
        print(f"Processed {i}/{len(df)} samples...")
    
    try:
        ligand_embedding = create_stripes_embedding(stripes_str, token2label)
        embeddings_list.append(ligand_embedding)
        
        entropy_features = calculate_entropy_features(stripes_str, token2label)
        entropy_features_list.append(entropy_features)
        
        num_atoms = get_num_atoms(stripes_str)
        num_atoms_list.append(num_atoms)

        h_bonds = count_h_bonds(stripes_str)
        h_bond_list.append(h_bonds)
        
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        embeddings_list.append(np.zeros(len(token2label)))
        entropy_features_list.append([0, 0])
        num_atoms_list.append(0)
        h_bond_list.append(0)

embeddings_matrix = np.array(embeddings_list)
entropy_matrix = np.array(entropy_features_list)

print(f"Embeddings matrix created: {embeddings_matrix.shape}")

if np.any(np.isnan(embeddings_matrix)) or np.any(np.isinf(embeddings_matrix)):
    print("WARNING: Replacing NaN/infinite values with 0")
    embeddings_matrix = np.nan_to_num(embeddings_matrix, nan=0.0, posinf=0.0, neginf=0.0)


df = df.reset_index(drop=True)
df['num_atoms'] = num_atoms_list
df['overall_entropy'] = entropy_matrix[:, 0]
df['avg_atom_entropy'] = entropy_matrix[:, 1]
df['count_h_bonds'] = h_bond_list

tsne_param_grid = {
    'n_components': [2],
    'perplexity': [5, 15, 30, 50],
    'n_iter': [500,1000, 2000],
    'learning_rate': [10, 100, 500],
    'init': ['random'],
    'metric': ['euclidean']
}

param_combinations = list(product(
    tsne_param_grid['n_components'],
    tsne_param_grid['perplexity'],
    tsne_param_grid['n_iter'],
    tsne_param_grid['learning_rate'],
    tsne_param_grid['init'],
    tsne_param_grid['metric']
))

print(f"Testing {len(param_combinations)} t-SNE hyperparameter combinations...")

tsne_results_dict = {}
quality_metrics = []

for i, (n_comp, perp, n_iter, lr, init, metric) in enumerate(param_combinations):
    print(f"Combination {i+1}/{len(param_combinations)}: perplexity={perp}, n_iter={n_iter}, lr={lr}, init={init}, metric={metric}")
    
    adjusted_perplexity = min(perp, len(df) - 1)
    if adjusted_perplexity < 5:
        adjusted_perplexity = min(5, len(df) - 1)
    
    try:
        tsne = TSNE(
            n_components=n_comp,
            perplexity=adjusted_perplexity,
            random_state=42,
            max_iter=n_iter,
            learning_rate=lr,
            verbose=0
        )
        
        results = tsne.fit_transform(embeddings_matrix)
        
        trustworthiness = calculate_trustworthiness(embeddings_matrix, results)
        continuity = calculate_continuity(embeddings_matrix, results)
        
        config_name = f"p{adjusted_perplexity}_i{n_iter}_lr{lr}"
        
        params = {
            'perplexity': adjusted_perplexity,
            'n_iter': n_iter,
            'learning_rate': lr,
            'trustworthiness': trustworthiness,
            'continuity': continuity
        }
        
        tsne_results_dict[config_name] = {
            'results': results,
            'params': params
        }
        
        quality_metrics.append({
            'config_name': config_name,
            'perplexity': adjusted_perplexity,
            'n_iter': n_iter,
            'learning_rate': lr,
            'trustworthiness': trustworthiness,
            'continuity': continuity,
            'combined_score': (trustworthiness + continuity) / 2
        })
        
        print(f"  Trustworthiness: {trustworthiness:.3f}, Continuity: {continuity:.3f}")
        
    except Exception as e:
        print(f"Error running t-SNE for combination {i+1}: {e}")
        continue

if not tsne_results_dict:
    print("ERROR: No valid t-SNE results obtained")
    exit(1)

quality_df = pd.DataFrame(quality_metrics)
best_config = quality_df.loc[quality_df['combined_score'].idxmax()]
best_config_name = best_config['config_name']

print(f"\nBest configuration found: {best_config_name}")
print(f"Trustworthiness: {best_config['trustworthiness']:.3f}")
print(f"Continuity: {best_config['continuity']:.3f}")
print(f"Combined Score: {best_config['combined_score']:.3f}")

print("\nCreating plots for all configurations...")
for config_name, data in tsne_results_dict.items():
    results = data['results']
    params = data['params']

    config_svg_dir = os.path.join(output_dir, config_name)
    plot_filename = create_tsne_plot(results, df, params, config_name, output_dir, svg_save_path=config_svg_dir)
    corr_filename = create_correlation_plot(results, df, params, config_name, output_dir)

    if config_name == best_config_name:
        print(f"  BEST - {config_name}: {plot_filename}")
        print(f"  BEST - {config_name}: {corr_filename}")

best_results = tsne_results_dict[best_config_name]['results']
best_params = tsne_results_dict[best_config_name]['params']

best_plot_filename = create_tsne_plot(best_results, df, best_params, f"BEST_{best_config_name}", output_dir, svg_save_path=svg_save_dir)
best_corr_filename = create_correlation_plot(best_results, df, best_params, f"BEST_{best_config_name}", output_dir)

best_results_df = df.copy()
best_results_df['tsne_x'] = best_results[:, 0]
best_results_df['tsne_y'] = best_results[:, 1]
best_results_df['parameter_set'] = best_config_name

embedding_stats = {
    'embedding_mean': np.mean(embeddings_matrix, axis=1),
    'embedding_std': np.std(embeddings_matrix, axis=1),
    'embedding_max': np.max(embeddings_matrix, axis=1),
    'embedding_sparsity': np.sum(embeddings_matrix == 0, axis=1) / embeddings_matrix.shape[1]
}

for stat_name, stat_values in embedding_stats.items():
    best_results_df[stat_name] = stat_values

best_file = os.path.join(output_dir, 'best_configuration_results.csv')
best_results_df.to_csv(best_file, index=False)

quality_file = os.path.join(output_dir, 'quality_metrics.csv')
quality_df.to_csv(quality_file, index=False)

summary_file = os.path.join(output_dir, 'experiment_summary.txt')
with open(summary_file, 'w') as f:
    f.write("=== TSNE GRID SEARCH EXPERIMENT SUMMARY ===\n\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Number of samples: {len(df)}\n")
    f.write(f"Vocabulary size: {len(token2label)}\n")
    f.write(f"Embeddings dimensions: {embeddings_matrix.shape}\n")
    f.write(f"t-SNE configurations tested: {len(tsne_results_dict)}\n\n")

    f.write("=== BEST CONFIGURATION ===\n")
    f.write(f"Name: {best_config_name}\n")
    f.write(f"Perplexity: {best_config['perplexity']}\n")
    f.write(f"N_iter: {best_config['n_iter']}\n")
    f.write(f"Learning_rate: {best_config['learning_rate']}\n")
    f.write(f"Trustworthiness: {best_config['trustworthiness']:.6f}\n")
    f.write(f"Continuity: {best_config['continuity']:.6f}\n")
    f.write(f"Combined Score: {best_config['combined_score']:.6f}\n\n")
    
    f.write("=== TOP 5 CONFIGURATIONS ===\n")
    top_5 = quality_df.nlargest(5, 'combined_score')
    for i, row in top_5.iterrows():
        f.write(f"{i+1}. {row['config_name']}: Score={row['combined_score']:.3f} (T={row['trustworthiness']:.3f}, C={row['continuity']:.3f})\n")
    
    f.write("\n=== TESTED PARAMETERS ===\n")
    f.write(f"Perplexity: {tsne_param_grid['perplexity']}\n")
    f.write(f"N_iter: {tsne_param_grid['n_iter']}\n")
    f.write(f"Learning_rate: {tsne_param_grid['learning_rate']}\n\n")
    
    f.write("=== EMBEDDING STATISTICS ===\n")
    f.write(f"Overall mean: {np.mean(embeddings_matrix):.6f}\n")
    f.write(f"Standard deviation: {np.std(embeddings_matrix):.6f}\n")
    f.write(f"Mean sparsity: {np.mean(embedding_stats['embedding_sparsity']):.3f}\n\n")

    f.write("=== PROPERTY STATISTICS ===\n")
    for prop in ['lig_MW', 'lig_logP', 'lig_TPSA', 'num_atoms', 'overall_entropy', 'avg_atom_entropy']:
        mean_val = df[prop].mean()
        std_val = df[prop].std()
        f.write(f"{prop}: μ={mean_val:.3f}, σ={std_val:.3f}\n")
    
    f.write("\n=== GENERATED FILES ===\n")
    f.write("- best_configuration_results.csv (best configuration)\n")
    f.write("- quality_metrics.csv (quality metrics for all configurations)\n")
    f.write(f"- tsne_BEST_{best_config_name}_<prop>.svg (separate SVG plots for best configuration)\n")
    f.write(f"- correlation_BEST_{best_config_name}.png (correlation plot for best configuration)\n")
    f.write(f"- <config>/<tsne_*_<prop>.svg ({len(tsne_results_dict)} configurations × 12 properties, separate SVGs)\n")
    f.write(f"- correlation_*.png ({len(tsne_results_dict)} correlation matrices)\n")
    f.write("- experiment_summary.txt (this file)\n")

print("\n=== ANALYSIS COMPLETE ===")
print(f"Best configuration: {best_config_name}")
print(f"Trustworthiness: {best_config['trustworthiness']:.3f}")
print(f"Continuity: {best_config['continuity']:.3f}")
print(f"Combined Score: {best_config['combined_score']:.3f}")
print(f"\nAll files saved in directory: {output_dir}")
print(f"Best configuration file: {best_file}")
print(f"Quality metrics: {quality_file}")
print(f"Summary: {summary_file}")
 # %%
