import argparse
import json
import pandas as pd
import numpy as np
from typing import List
import time
from scipy.optimize import linear_sum_assignment 


def parse_stripe(STRIPES_str: str, max_tokens: int = 3000) -> List[List[str]]:
    if pd.isna(STRIPES_str) or STRIPES_str == '':
        return []

    try:
        atoms = STRIPES_str.split(';')
        parsed_atoms = []
        total_tokens = 0

        for atom in atoms:
            if atom.strip():
                tokens = atom.strip().split('.')
                total_tokens += len(tokens)

                if total_tokens > max_tokens:
                    return []

                parsed_atoms.append(tokens)

        return parsed_atoms

    except Exception as e:
        print(f"Error during parsing: {e}")
        return []


def build_token_index_from_file(file_path):
    """Load a token:index dictionary from a JSON file."""
    with open(file_path, 'r') as f:
        token_to_index = json.load(f)
    return token_to_index


def atom_to_index_vector(atom_tokens, token_to_index):
    """Convert the tokens of an atom into a list of indices."""
    return [token_to_index[token] for token in atom_tokens if token in token_to_index]


def precompute_atom_vectors(parsed_stripe, token_to_index):
    """Precompute index vectors for all atoms in a sequence."""
    return [atom_to_index_vector(atom, token_to_index) for atom in parsed_stripe]


def atom_multiset_similarity_optimized(vec1: List[int], vec2: List[int]) -> float:
    """Compute Jaccard similarity between two index vectors (optimized)."""
    if not vec1 and not vec2:
        return 1.0
    if not vec1 or not vec2:
        return 0.0

    set1, set2 = set(vec1), set(vec2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 1.0


def hungarian_similarity(vecs1: List[List[int]], vecs2: List[List[int]]) -> float:
    """Similarity with exclusive 1-to-1 matching via the Hungarian algorithm."""
    n_A = len(vecs1)
    n_B = len(vecs2)

    if n_A == 0 and n_B == 0:
        return 1.0
    if n_A == 0 or n_B == 0:
        return 0.0

    # Build similarity matrix (n_A x n_B)
    sim_matrix = np.zeros((n_A, n_B))
    for i, vec_a in enumerate(vecs1):
        for j, vec_b in enumerate(vecs2):
            sim_matrix[i, j] = atom_multiset_similarity_optimized(vec_a, vec_b)

    # Convert to cost matrix for the Hungarian algorithm (minimization)
    cost_matrix = 1.0 - sim_matrix

    # Solve the optimal 1-to-1 matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute average similarity over matched pairs
    matched_sims = sim_matrix[row_ind, col_ind]

    # Penalize unmatched atoms when sequences have different lengths
    unmatched = abs(n_A - n_B)
    total_pairs = max(n_A, n_B)
    sim_score = (np.sum(matched_sims)) / total_pairs

    # Apply penalty for unmatched atoms
    sim_score *= (total_pairs - unmatched) / total_pairs

    return float(sim_score)


def compare_STRIPES_direct(vecs1: List[List[int]], vecs2: List[List[int]]) -> float:
    """Compare two preprocessed sequences directly."""
    return hungarian_similarity(vecs1, vecs2)


def filter_valid_STRIPES(df: pd.DataFrame, max_length: int = 3000) -> pd.DataFrame:
    """Filter and clean valid STRIPES sequences."""
    print("Filtering STRIPES sequences...")

    df_clean = df.dropna(subset=['STRIPES'])
    df_clean = df_clean[df_clean['STRIPES'].str.strip() != '']

    df_clean = df_clean[df_clean['STRIPES'].str.len() < max_length]

    print(f"Sequences after filtering: {len(df)} -> {len(df_clean)}")
    return df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute pairwise STRIPES similarity using the Hungarian algorithm.")
    parser.add_argument('--dataset', required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--token-index', required=True, help="Path to the token-to-index JSON file.")
    parser.add_argument('--output', required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    start_time = time.time()

    print("Loading dataset...")
    df = pd.read_csv(args.dataset)

    print("Loading token dictionary...")
    token_to_index = build_token_index_from_file(args.token_index)

    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of tokens in dictionary: {len(token_to_index)}")

    if 'STRIPES' not in df.columns:
        print("Error: column 'STRIPES' not found in the dataset")
        exit(1)

    df_clean = filter_valid_STRIPES(df)

    print(f"Number of filtered STRIPES sequences: {len(df_clean)}")

    # Preprocess each sequence once
    print("Preprocessing STRIPES sequences...")

    # Keep track of valid indices and corresponding data
    valid_data = []
    invalid_count = 0

    for idx, row in df_clean.iterrows():
        stripe_str = row['STRIPES']
        parsed = parse_stripe(stripe_str)

        if parsed:  # Only if parsing succeeded
            vectors = precompute_atom_vectors(parsed, token_to_index)
            if vectors:  # Only if valid vectors were produced
                valid_data.append({
                    'original_index': idx,
                    'smiles': row['smiles'],
                    'STRIPES': stripe_str,
                    'pKi': row['pKi'],
                    'vectors': vectors
                })
            else:
                invalid_count += 1
        else:
            invalid_count += 1

        if len(valid_data) % 1000 == 0:
            print(f"Preprocessed {len(valid_data)} valid sequences")

    n_valid = len(valid_data)
    print(f"Preprocessing complete. Valid: {n_valid}, Invalid: {invalid_count}")

    if n_valid < 2:
        print("Error: not enough valid sequences to compute similarities")
        exit(1)

    print("Computing pairwise similarities...")

    rows = []
    similarity_values = []
    total_pairs = (n_valid * (n_valid + 1)) // 2  # Includes the diagonal
    computed_pairs = 0

    # Compute similarity for all pairs
    for i in range(n_valid):
        data_i = valid_data[i]

        for j in range(i, n_valid):  # Includes the diagonal
            data_j = valid_data[j]

            if i == j:
                similarity = 1.0
            else:
                similarity = compare_STRIPES_direct(data_i['vectors'], data_j['vectors'])
                similarity_values.append(similarity)  # Off-diagonal values only

            rows.append({
                'smiles1': data_i['smiles'],
                'smiles2': data_j['smiles'],
                'STRIPES1': data_i['STRIPES'],
                'STRIPES2': data_j['STRIPES'],
                'pKi1': data_i['pKi'],
                'pKi2': data_j['pKi'],
                'similarity': similarity
            })

            computed_pairs += 1

            if computed_pairs % 10000 == 0 or computed_pairs == total_pairs:
                elapsed = time.time() - start_time
                progress = computed_pairs / total_pairs * 100
                print(f"Processed {computed_pairs}/{total_pairs} pairs ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")

    print("Computation complete!")

    result_df = pd.DataFrame(rows)

    result_df.to_csv(args.output, index=False)
    print(f"CSV file saved: {args.output}")

    # Similarity statistics (excluding the diagonal)
    if similarity_values:
        print("\nSimilarity statistics:")
        print(f"Mean:   {np.mean(similarity_values):.4f}")
        print(f"Median: {np.median(similarity_values):.4f}")
        print(f"Min:    {np.min(similarity_values):.4f}")
        print(f"Max:    {np.max(similarity_values):.4f}")
        print(f"Std:    {np.std(similarity_values):.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if computed_pairs > 0:
        print(f"Average speed: {computed_pairs/total_time:.1f} pairs/second")
