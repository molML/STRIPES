#%%
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
    """Convert the tokens of an atom into a list of indices.

    Tokens missing from the dictionary are appended to it with a fresh index
    rather than discarded: dropping them would silently inflate the similarity
    (an unmatched token can no longer count as a mismatch). Existing indices are
    never renumbered, so an extended dictionary stays compatible with the one it
    came from. Only token identity matters for the Jaccard score below, so the
    index values themselves are arbitrary.
    """
    vector = []
    for token in atom_tokens:
        if token not in token_to_index:
            token_to_index[token] = len(token_to_index)
        vector.append(token_to_index[token])
    return vector


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

    sim_matrix = np.zeros((n_A, n_B))
    for i, vec_a in enumerate(vecs1):
        for j, vec_b in enumerate(vecs2):
            sim_matrix[i, j] = atom_multiset_similarity_optimized(vec_a, vec_b)

    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_sims = sim_matrix[row_ind, col_ind]

    unmatched = abs(n_A - n_B)
    total_pairs = max(n_A, n_B)
    sim_score = np.sum(matched_sims) / total_pairs
    sim_score *= (total_pairs - unmatched) / total_pairs

    return float(sim_score)


def compute_similarity_for_row(row, token_to_index, col_stripes='stripes', col_gen='gen stripes'):
    """Compute Hungarian similarity for a single pair (stripes, gen stripes)."""
    parsed1 = parse_stripe(row[col_stripes])
    parsed2 = parse_stripe(row[col_gen])

    if not parsed1 or not parsed2:
        return np.nan

    vecs1 = precompute_atom_vectors(parsed1, token_to_index)
    vecs2 = precompute_atom_vectors(parsed2, token_to_index)

    if not vecs1 or not vecs2:
        return np.nan

    return hungarian_similarity(vecs1, vecs2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute STRIPES similarity for pre-formed pairs in a CSV file."
    )
    parser.add_argument('--dataset', required=True,
                        help="Path to the input CSV file with 'stripes' and 'gen stripes' columns.")
    parser.add_argument('--token-index', default=None,
                        help="Path to the token-to-index JSON file. Optional: the "
                             "score only depends on token identity, so an empty "
                             "dictionary built from the data gives the same result.")
    parser.add_argument('--save-token-index', default=None,
                        help="Write the dictionary back out, extended with any token "
                             "found in the data but missing from --token-index.")
    parser.add_argument('--output', required=True,
                        help="Path to the output CSV file.")
    parser.add_argument('--col-stripes', default='stripes',
                        help="Name of the stripes column (default: stripes).")
    parser.add_argument('--col-gen', default='gen stripes',
                        help="Name of the generated stripes column (default: gen stripes).")
    args = parser.parse_args()

    start_time = time.time()

    if args.token_index:
        print("Loading token dictionary...")
        token_to_index = build_token_index_from_file(args.token_index)
    else:
        print("No token dictionary given, building one from the data...")
        token_to_index = {}
    n_tokens_loaded = len(token_to_index)
    print(f"Tokens in dictionary: {n_tokens_loaded}")

    print("Loading dataset...")
    df = pd.read_csv(args.dataset)
    print(f"Dataset shape: {df.shape}")

    for col in [args.col_stripes, args.col_gen]:
        if col not in df.columns:
            print(f"Error: column '{col}' not found in dataset. Available: {list(df.columns)}")
            exit(1)

    print(f"Computing similarity for {len(df)} pairs...")

    similarities = []
    invalid_count = 0

    for i, row in df.iterrows():
        sim = compute_similarity_for_row(row, token_to_index, args.col_stripes, args.col_gen)
        similarities.append(sim)

        if pd.isna(sim):
            invalid_count += 1

        if (i + 1) % 1000 == 0 or (i + 1) == len(df):
            elapsed = time.time() - start_time
            print(f"Processed {i + 1}/{len(df)} pairs ({(i+1)/len(df)*100:.1f}%) - Elapsed: {elapsed:.1f}s")

    # Insert stripes_sim between the two stripe columns
    stripes_col_pos = df.columns.get_loc(args.col_stripes)
    gen_col_pos = df.columns.get_loc(args.col_gen)
    insert_pos = max(stripes_col_pos, gen_col_pos) + 1

    df.insert(insert_pos, 'stripes_sim', similarities)

    df.to_csv(args.output, index=False)
    print(f"\nOutput saved: {args.output}")
    print(f"Valid pairs: {len(df) - invalid_count}, Invalid (NaN): {invalid_count}")

    new_tokens = sorted(list(token_to_index)[n_tokens_loaded:])
    if new_tokens:
        print(f"\nTokens found in the data but missing from the dictionary: "
              f"{len(new_tokens)} (dictionary now holds {len(token_to_index)})")
        for token in new_tokens:
            print(f"  {token}")
        if not args.save_token_index:
            print("Pass --save-token-index to store the extended dictionary.")

    if args.save_token_index:
        with open(args.save_token_index, 'w') as f:
            json.dump(token_to_index, f, indent=1)
        print(f"Token dictionary saved: {args.save_token_index} "
              f"({len(token_to_index)} tokens)")

    valid_sims = [s for s in similarities if not pd.isna(s)]
    if valid_sims:
        print("\nSimilarity statistics:")
        print(f"  Mean:   {np.mean(valid_sims):.4f}")
        print(f"  Median: {np.median(valid_sims):.4f}")
        print(f"  Min:    {np.min(valid_sims):.4f}")
        print(f"  Max:    {np.max(valid_sims):.4f}")
        print(f"  Std:    {np.std(valid_sims):.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    if len(df) > 0:
        print(f"Average speed: {len(df)/total_time:.1f} pairs/second")
# %%
