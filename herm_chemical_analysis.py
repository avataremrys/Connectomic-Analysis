import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

def parse_csv_to_adjacency_matrix(file_path):
    """
    Creates an adjacency matrix from the CSV file.
    Row and column labels are unified and stripped of whitespace.
    """
    try:
        df = pd.read_csv(file_path, index_col=0, keep_default_na=False, na_values=[''])
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None, None

    df.columns = [str(lbl).strip() for lbl in df.columns]
    df.index = [str(idx).strip() for idx in df.index]

    col_labels_from_csv = df.columns.tolist()
    row_labels_from_csv = df.index.tolist()

    all_unique_labels = sorted(list(set(row_labels_from_csv + col_labels_from_csv)))
    label_to_idx_map = {label: i for i, label in enumerate(all_unique_labels)}
    
    num_labels = len(all_unique_labels)
    adj_matrix = np.zeros((num_labels, num_labels), dtype=float)

    for r_label_csv_stripped, row_data in df.iterrows():
        if r_label_csv_stripped not in label_to_idx_map:
            continue 
        r_idx_mat = label_to_idx_map[r_label_csv_stripped]
        
        for c_label_csv_stripped, val in row_data.items():
            if c_label_csv_stripped not in label_to_idx_map:
                continue
            c_idx_mat = label_to_idx_map[c_label_csv_stripped]
            
            if isinstance(val, str) and val.strip() == '':
                adj_matrix[r_idx_mat, c_idx_mat] = 0.0
            elif pd.notna(val): 
                try:
                    num_val = float(val)
                    adj_matrix[r_idx_mat, c_idx_mat] = max(0.0, num_val) # Treat negative as 0
                except ValueError:
                    adj_matrix[r_idx_mat, c_idx_mat] = 0.0
            else: 
                adj_matrix[r_idx_mat, c_idx_mat] = 0.0
                
    return adj_matrix, all_unique_labels, label_to_idx_map

def build_markov_matrix_for_scc(scc_subgraph, scc_name_for_debug=""):
    nodes_in_scc = list(scc_subgraph.nodes())
    n_scc = len(nodes_in_scc)

    if n_scc == 0:
        return np.array([]), []

    node_to_idx_scc = {node: i for i, node in enumerate(nodes_in_scc)}
    P = np.zeros((n_scc, n_scc), dtype=float)

    # First pass: calculate out-degrees
    out_degrees = np.zeros(n_scc)
    for i, u_node in enumerate(nodes_in_scc):
        weighted_out_degree_u = scc_subgraph.out_degree(u_node, weight='weight')
        out_degrees[i] = weighted_out_degree_u

    # Second pass: build transition matrix
    for i, u_node in enumerate(nodes_in_scc):
        if out_degrees[i] > 0:
            for v_node in scc_subgraph.successors(u_node): 
                if v_node not in node_to_idx_scc: 
                    continue
                j = node_to_idx_scc[v_node]
                edge_data = scc_subgraph.get_edge_data(u_node, v_node)
                edge_weight = edge_data.get('weight', 0.0) if edge_data else 0.0
                
                if edge_weight > 0:
                    P[i, j] += edge_weight / out_degrees[i]
        else:
            # Add self-loop for nodes with no outgoing edges
            P[i, i] = 1.0

    # Debug information for largest SCC
    if "SCC #1" in scc_name_for_debug:
        print(f"\nDebug information for {scc_name_for_debug}:")
        print(f"Matrix shape: {P.shape}")
        print(f"Number of nodes with zero out-degree: {np.sum(out_degrees == 0)}")
        
        # Check row sums
        row_sums = P.sum(axis=1)
        print("\nRow sum statistics:")
        print(f"Min row sum: {row_sums.min():.6f}")
        print(f"Max row sum: {row_sums.max():.6f}")
        print(f"Mean row sum: {row_sums.mean():.6f}")
        print(f"Number of rows not summing to 1: {np.sum(~np.isclose(row_sums, 1.0, rtol=1e-10))}")
        
        # Print first few rows and their sums
        print("\nFirst 5 rows and their sums:")
        for i in range(min(5, len(nodes_in_scc))):
            print(f"Row {i} ({nodes_in_scc[i]}): sum = {row_sums[i]:.6f}")
            print(f"Non-zero elements: {np.nonzero(P[i])[0]}")
            print(f"Values: {P[i, np.nonzero(P[i])[0]]}")

    return P, nodes_in_scc

def find_stationary_distribution(P_matrix, scc_nodes_list_for_debug=None, scc_name_for_debug=""):
    if P_matrix.shape[0] == 0:
        return None
    
    if P_matrix.shape == (1,1):
        if np.isclose(P_matrix[0,0], 1.0):
            return np.array([1.0])
        else:
            return None

    # Check for irreducibility
    if "SCC #1" in scc_name_for_debug:
        print(f"\nChecking irreducibility for {scc_name_for_debug}:")
        # Compute P^n for large n to check if all entries are positive
        P_power = np.linalg.matrix_power(P_matrix, 100)
        is_irreducible = np.all(P_power > 0)
        print(f"Matrix is {'irreducible' if is_irreducible else 'reducible'}")
        if not is_irreducible:
            print(f"Number of zero entries in P^100: {np.sum(P_power == 0)}")
            print(f"Min non-zero entry in P^100: {np.min(P_power[P_power > 0])}")

    try:
        eigenvalues, eigenvectors = np.linalg.eig(P_matrix.T)
        
        if "SCC #1" in scc_name_for_debug:
            print("\nEigenvalue analysis:")
            # Sort eigenvalues by magnitude
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            print("Top 5 eigenvalues by magnitude:")
            for i in range(min(5, len(eigenvalues))):
                print(f"λ_{i} = {eigenvalues[i]:.6f}")
            
            # Check for complex eigenvalues
            complex_eigenvalues = eigenvalues[np.abs(eigenvalues.imag) > 1e-10]
            if len(complex_eigenvalues) > 0:
                print("\nComplex eigenvalues found:")
                for i, val in enumerate(complex_eigenvalues):
                    print(f"λ_{i} = {val:.6f}")
    except np.linalg.LinAlgError as e:
        print(f"    Error ({scc_name_for_debug}): Eigenvalue decomposition failed: {str(e)}")
        return None

    one_indices = np.where(np.isclose(eigenvalues, 1.0))[0]
    
    if len(one_indices) == 0:
        closest_eigenvalue_idx = np.argmin(np.abs(eigenvalues - 1.0))
        print(f"    Warning ({scc_name_for_debug}): Eigenvalue 1 not found for P_matrix (closest is {eigenvalues[closest_eigenvalue_idx]:.4f}). Matrix shape: {P_matrix.shape}")
        return None

    stationary_vector_complex = eigenvectors[:, one_indices[0]]
    stationary_vector = np.real(stationary_vector_complex)
    
    # If all entries are negative, flip sign
    if np.all(stationary_vector < 0):
        stationary_vector = -stationary_vector
    # Do not set negative values to zero; keep as is
    
    sum_vec = np.sum(stationary_vector)
    if sum_vec > 1e-9:
        stationary_vector = stationary_vector / sum_vec
    else:
        print(f"    Warning ({scc_name_for_debug}): Stationary vector sums to {sum_vec} (near zero). Cannot normalize reliably.")
        return None 

    return stationary_vector

def plot_heatmap(matrix, title, filename, plot_dir="plots", 
                 xticklabels='auto', yticklabels='auto', cmap=None,
                 show_labels_threshold=50):
    if matrix.size == 0:
        print(f"Skipping heatmap for {title} as matrix is empty.")
        return

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plt.figure(figsize=(12, 10))
    
    # Determine if labels should be shown based on threshold
    actual_xticklabels = xticklabels
    if isinstance(xticklabels, list) and len(xticklabels) > show_labels_threshold:
        actual_xticklabels = False
    elif xticklabels == 'auto' and matrix.shape[1] > show_labels_threshold:
        actual_xticklabels = False
        
    actual_yticklabels = yticklabels
    if isinstance(yticklabels, list) and len(yticklabels) > show_labels_threshold:
        actual_yticklabels = False
    elif yticklabels == 'auto' and matrix.shape[0] > show_labels_threshold:
        actual_yticklabels = False

    # Use default colormap if none specified
    if cmap is None:
        cmap = 'viridis'  # Default matplotlib colormap

    sns.heatmap(matrix, cmap=cmap, xticklabels=actual_xticklabels, yticklabels=actual_yticklabels)
    plt.title(title)
    plt.xlabel("Neuron (column)")
    plt.ylabel("Neuron (row)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()
    print(f"Saved heatmap: {os.path.join(plot_dir, filename)}")


def main(file_path="Herm Chemical.csv"):
    adj_matrix, all_unique_labels, label_to_idx_map = parse_csv_to_adjacency_matrix(file_path)
    
    if adj_matrix is None:
        return

    print(f"Adjacency matrix created with shape: {adj_matrix.shape}")
    print(f"Total unique labels: {len(all_unique_labels)}")

    G_int_labels = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
    G = nx.relabel_nodes(G_int_labels, {i: lbl for lbl, i in label_to_idx_map.items()}, copy=True)

    sccs_gen = nx.strongly_connected_components(G)
    sccs = [list(scc_nodes) for scc_nodes in sccs_gen] # Convert generator to list of lists
    print(f"\nFound {len(sccs)} strongly connected components (SCCs).")

    scc_details = [] 
    non_trivial_scc_count = 0
    for i, scc_nodes_list in enumerate(sccs):
        if len(scc_nodes_list) <= 1: 
            continue
        
        non_trivial_scc_count += 1
        scc_id_str = f"SCC #{non_trivial_scc_count} (Original discovery index {i})"
        
        scc_subgraph = G.subgraph(scc_nodes_list).copy()
        P_scc, P_scc_node_order = build_markov_matrix_for_scc(scc_subgraph, scc_id_str)
        
        current_stationary_dist_dict = None
        if P_scc.size > 0 and P_scc.shape[0] == len(scc_nodes_list):
            is_P_fully_stochastic = True
            for r_idx in range(P_scc.shape[0]):
                row_sum = np.sum(P_scc[r_idx, :])
                if not np.isclose(row_sum, 1.0):
                    if not np.isclose(row_sum, 0.0): # Row sum isn't 1 and also isn't 0 (sink)
                        # print(f"    Error ({scc_id_str}): Row {r_idx} (Node {P_scc_node_order[r_idx]}) of P_scc sums to {row_sum:.4f}, not 1.0 or 0.0.")
                        pass
                    is_P_fully_stochastic = False # If any row not 1.0, it's not suitable for unique stationary dist.
                    break 
            
            if is_P_fully_stochastic:
                stationary_dist_vector = find_stationary_distribution(P_scc, P_scc_node_order, scc_id_str)
                if stationary_dist_vector is not None:
                    current_stationary_dist_dict = {
                        P_scc_node_order[k]: stationary_dist_vector[k] 
                        for k in range(len(P_scc_node_order))
                    }
            # else:
                # print(f"    Info ({scc_id_str}): P_scc is not fully row-stochastic (all rows summing to 1). Skipping stationary distribution.")
        
        scc_details.append({
            "id_str": scc_id_str,
            "nodes": scc_nodes_list, # Order might matter if P_scc_node_order is used
            "size": len(scc_nodes_list),
            "stationary_distribution": current_stationary_dist_dict, 
            "p_scc_node_order": P_scc_node_order # Original order of nodes for P_scc
        })

    print(f"Found {non_trivial_scc_count} non-trivial SCCs (size > 1).")

    # Sort all non-trivial SCCs by size for processing
    scc_details_sorted_by_size = sorted(scc_details, key=lambda x: x["size"], reverse=True)
    
    # --- Hub Reporting ---
    sccs_for_hub_reporting = [scc for scc in scc_details_sorted_by_size if scc["stationary_distribution"] is not None]

    if len(sccs_for_hub_reporting) > 0:
        largest_scc_for_hubs = sccs_for_hub_reporting[0]
        print(f"\n--- Top 15 Hubs for Largest SCC with Valid Stationary Distribution ({largest_scc_for_hubs['id_str']}, Size: {largest_scc_for_hubs['size']}) ---")
        dist = largest_scc_for_hubs["stationary_distribution"]
        sorted_hubs = sorted(dist.items(), key=lambda item: item[1], reverse=True)
        for k in range(min(15, len(sorted_hubs))):
            print(f"  {k+1}. Node: {sorted_hubs[k][0]}, Probability: {sorted_hubs[k][1]:.8f}")
    else:
        print("\nNo non-trivial SCCs found with a valid stationary distribution for hub reporting (Largest SCC).")

    if len(sccs_for_hub_reporting) > 1:
        second_largest_scc_for_hubs = sccs_for_hub_reporting[1]
        print(f"\n--- Top 7 Hubs for Second Largest SCC with Valid Stationary Distribution ({second_largest_scc_for_hubs['id_str']}, Size: {second_largest_scc_for_hubs['size']}) ---")
        dist = second_largest_scc_for_hubs["stationary_distribution"]
        sorted_hubs = sorted(dist.items(), key=lambda item: item[1], reverse=True)
        for k in range(min(7, len(sorted_hubs))):
            print(f"  {k+1}. Node: {sorted_hubs[k][0]}, Probability: {sorted_hubs[k][1]:.8f}")
    else:
        print("\nFewer than two non-trivial SCCs with valid stationary distribution found for hub reporting (Second Largest SCC).")

    # --- Heatmap Generation ---
    plot_dir = "plots_herm_chemical" # Specific directory for these plots

    # 1. Whole Adjacency Matrix Heatmap (Standard)
    # For very large matrices, labels are often omitted or sampled. Here, we omit.
    plot_heatmap(adj_matrix, "Heatmap of Whole Adjacency Matrix", "adj_matrix_full.png", plot_dir=plot_dir,
                 xticklabels=False, yticklabels=False) # Labels likely too dense

    # Identify SCCs for heatmaps (largest and second largest by size, regardless of dist for standard heatmap)
    largest_scc_by_size = scc_details_sorted_by_size[0] if len(scc_details_sorted_by_size) > 0 else None
    second_largest_scc_by_size = scc_details_sorted_by_size[1] if len(scc_details_sorted_by_size) > 1 else None

    # 2. Largest SCC Heatmap (Standard & Sorted)
    if largest_scc_by_size:
        scc_info = largest_scc_by_size
        scc_nodes = scc_info["nodes"]
        scc_indices = [label_to_idx_map[node] for node in scc_nodes]
        sub_adj_matrix_l = adj_matrix[np.ix_(scc_indices, scc_indices)]
        
        plot_heatmap(sub_adj_matrix_l, f"Heatmap of Largest SCC ({scc_info['id_str']})", f"scc_largest_standard.png", plot_dir=plot_dir,
                     xticklabels=scc_nodes, yticklabels=scc_nodes)
        
        if scc_info["stationary_distribution"]:
            dist = scc_info["stationary_distribution"]
            # Original scc_nodes are in the order used for P_scc, which is the order of rows/cols in sub_adj_matrix_l
            # if we build sub_adj_matrix_l based on scc_info["p_scc_node_order"] instead of scc_info["nodes"]
            # For simplicity, let's re-get indices based on p_scc_node_order
            ordered_nodes_for_submatrix = scc_info["p_scc_node_order"]
            scc_indices_ordered = [label_to_idx_map[node] for node in ordered_nodes_for_submatrix]
            sub_adj_for_sort = adj_matrix[np.ix_(scc_indices_ordered, scc_indices_ordered)]

            sorted_node_names_l = sorted(ordered_nodes_for_submatrix, key=lambda n: dist.get(n, -1), reverse=True)
            
            # Create map from original node order in sub_adj_for_sort to new sorted order
            current_order_map_l = {node: idx for idx, node in enumerate(ordered_nodes_for_submatrix)}
            new_order_indices_l = [current_order_map_l[name] for name in sorted_node_names_l]
            
            sorted_sub_adj_matrix_l = sub_adj_for_sort[np.ix_(new_order_indices_l, new_order_indices_l)]
            plot_heatmap(sorted_sub_adj_matrix_l, f"Sorted Heatmap of Largest SCC ({scc_info['id_str']})", f"scc_largest_sorted.png", plot_dir=plot_dir,
                         xticklabels=False, yticklabels=False)
        else:
            print(f"Skipping sorted heatmap for Largest SCC ({scc_info['id_str']}) as it has no valid stationary distribution.")

    # 3. Second Largest SCC Heatmap (Standard & Sorted)
    if second_largest_scc_by_size:
        scc_info = second_largest_scc_by_size
        scc_nodes = scc_info["nodes"]
        scc_indices = [label_to_idx_map[node] for node in scc_nodes]
        sub_adj_matrix_sl = adj_matrix[np.ix_(scc_indices, scc_indices)]
        
        plot_heatmap(sub_adj_matrix_sl, f"Heatmap of Second Largest SCC ({scc_info['id_str']})", f"scc_second_largest_standard.png", plot_dir=plot_dir,
                     xticklabels=scc_nodes, yticklabels=scc_nodes)

        if scc_info["stationary_distribution"]:
            dist = scc_info["stationary_distribution"]
            ordered_nodes_for_submatrix_sl = scc_info["p_scc_node_order"]
            scc_indices_ordered_sl = [label_to_idx_map[node] for node in ordered_nodes_for_submatrix_sl]
            sub_adj_for_sort_sl = adj_matrix[np.ix_(scc_indices_ordered_sl, scc_indices_ordered_sl)]
            
            sorted_node_names_sl = sorted(ordered_nodes_for_submatrix_sl, key=lambda n: dist.get(n, -1), reverse=True)
            
            current_order_map_sl = {node: idx for idx, node in enumerate(ordered_nodes_for_submatrix_sl)}
            new_order_indices_sl = [current_order_map_sl[name] for name in sorted_node_names_sl]

            sorted_sub_adj_matrix_sl = sub_adj_for_sort_sl[np.ix_(new_order_indices_sl, new_order_indices_sl)]
            plot_heatmap(sorted_sub_adj_matrix_sl, f"Sorted Heatmap of Second Largest SCC ({scc_info['id_str']})", f"scc_second_largest_sorted.png", plot_dir=plot_dir,
                         xticklabels=False, yticklabels=False)
        else:
            print(f"Skipping sorted heatmap for Second Largest SCC ({scc_info['id_str']}) as it has no valid stationary distribution.")

    # 4. Sorted Whole Adjacency Matrix Heatmap
    final_node_order_for_full_heatmap = []
    nodes_placed_in_full_heatmap_order = set()

    # scc_details_sorted_by_size was already calculated (all non-trivial sccs)
    for scc_info in scc_details_sorted_by_size:
        scc_nodes_current = scc_info["p_scc_node_order"] # Use the order consistent with P_scc
        s_dist_dict = scc_info["stationary_distribution"]
        
        ordered_nodes_this_scc_segment = []
        if s_dist_dict:
            ordered_nodes_this_scc_segment = sorted(scc_nodes_current, key=lambda n: s_dist_dict.get(n, -1), reverse=True)
        else:
            ordered_nodes_this_scc_segment = sorted(scc_nodes_current) # Alphabetical if no dist

        for node in ordered_nodes_this_scc_segment:
            if node not in nodes_placed_in_full_heatmap_order:
                final_node_order_for_full_heatmap.append(node)
                nodes_placed_in_full_heatmap_order.add(node)
    
    remaining_nodes = [node for node in all_unique_labels if node not in nodes_placed_in_full_heatmap_order]
    remaining_nodes.sort() # Alphabetical for nodes not in non-trivial SCCs (likely trivial SCCs)
    final_node_order_for_full_heatmap.extend(remaining_nodes)

    if len(final_node_order_for_full_heatmap) == len(all_unique_labels):
        original_indices_in_new_order = [label_to_idx_map[node] for node in final_node_order_for_full_heatmap]
        adj_matrix_globally_sorted = adj_matrix[np.ix_(original_indices_in_new_order, original_indices_in_new_order)]
        
        plot_heatmap(adj_matrix_globally_sorted, "Sorted Heatmap of Whole Adjacency Matrix", "adj_matrix_full_sorted.png", plot_dir=plot_dir,
                     xticklabels=False, yticklabels=False)
    else:
        print("Error: Node list for sorted full heatmap does not match total unique labels. Skipping this heatmap.")

if __name__ == '__main__':
    csv_file_path = 'Herm Chemical.csv'
    main(csv_file_path) 