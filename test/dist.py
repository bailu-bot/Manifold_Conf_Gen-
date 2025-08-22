import torch
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from rdkit import Chem

from featurizer import Gen39AtomFeatures



def compute_augmented_graph_distance(mol, 
                                      alpha=1.0, 
                                      beta=1.0):
    """
    Compute an augmented distance matrix D_aug ∈ ℝ^{N×N} for atoms in a molecule,
    integrating three components:
        - topological path distance based on edge weights (D_topo)
        - node feature distance via Euclidean metric (D_feat)
        - edge-aggregated path distance based on edge attributes (D_edge)

    D_aug = D_topo + α·D_feat + β·D_edge

    Args:
        mol (rdkit.Chem.Mol): Input RDKit molecule object.
        alpha (float): Weight for node feature distance.
        beta (float): Weight for edge feature path distance.

    Returns:
        D_aug (ndarray): Normalized augmented distance matrix of shape (N, N).
    """

    smiles = Chem.MolToSmiles(mol)
    data = type('obj', (object,), {'smiles': smiles})()  # mock object
    featurizer = Gen39AtomFeatures()
    data = featurizer(data)
    node_features=data.x
    edge_index=data.edge_index
    edge_attr=data.edge_attr


    N = node_features.size(0)

    # Step 1: Build weighted NetworkX graph from edge_index and edge_attr
    G = nx.Graph()
    edge_weights = {}
    for i in range(edge_index.size(1)):
        u, v = edge_index[:, i].tolist()
        edge_feat = edge_attr[i].detach().cpu().numpy()
        weight = np.linalg.norm(edge_feat)  # ||e||₂
        G.add_edge(u, v, weight=weight, edge_feat=edge_feat)
        edge_weights[(u, v)] = edge_feat
        edge_weights[(v, u)] = edge_feat  # Undirected

    # Step 2: Compute edge-aggregated shortest path matrix D_edge ∈ ℝ^{N×N}
    D_edge = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                D_edge[i, j] = 0
            else:
                try:
                    path = nx.shortest_path(G, source=i, target=j)
                    total_edge_feat = 0
                    for k in range(len(path) - 1):
                        e_feat = edge_weights.get((path[k], path[k + 1]), None)
                        if e_feat is not None:
                            total_edge_feat += np.linalg.norm(e_feat)
                    D_edge[i, j] = total_edge_feat
                except nx.NetworkXNoPath:
                    D_edge[i, j] = np.inf

    # Step 3: Node feature Euclidean distance D_feat ∈ ℝ^{N×N}
    node_feat_np = node_features.detach().cpu().numpy()
    D_feat = cdist(node_feat_np, node_feat_np, metric='euclidean')

    # Step 4: Dijkstra topological distance D_topo ∈ ℝ^{N×N}
    D_topo = np.zeros((N, N))
    for i in range(N):
        lengths = nx.single_source_dijkstra_path_length(G, i, weight='weight')
        for j in range(N):
            D_topo[i, j] = lengths.get(j, np.inf)

    # Step 5: Combine components into D_aug and normalize
    D_aug = D_topo + alpha * D_feat + beta * D_edge
    D_aug = np.nan_to_num(D_aug, nan=1e6, posinf=1e6)
    D_aug = D_aug / D_aug.max()

    return D_aug


from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN

def compute_AE_tanimoto_distance(mol, radius=2, nBits=2048):
    """
    Compute the Tanimoto-based atom distance matrix using Atom Environment (AE) fingerprints.
    
    D[i, j] = 1 - Tanimoto(AE_i, AE_j)

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.
        radius (int): Radius for atom environment.
        nBits (int): Bit size for Morgan fingerprints.

    Returns:
        dist_matrix (ndarray): Atom-wise distance matrix of shape (N_atoms, N_atoms).
    """
    num_atoms = mol.GetNumAtoms()
    atom_fps = []

    # Step 1: Extract local atom environments and compute fingerprints
    for atom_idx in range(num_atoms):
        env_ids = FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        env_mol = Chem.PathToSubmol(mol, env_ids)
        Chem.GetSymmSSSR(env_mol)  # Ensure RingInfo is initialized
        fp = AllChem.GetMorganFingerprintAsBitVect(env_mol, radius, nBits=nBits)
        atom_fps.append(fp)

    # Step 2: Compute pairwise Tanimoto similarities
    sim_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(num_atoms):
            sim = DataStructs.TanimotoSimilarity(atom_fps[i], atom_fps[j])
            sim_matrix[i, j] = sim

    # Step 3: Convert similarity to distance
    dist_matrix = 1.0 - sim_matrix
    return dist_matrix



from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist
import numpy as np

def compute_embed3d_distance(mol, embed=True, optimize=True, maxIters=200):


    if mol is None:
        raise ValueError("Invalid SMILES")

    # 保持不加 H
    if embed:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if optimize:
        AllChem.UFFOptimizeMolecule(mol, maxIters=maxIters)

    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])

    dist_matrix = cdist(coords, coords, metric='euclidean')
    return dist_matrix #/ dist_matrix.max()

