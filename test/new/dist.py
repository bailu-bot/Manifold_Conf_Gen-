# dist.py
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType

# ========== 原子/边特征（若找不到你的 Gen39，可回退到轻量实现） ==========
def _fallback_node_edge_features(mol: Chem.Mol):
    """
    简易原子/边特征，确保无外部依赖也能跑：
      - node: [Z one-hot(<=20), degree<=4 one-hot, aromatic, formal_charge, hybridization(one-hot)]
      - edge: [bond_type one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC]
    """
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    N = len(atoms)
    Zs = [min(20, a.GetAtomicNum()) for a in atoms]  # 1..20 之外合并到 20
    Z_oh = np.zeros((N, 21), dtype=float)
    for i, z in enumerate(Zs):
        Z_oh[i, z] = 1.0

    deg_oh = np.zeros((N, 5), dtype=float)  # degree 0..4, >=4 归到4
    fc = np.zeros((N, 1), dtype=float)
    arom = np.zeros((N, 1), dtype=float)
    hyb_kinds = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]
    hyb_oh = np.zeros((N, len(hyb_kinds)), dtype=float)

    for i, a in enumerate(atoms):
        d = min(4, a.GetDegree())
        deg_oh[i, d] = 1.0
        fc[i, 0] = float(a.GetFormalCharge())
        arom[i, 0] = 1.0 if a.GetIsAromatic() else 0.0
        hyb = a.GetHybridization()
        for k, h in enumerate(hyb_kinds):
            if hyb == h:
                hyb_oh[i, k] = 1.0

    node_features = np.concatenate([Z_oh, deg_oh, arom, fc, hyb_oh], axis=1)  # (N, Dn)

    # edges
    idx_u = []
    idx_v = []
    edge_feat = []
    def _bt_one_hot(bt):
        bt_vec = np.zeros(4, dtype=float)  # SINGLE, DOUBLE, TRIPLE, AROMATIC
        if bt == BondType.SINGLE: bt_vec[0] = 1.0
        elif bt == BondType.DOUBLE: bt_vec[1] = 1.0
        elif bt == BondType.TRIPLE: bt_vec[2] = 1.0
        elif bt == BondType.AROMATIC: bt_vec[3] = 1.0
        return bt_vec

    for b in bonds:
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = _bt_one_hot(b.GetBondType())
        idx_u += [i, j]
        idx_v += [j, i]
        edge_feat += [f, f]
    edge_index = np.vstack([idx_u, idx_v])  # (2, E)
    edge_attr = np.array(edge_feat, dtype=float)  # (E, De)

    return node_features, edge_index, edge_attr


# =================== D1: 拓扑+节点特征+边特征（解耦） ===================
def compute_augmented_graph_distance(mol: Chem.Mol, alpha: float = 0.5, beta: float = 1.0, gamma: float = 1.0):
    """
    D_aug = gamma * D_topo + alpha * D_feat + beta * D_edge
      - D_topo: 图 hop 数（每条边 hop=1）
      - D_feat: 原子表征的欧氏距离
      - D_edge: 按边特征范数的最短路
    """
    # 获取原子/边特征
    node_features, edge_index, edge_attr = _fallback_node_edge_features(mol)
    N = node_features.shape[0]

    # 构图：为每条边同时存 hop=1.0 与 w_feat=||edge_feat||2
    G = nx.Graph()
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        e_feat = edge_attr[e]
        G.add_edge(u, v, hop=1.0, w_feat=float(np.linalg.norm(e_feat) + 1e-12))

    # D_topo（hop 计数）
    D_topo = np.zeros((N, N), dtype=float)
    for i in range(N):
        lengths = nx.single_source_shortest_path_length(G, i)  # hop=1 等权
        for j in range(N):
            D_topo[i, j] = lengths.get(j, np.inf)

    # D_edge（边特征范数的最短路）
    D_edge = np.zeros((N, N), dtype=float)
    for i in range(N):
        lengths = nx.single_source_dijkstra_path_length(G, i, weight='w_feat')
        for j in range(N):
            D_edge[i, j] = lengths.get(j, np.inf)

    # D_feat（原子特征欧氏距离）
    D_feat = cdist(node_features, node_features, metric='euclidean')

    # 安全归一化
    def _norm(M):
        M = np.array(M, dtype=float)
        M = np.where(np.isfinite(M), M, 1e6)
        valid = M[M < 1e6]
        s = valid.max() if valid.size > 0 else 1.0
        return M / (s + 1e-12)

    D_topo = _norm(D_topo)
    D_edge = _norm(D_edge)
    D_feat = _norm(D_feat)

    D_aug = gamma * D_topo + alpha * D_feat + beta * D_edge
    D_aug = _norm(D_aug)
    return D_aug


# =================== D2: 原子环境“相似度”改成 Tanimoto 距离 ===================
def compute_AE_tanimoto_distance(mol: Chem.Mol, use_chirality: bool = True, use_features: bool = False, n_bits: int = 256):
    """
    为每个原子构造一个简单的布尔特征向量（元素/度数/芳香/杂化等），
    用 Jaccard/Tanimoto 计算两两“相似度”，输出 1-相似度 作为距离。
    （这是简化版的 AE 距离，作为弱先验足够用。）
    """
    node_features, _, _ = _fallback_node_edge_features(mol)
    # 将实值特征二值化（>0 视为1）
    F = (node_features > 0).astype(int)
    N = F.shape[0]
    # 计算按位交并
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        fi = F[i]
        ai = fi.sum()
        for j in range(N):
            fj = F[j]
            b = fj.sum()
            c = (fi & fj).sum()
            denom = ai + b - c
            sim = (c / denom) if denom > 0 else 0.0
            D[i, j] = 1.0 - sim
    # 归一化
    if D.max() > 0:
        D = D / D.max()
    return D


# =================== RDKit ETKDG 参考 3D 距离（评估/对照用） ===================
def compute_embed3d_distance(mol: Chem.Mol, embed: bool = True, optimize: bool = False, maxIters: int = 200):
    """
    用 ETKDG 生成一个 3D 参考构象，返回坐标欧氏距离矩阵（仅作评估/可视化对照）。
    """
    mol = Chem.Mol(mol)
    if embed:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if optimize:
        AllChem.UFFOptimizeMolecule(mol, maxIters=maxIters)
    conf = mol.GetConformer()
    N = mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(N)], dtype=float)
    D = cdist(coords, coords, metric='euclidean')
    if D.max() > 0:
        D = D / D.max()
    return D
