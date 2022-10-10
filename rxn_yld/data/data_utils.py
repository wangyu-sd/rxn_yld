# !/usr/bin/python3
# @File: data_utils.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.20.22
import torch
from torch import Tensor
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from functools import cmp_to_key
from rdkit.Chem import rdDistGeom as molDG

BOND_ORDER_MAP = {0: 0, 1: 1, 1.5: 2, 2: 3, 3: 4}


def rxn_dict_process(prod, rct, ctl=None):
    del prod['mol']
    del rct['mol']
    del ctl['mol']

    prod['atom_fea'] = torch.cat([prod['atom_fea'], ctl['atom_fea']], dim=1)
    rct['atom_fea'] = torch.cat([rct['atom_fea'], ctl['atom_fea']], dim=1)
    for key in ['bond_adj', 'dist_adj', 'dist_adj_3d']:
        if prod[key] is not None and rct[key] is not None:
            prod[key] = cat_adj(prod[key], ctl[key])
            rct[key] = cat_adj(rct[key], ctl[key])
    prod['n_atom'] += ctl['n_atom']
    rct['n_atom'] += ctl['n_atom']

    return prod, rct


def cat_adj(m1, m2):
    n1, n2 = m1.size(0), m2.size(0)
    m_new = torch.zeros(n1 + n2, n1 + n2, dtype=m1.dtype)
    m_new[:n1, :n1] = m1
    m_new[n1:, n1:] = m2
    return m_new


class AtomFeaParser:
    def __init__(self):
        self.discrete_max = (
            90,  # 0    max atomic number
            10,  # 1    max total degree
            7,   # 2    max hybrid state number
            10,  # 3    max total hydrogen
            3,   # 4    max aromatic state number
            11,  # 5    max ring state number
            4,   # 6    max chiral tag state number
            20,  # 7    max formal charge number
            4,   # 8    reactant/product/catalyst/<PAD>
            10,   # 9    if in reaction center
        )
        self.num_continuous_fea = 1


def smile_to_mol_info(smile, calc_dist=True, use_3d_info=False, obj=None):
    mol = Chem.MolFromSmiles(smile)
    bond_adj = get_bond_adj(mol)
    dist_adj = get_dist_adj(mol) if calc_dist else None
    dist_adj_3d = get_dist_adj(mol, use_3d_info) if calc_dist else None
    atom_fea, n_atom = get_atoms_info(mol)
    if obj == 'reactant':
        atom_fea[8, :] += 1
        bond_adj += (1 << 7)
    elif obj == 'product':
        atom_fea[8, :] += 2
        bond_adj += (1 << 8)
    elif obj == 'catalyst':
        atom_fea[8, :] += 3
        bond_adj += (1 << 9)

    return {
        "mol": mol,
        "bond_adj": bond_adj,
        "dist_adj": dist_adj,
        "dist_adj_3d": dist_adj_3d,
        "atom_fea": atom_fea,
        "n_atom": n_atom
    }


def get_atoms_info(mol):
    atoms = mol.GetAtoms()
    n_atom = len(atoms)
    atom_fea = torch.zeros(11, n_atom, dtype=torch.half)
    AllChem.ComputeGasteigerCharges(mol)
    for idx, atom in enumerate(atoms):
        atom_fea[0, idx] = atom.GetAtomicNum()
        atom_fea[1, idx] = atom.GetTotalDegree() + 1
        atom_fea[2, idx] = int(atom.GetHybridization()) + 1
        atom_fea[3, idx] = atom.GetTotalNumHs() + 1
        atom_fea[4, idx] = atom.GetIsAromatic() + 1
        for n_ring in range(3, 9):
            if atom.IsInRingSize(n_ring):
                atom_fea[5, idx] = n_ring + 1
                break
        else:
            if atom.IsInRing():
                atom_fea[5, idx] = 10
        atom_fea[6, idx] = int(atom.GetChiralTag()) + 1
        atom_fea[7, idx] = atom.GetFormalCharge() + 9
        atom_fea[10, idx] = atom.GetDoubleProp("_GasteigerCharge")

    atom_fea = torch.nan_to_num(atom_fea)
    return atom_fea, n_atom


def get_bond_order_adj(mol):
    n_atom = len(mol.GetAtoms())
    bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.uint8)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_adj[i, j] = bond_adj[j, i] = BOND_ORDER_MAP[bond.GetBondTypeAsDouble()]
    return bond_adj


def get_bond_adj(mol):
    """
    :param mol: rdkit mol
    :return: multi graph for {
                sigmoid_bond_graph,
                pi_bond_graph,
                2pi_bond_graph,
                aromic_graph,
                conjugate_graph,
                ring_graph,
    }
    """
    n_atom = len(mol.GetAtoms())
    bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.uint8)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        bond_adj[i, j] = bond_adj[j, i] = 1
        if bond_type in [2, 3]:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 1)
        if bond_type == 3:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 2)
        if bond_type == 1.5:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 3)
        if bond.GetIsConjugated():
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 4)
        if bond.IsInRing():
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 5)
    return bond_adj


def get_tgt_adj_order(product, reactants):
    p_idx2map_idx = {}
    for atom in product.GetAtoms():
        p_idx2map_idx[atom.GetIdx()] = atom.GetAtomMapNum()

    map_idx2r_idx = {0: []}

    for atom in reactants.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            map_idx2r_idx[0].append(atom.GetIdx())
        else:
            map_idx2r_idx[atom.GetAtomMapNum()] = atom.GetIdx()

    order = []
    for atom in product.GetAtoms():
        order.append(map_idx2r_idx[p_idx2map_idx[atom.GetIdx()]])

    order.extend(map_idx2r_idx[0])

    return torch.tensor(order, dtype=torch.long)


def atom_cmp(a1, a2):
    an1, an2 = a1.GetAtomicNum(), a2.GetAtomicNum()
    if an1 != an2:
        return an1 - an2
    hy1, hy2 = a1.GetHybridization(), a2.GetHybridization()
    return hy1 - hy2


def get_dist_adj(mol, use_3d_info=False):
    if use_3d_info:
        m2 = Chem.AddHs(mol)
        is_success = AllChem.EmbedMolecule(m2, enforceChirality=False)
        if is_success == -1:
            dist_adj = None
        else:
            AllChem.MMFFOptimizeMolecule(m2)
            m2 = Chem.RemoveHs(m2)
            dist_adj = (-1 * torch.tensor(AllChem.Get3DDistanceMatrix(m2), dtype=torch.float))
    else:
        dist_adj = (-1 * torch.tensor(molDG.GetMoleculeBoundsMatrix(mol), dtype=torch.float))

    return dist_adj


def pad_1d(x, n_max_nodes):
    if not isinstance(x, Tensor):
        raise TypeError(type(x), "is not a torch.Tensor.")
    n = x.size(0)
    new_x = torch.zeros(n_max_nodes).to(x)
    new_x[:n] = x
    return new_x


def pad_adj(x, n_max_nodes):
    if x is None:
        return None
    if not isinstance(x, Tensor):
        raise TypeError(type(x), "is not a torch.Tensor.")
    n = x.size(0)
    assert x.size(0) == x.size(1)
    new_x = torch.zeros([n_max_nodes, n_max_nodes], dtype=x.dtype)
    new_x[:n, :n] = x
    return new_x
