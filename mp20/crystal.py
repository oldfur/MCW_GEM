# 确保这是文件的第一行
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import os
import smact
import itertools
from collections import Counter
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
from smact.screening import pauling_test

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")

class Crystal:
    """Crystal object that holds information about a crystal structure and its validity, including
    PyMatGen Structure object, composition, and fingerprints.

    Adapted from: https://github.com/txie-93/cdvae
    """

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict["frac_coords"]
        self.atom_types = crys_array_dict["atom_types"]
        self.sample_idx = crys_array_dict["sample_idx"]

        self.lengths = crys_array_dict["lengths"].squeeze()
        assert self.lengths.ndim == 1
        self.angles = crys_array_dict["angles"].squeeze()
        assert self.lengths.ndim == 1
        self.dict = crys_array_dict

        self.get_structure()
        if self.constructed:
            self.get_composition()
            self.get_validity()
            self.get_fingerprints()
        else:
            self.valid = False
            self.comp_valid = False
            self.struct_valid = False

    def get_structure(self):
        if min(self.lengths) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
        if (
            np.isnan(self.lengths).any()
            or np.isnan(self.angles).any()
            or np.isnan(self.frac_coords).any()
        ):
            self.constructed = False
            self.invalid_reason = "nan_value"
        # this catches validity failures down the line
        elif (1 > self.atom_types).any() or (self.atom_types > 104).any():
            self.constructed = False
            self.invalid_reason = f"{self.atom_types=} are not with range"
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())
                    ),
                    species=self.atom_types,
                    coords=self.frac_coords,
                    coords_are_cartesian=False,
                )
                self.constructed = True
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = "unrealistically_small_lattice"
            except TypeError:
                self.constructed = False
                self.invalid_reason = f"{self.atom_types=} are not possible"
            except Exception:
                self.constructed = False
                self.invalid_reason = "construction_raises_exception"

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        """具体验证有效性的方法视处理对象而定，可改动！"""
        # Check if the crystal is valid
        # Check if the composition is valid
        # Check if the structure is valid
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [
                CrystalNNFP.featurize(self.structure, i) for i in range(len(self.structure))
            ]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)

    def __repr__(self):
        return f"Crystal(valid={self.valid}, atoms={self.atom_types}, lengths={self.lengths}, angles={self.angles}, idx={self.sample_idx})"


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    """综合考虑了氧化态、电荷中性、电负性和金属性等多重化学合理性标准，是自动化材料筛选和结构生成中常用的有效性判据"""
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    # if len(list(itertools.product(*ox_combos))) > 1e5:
    #     return False
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def array_dict_to_crystal(
    x: dict,
    save: bool = False,
    save_dir_name: str = "",
) -> Crystal:
    """Method to convert a dictionary of numpy arrays to a Crystal object which is compatible with
    StructureMatcher (used for evaluations). Previously called 'safe_crystal', as it return a
    generic crystal if the input is invalid.

    Adapted from: https://github.com/facebookresearch/flowmm

    Args:
        x: Dictionary of numpy arrays with keys:
            - 'frac_coords': Fractional coordinates of atoms.
            - 'atom_types': Atomic numbers of atoms.
            - 'lengths': Lengths of the lattice vectors.
            - 'angles': Angles between the lattice vectors.
            - 'sample_idx': Index of the sample in the dataset.
        save: Whether to save the crystal as a CIF file.
        save_dir_name: Directory to save the CIF file.

    Returns:
        Crystal: Crystal object, optionally saved as a CIF file.
    """
    # Check if the lattice angles are in a valid range
    if np.all(50 < x["angles"]) and np.all(x["angles"] < 130):
        crys = Crystal(x)   # 这一行创建crystal对象，包含有效性
        # Check if the crystal is valid
        if save:    # 如果需要保存
            if crys.valid:
                os.makedirs(save_dir_name, exist_ok=True)
                crys.structure.to(os.path.join(save_dir_name, f"crystal_{x['sample_idx']}.cif"))
                print(f"save to {save_dir_name}!!")
            else:
                print(f"Crystal is not valid, not saving: {crys.invalid_reason}")
    else:
        # returns an absurd crystal
        crys = Crystal(
            {
                "frac_coords": np.zeros_like(x["frac_coords"]),
                "atom_types": np.zeros_like(x["atom_types"]),
                "lengths": 100 * np.ones_like(x["lengths"]),
                "angles": np.ones_like(x["angles"]) * 90,
                "sample_idx": x["sample_idx"],
            }
        )
    return crys


def lattice_matrix(a, b, c, alpha, beta, gamma):
    # 角度转弧度
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    # 三角函数
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    # 基矢构造
    a1 = [a, 0, 0]
    a2 = [b * cos_gamma, b * sin_gamma, 0]

    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = c * np.sqrt(
        1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 +
        2 * cos_alpha * cos_beta * cos_gamma
    ) / sin_gamma

    a3 = [cx, cy, cz]

    return np.array([a1, a2, a3])


# 假设 lattice 为 row-wise 格式（每行一个基矢）
def frac_to_cart(frac_coords, lattice):
    # lattice: 3x3 matrix, row-wise
    return np.dot(frac_coords, lattice)

def cart_to_frac(cart_coords, lattice):
    det = np.linalg.det(lattice)
    if abs(det) < 1e-8:
        print(f"Warning: lattice is nearly singular, det={det:.2e}")
        return np.dot(cart_coords, np.zeros(lattice.shape))
    return np.dot(cart_coords, np.linalg.inv(lattice))



chemical_symbols = [
    # 0
    "X",
    # 1
    "H","He",
    # 2
    "Li","Be","B","C","N","O","F","Ne",
    # 3
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    # 4
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe",
    "Co","Ni","Cu","Zn","Ga","Ge","As","Se",
    "Br","Kr",
    # 5
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru",
    "Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
    "I","Xe",
    # 6
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm",
    "Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
    "Lu","Hf","Ta","W","Re","Os","Ir","Pt",
    "Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    # 7
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu",
    "Am","Cm","Bk","Cf","Es","Fm","Md","No",
    "Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds",
    "Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
]
