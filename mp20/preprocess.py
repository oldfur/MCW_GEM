import numpy as np
import pandas as pd
from p_tqdm import p_umap
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis import local_env
from pyxtal import pyxtal


CrystalNN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

def get_symmetry_info(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()  
    try:
        c.from_seed(crystal, tol=0.01)

    except:
        c.from_seed(crystal, tol=0.0001)
    space_group = c.group.number
    species = []
    anchors = []
    matrices = []
    coords = []
    for site in c.atom_sites:
        specie = site.specie
        anchor = len(matrices)
        coord = site.position

        for syms in site.wp:
            species.append(specie)
            matrices.append(syms.affine_matrix)
            coords.append(syms.operate(coord))
            anchors.append(anchor)

    anchors = np.array(anchors)
    matrices = np.array(matrices)
    coords = np.array(coords) % 1.0
    sym_info = {"anchors": anchors, "wyckoff_ops": matrices, "spacegroup": space_group}
    crystal = Structure(
        lattice=Lattice.from_parameters(*np.array(c.lattice.get_para(degree=True))),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )
    
    return crystal, sym_info


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trigonometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.

    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt="cif")    

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )   
    # match is gaurantteed because cif only uses lattice params & frac_coords
    # assert canonical_crystal.matches(crystal)

    return canonical_crystal


def build_crystal_graph(crystal, graph_method="crystalnn"):
    """Build crystal graph from crystal structure."""
    if graph_method == "crystalnn":
        try:
            # crystal_graph = StructureGraph.from_local_env_strategy(crystal, CrystalNN)
            crystal_graph = StructureGraph.with_local_env_strategy(crystal, CrystalNN)
        except:
            crystalNN_tmp = local_env.CrystalNN(
                distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=10
            )
            # crystal_graph = StructureGraph.from_local_env_strategy(crystal, crystalNN_tmp)
            crystal_graph = StructureGraph.with_local_env_strategy(crystal, crystalNN_tmp)
    elif graph_method == "none":
        pass
    else:
        raise NotImplementedError

    cell = crystal.lattice.matrix
    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers

    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix, lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return {
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "cell": cell,
        "lattices": lattice_parameters,
        "lengths": lengths,
        "angles": angles,
        "edge_indices": edge_indices,
        "to_jimages": to_jimages,
        "num_atoms": num_atoms,
    }


def process_one(row, niggli, primitive, graph_method, prop_list, use_space_group=False, tol=0.01):
    crystal_str = row["cif"]
    crystal = build_crystal(crystal_str, niggli=niggli, primitive=primitive)
 
    result_dict = {}
    if use_space_group:
        # crystal, sym_info = get_symmetry_info(crystal, tol=tol)
        _, sym_info = get_symmetry_info(crystal, tol=tol)  # do not modify crystal
        result_dict.update(sym_info)
    else:
        result_dict["spacegroup"] = 1
    graph_arrays = build_crystal_graph(crystal, graph_method)
    properties = {k: row[k] for k in prop_list if k in row.keys()}

    result_dict.update(
        {"mp_id": row["material_id"], "cif": crystal_str, "graph_arrays": graph_arrays}
    )
    result_dict.update(properties)
    return result_dict


def preprocess(
    input_file,
    num_workers,
    niggli,
    primitive,
    graph_method,
    prop_list,
    use_space_group=False,
    tol=0.01,
):
    df = pd.read_csv(input_file) 

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],   
        [niggli] * len(df), 
        [primitive] * len(df),
        [graph_method] * len(df),
        [prop_list] * len(df),
        [use_space_group] * len(df),
        [tol] * len(df),
        num_cpus=num_workers,
    )

    mpid_to_results = {result["mp_id"]: result for result in unordered_results}
    ordered_results = [mpid_to_results[df.iloc[idx]["material_id"]] for idx in range(len(df))]

    return ordered_results
