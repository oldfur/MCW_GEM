import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from typing import List, Callable, Optional
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
"""
这行代码创建了一个 CrystalNN 实例,用于确定晶体结构中原子之间的连接关系(即键);
CrystalNN 是 pymatgen.analysis.local_env 模块中的一种局部环境策略.
它通过分析原子之间的距离、对称性和其他几何特性，动态地确定每个原子的邻居;
这种方法特别适合处理复杂的晶体结构，例如多孔材料或具有非标准键长的晶体。
"""


def get_symmetry_info(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    # 是一个用于分析晶体对称性的类
    # 参数 symprec=tol 指定了对称性分析的容差值（tol）；
    # 较小的容差值会更严格地判断对称性，而较大的容差值允许更大的偏差。
    crystal = spga.get_refined_structure()
    c = pyxtal()    # pyxtal是用于处理基于对称性的晶体结构的类
    try:
        c.from_seed(crystal, tol=0.01)
        """
        from_seed 是 pyxtal 类的一个方法，用于从各种输入（如 CIF 文件、
        Pymatgen Structure 对象等）加载晶体结构。它会解析输入的晶体结构，
        并根据对称性和几何约束将其转换为 pyxtal 的内部表示形式
        """
        # 首先尝试使用较大的容差值 tol=0.01 加载晶体结构。
        # 较大的容差值允许更大的几何偏差
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
        """
        c.atom_sites 是 pyxtal 对象中的一个属性，表示晶体中的所有原子位点。
        site.specie 获取当前位点的原子种类（例如，"H" 表示氢原子，"O" 表示氧原子）。
        len(matrices) 用于记录当前对称操作矩阵的数量,作为锚点anchor,以标记该原子位点的起始位置。
        site.position 获取当前位点的分数坐标（fractional coordinates）,表示原子在晶格中的位置。
        """
        for syms in site.wp:
            species.append(specie)
            matrices.append(syms.affine_matrix)
            coords.append(syms.operate(coord))
            anchors.append(anchor)
            """
            site.wp 是当前位点的 Wyckoff 位点信息，包含一组对称操作。
            syms.affine_matrix 是对称操作的仿射矩阵，表示晶体对称性变换的数学描述。
            syms.operate(coord) 对当前位点的坐标 coord 应用对称操作，生成新的原子坐标
            """
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
    """
    c.lattice.get_para(degree=True) 提供晶格参数,包括晶格的边长(a, b, c)和
    角度(alpha, beta, gamma); Lattice.from_parameters 使用这些晶格参数构建晶
    格对象：它将晶格参数（边长和角度）转换为直角坐标系下的晶格向量,便于描述晶体的几何形状。
    """
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
    # 使用 Structure.from_str 方法从 CIF 字符串中解析晶体结构。
    # 参数 fmt="cif" 指定输入字符串的格式为 CIF。
    # 返回的 crystal 是一个 Structure 对象，包含晶体的晶格、原子种类和坐标等信息 

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()   # Niggli还原算法，标准化晶格

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        # crystal.lattice.parameters 提供了晶体的晶格参数，包括晶格的边长（a, 
        # b, c）和角度（alpha, beta, gamma)
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False, # 指定输入的原子坐标是分数坐标，而不是笛卡尔坐标（Cartesian)
    )   
    # match is gaurantteed because cif only uses lattice params & frac_coords
    # assert canonical_crystal.matches(crystal)

    """
    通过重新构建晶格和传递原子信息，这段代码生成了一个标准化的晶体结构对象canonical_crystal
    标准化的晶体结构确保了晶格参数和原子坐标的一致性，便于后续的比较、存储和分析
    """
    return canonical_crystal


def build_crystal_graph(crystal, graph_method="crystalnn"):
    """"""

    if graph_method == "crystalnn":
        try:
            # crystal_graph = StructureGraph.from_local_env_strategy(crystal, CrystalNN)
            crystal_graph = StructureGraph.with_local_env_strategy(crystal, CrystalNN)
            """
            StructureGraph 是 pymatgen.analysis.graphs 模块中的一个类,用于表示晶体的图结构。
            在晶体图中,节点(nodes)表示晶体中的原子。
            from_local_env_strategy 是 StructureGraph 的一个类方法，用于基于局部环境策略构建晶体图。
            """
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
    """
    crystal.lattice.parameters 提供了晶体的晶格参数，包括：
    lengths: 晶格的三个边长:a, b, c。
    angles: 晶格的三个夹角:alpha, beta, gamma,单位为度。
    通过切片操作, lengths 提取了前 3 个参数(边长),angles 提取了后 3 个参数(角度)
    """
    assert np.allclose(crystal.lattice.matrix, lattice_params_to_matrix(*lengths, *angles))
    """
    crystal.lattice.matrix 是一个 3x3 的矩阵,表示晶体的晶格向量。
    lattice_params_to_matrix(*lengths, *angles) 将长度和角度转换为矩阵
    np.allclose 用于检查两个数组是否在给定的容差范围内相等
    """

    edge_indices, to_jimages = [], []
    if graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            """
            crystal_graph.graph.edges(data="to_jimage") 遍历晶体图中的所有边。
            i: 边的起始节点索引。
            j: 边的结束节点索引。
            to_jimage: 边的周期性边界条件信息，通常是一个三元组(如 (0, 0, 1))，表示边跨越晶胞的方向。
            """
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
    # 建立晶体对象
    result_dict = {}
    if use_space_group:
        # crystal, sym_info = get_symmetry_info(crystal, tol=tol)
        _, sym_info = get_symmetry_info(crystal, tol=tol)  # do not modify crystal
        result_dict.update(sym_info)
    else:
        result_dict["spacegroup"] = 1
    graph_arrays = build_crystal_graph(crystal, graph_method)
    properties = {k: row[k] for k in prop_list if k in row.keys()}
    # prop_list 是一个包含属性名称的列表，表示需要从 row 中提取的属性。
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
    df = pd.read_csv(input_file)    # 读取原始文件

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],   
        # 通过列表推导式生成了一个列表，其中每个元素是数据框df的一行（df.iloc[idx]）
        [niggli] * len(df), 
        [primitive] * len(df),
        [graph_method] * len(df),
        [prop_list] * len(df),
        [use_space_group] * len(df),
        [tol] * len(df),
        # 将这些参数重复len(df)次，确保每个函数调用都能接收到正确的参数
        # 生成了与数据框 df 行数相同长度的列表，每个列表的元素都是相同的值
        num_cpus=num_workers,
    )
    # p_umap函数将指定的函数（这里是process_one）应用到多个输入数据上，并利用多核CPU加速计算

    mpid_to_results = {result["mp_id"]: result for result in unordered_results}
    ordered_results = [mpid_to_results[df.iloc[idx]["material_id"]] for idx in range(len(df))]
    # 按照原始数据框的material_id的顺序，重新排列结果

    return ordered_results


"""
    mp20_dataset = MP20(root=self.hparams.datasets.mp20.root)
    self.mp20_train_dataset = mp20_dataset[:27138]
    self.mp20_val_dataset = mp20_dataset[27138 : 27138 + 9046]
    self.mp20_test_dataset = mp20_dataset[27138 + 9046 :]
"""

if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF*")
    # 忽略与 CIF 解析相关的警告信息

    parser = argparse.ArgumentParser(description="Process cif files.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="./raw/all.csv",
        help="Input file containing cif strings.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="mp20_processed.csv",
        help="Output file to save processed data.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for parallel processing."
    )
    parser.add_argument(
        "--niggli", action="store_true", help="Use Niggli reduction."
    )
    parser.add_argument(
        "--primitive", action="store_true", help="Use primitive cell."
    )
    parser.add_argument(
        "--graph_method",
        type=str,
        default="crystalnn",
        choices=["crystalnn", "none"],
        help="Method to build crystal graph.",
    )
    parser.add_argument(
        "--prop_list",
        type=str,
        default="",
        help="Comma-separated list of properties to include.",
    )
    parser.add_argument(
        "--use_space_group",
        action="store_true",
        help="Use space group information.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="Tolerance for symmetry analysis.",
    )
    args = parser.parse_args()
    prop_list = args.prop_list.split(",") if args.prop_list else []
    # 将命令行参数解析为变量
    # prop_list 是一个包含属性名称的列表，表示需要从 row 中提取的属性。
    # 如果没有提供属性列表，则 prop_list 为空列表
    results = preprocess(
        args.input_file,
        args.num_workers,
        args.niggli,
        args.primitive,
        args.graph_method,
        prop_list,
        args.use_space_group,
        args.tol,
    )
    # 调用 preprocess 函数进行数据处理
    
    for i, data_dict in enumerate(results):
            # extract attributes from data_dict
            graph_arrays = data_dict["graph_arrays"]
            atom_types = graph_arrays["atom_types"]
            frac_coords = graph_arrays["frac_coords"]
            cell = graph_arrays["cell"]
            lattices = graph_arrays["lattices"]
            lengths = graph_arrays["lengths"]
            angles = graph_arrays["angles"]
            num_atoms = graph_arrays["num_atoms"]

            # normalize the lengths of lattice vectors, which makes
            # lengths for materials of different sizes at same scale
            _lengths = lengths / float(num_atoms) ** (1 / 3)
            # convert angles of lattice vectors to be in radians
            _angles = np.radians(angles)
            # add scaled lengths and angles to graph arrays
            graph_arrays["length_scaled"] = _lengths
            graph_arrays["angles_radians"] = _angles
            graph_arrays["lattices_scaled"] = np.concatenate([_lengths, _angles])

            data = Data(
                id=data_dict["mp_id"],
                atom_types=torch.LongTensor(atom_types),
                frac_coords=torch.Tensor(frac_coords),
                cell=torch.Tensor(cell).unsqueeze(0),
                lattices=torch.Tensor(lattices).unsqueeze(0),
                lattices_scaled=torch.Tensor(graph_arrays["lattices_scaled"]).unsqueeze(0),
                lengths=torch.Tensor(lengths).view(1, -1),
                lengths_scaled=torch.Tensor(graph_arrays["length_scaled"]).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                angles_radians=torch.Tensor(graph_arrays["angles_radians"]).view(1, -1),
                num_atoms=torch.LongTensor([num_atoms]),
                num_nodes=torch.LongTensor([num_atoms]),  # special attribute used for PyG batching
                token_idx=torch.arange(num_atoms),
                edge_index=torch.LongTensor(graph_arrays["edge_indices"]),
            )

            # 3D coordinates (NOTE do not zero-center prior to graph construction)
            # 计算晶体中每个原子的笛卡尔坐标,通过将原子的分数坐标(frac_coords)与晶胞矩阵(cell)相乘得到
            data.pos = torch.einsum(
                "bi,bij->bj",
                data.frac_coords,
                torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
            )
            
            # space group number
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])

            if i == 0:
                print(f"Processing {i + 1}/{len(results)}: {data_dict['mp_id']}")
                print("id: ", data.id)
                print("atom_types: ", data.atom_types)  # !!!
                print("frac_coords: ", data.frac_coords)
                print("cell: ", data.cell)
                print("lattices: ", data.lattices)
                print("lattices_scaled: ", data.lattices_scaled)
                print("lengths: ", data.lengths)
                print("lengths_scaled: ", data.lengths_scaled)
                print("angles: ", data.angles)
                print("angles_radians: ", data.angles_radians)
                print("num_atoms: ", data.num_atoms)    # !!!
                print("num_nodes: ", data.num_nodes)
                print("token_idx: ", data.token_idx)
                print("pos: ", data.pos)    # !!!
                print("spacegroup: ", data.spacegroup)
                print("edge_index: ", data.edge_index)  # !!!


                batch = data # 暂时默认batch_size为1
                """
                charges: 就是data.atom_types

                atom_mask和edge_mask:

                atom_mask = (data.atom_types).unsqueeze(0) > 0      # [1, 8]
                batch['atom_mask'] = atom_mask
                # Obtain edges
                batch_size, n_nodes = atom_mask.size()
                edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
                # mask diagonal
                diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
                edge_mask *= diag_mask
                batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)


                one_hot: 
                all_species = dataset['charges'].unique(sorted=True)
                data['one_hot'] = data['charges'].unsqueeze(-1) == all_species.unsqueeze(0).unsqueeze(0)
                """


                """
                输出: 
                id:  mp-10009
                atom_types:  tensor([31, 31, 31, 31, 52, 52, 52, 52])   # 用这个
                frac_coords:  tensor([[0.3333, 0.6667, 0.1830],
                                      [0.6667, 0.3333, 0.6830],
                                      [0.6667, 0.3333, 0.8170],
                                      [0.3333, 0.6667, 0.3170],
                                      [0.3333, 0.6667, 0.8862],
                                      [0.6667, 0.3333, 0.3862],
                                      [0.6667, 0.3333, 0.1138],
                                      [0.3333, 0.6667, 0.6138]])  # [8, 3]
                cell:  tensor([[[ 4.1346e+00,  0.0000e+00,  2.5317e-16],
                                [-2.0673e+00,  3.5807e+00,  2.5317e-16],
                                [ 0.0000e+00,  0.0000e+00,  1.8426e+01]]])
                lattices:  tensor([[  4.1346,   4.1346,  18.4256,  90.0000,  90.0000, 120.0000]])
                lattices_scaled:  tensor([[2.0673, 2.0673, 9.2128, 1.5708, 1.5708, 2.0944]])
                lengths:  tensor([[ 4.1346,  4.1346, 18.4256]])
                lengths_scaled:  tensor([[2.0673, 2.0673, 9.2128]])
                angles:  tensor([[ 90.0000,  90.0000, 120.0000]])
                angles_radians:  tensor([[1.5708, 1.5708, 2.0944]])
                num_atoms:  tensor([8]) # 用这个
                num_nodes:  tensor([8])
                token_idx:  tensor([0, 1, 2, 3, 4, 5, 6, 7])
                pos:  tensor([[4.7684e-07, 2.3871e+00, 3.3711e+00],
                              [2.0673e+00, 1.1936e+00, 1.2584e+01],
                              [2.0673e+00, 1.1936e+00, 1.5054e+01],
                              [4.7684e-07, 2.3871e+00, 5.8417e+00],
                              [4.7684e-07, 2.3871e+00, 1.6329e+01],
                              [2.0673e+00, 1.1936e+00, 7.1160e+00],
                              [2.0673e+00, 1.1936e+00, 2.0968e+00],
                              [4.7684e-07, 2.3871e+00, 1.1310e+01]])    # [8, 3], 用这个
                spacegroup:  tensor([1])
                """


            # if self.pre_filter is not None and not self.pre_filter(data):
            #     continue
            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)

            # data_list.append(data)

    # 将处理后的结果保存到 CSV 文件中
    # df = pd.DataFrame(results)
    # df.to_csv(args.output_file, index=False)
    # index=False 表示不保存行索引
    # 处理完成后，输出处理后的数据框到指定的 CSV 文件中
    # print(f"Processed data saved to {args.output_file}")
    # 输出处理完成的信息

    """
    需要先得到loader,然后才能得到data:
    preprocess = PreprocessMP20(load_charges=cfg.include_charges)
    mp20_loader = DataLoader(dataset, batch_size=batch_size,                  
            shuffle=args.shuffle if (split == 'train') else False, num_workers=num_workers,
            collate_fn=preprocess.collate_fn, drop_last=True)
    # 预处理最终生成分子图的函数是preprocess.collate_fn



    目标能够处理成以下形式：

    for i, data in enumerate(loader):
        # if i >= 3:
        #     break
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)




    batch_size, 简写B: 16
    num_atoms,简写N: 1~29都有可能,一个batch里面的最大可能分子原子数是29
    epoch: 0, i: 0, x shape: torch.Size([B, N, 3]), 
    node_mask shape: torch.Size([B, N, 1]), 
    edge_mask shape: torch.Size([B*N*N, 1]),
    one_hot shape: torch.Size([B, N, 5]), 对应qm9中的5种原子类型的one-hot编码
    这里的one-hot编码是针对每个原子类型的,而不是针对每个分子
    charges shape: torch.Size([B, N, 1])
    """