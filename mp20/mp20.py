import os
import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from typing import List, Callable, Optional
import numpy as np
from mp20.preprocess import preprocess


class MP20(InMemoryDataset):
    """The MP20 dataset from Materials Project, as a PyG InMemoryDataset.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["all.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["mp20.pt"]

    def download(self) -> None:
        raise NotImplementedError(
            f"Manually download the dataset and place it at {self.root}/raw."
        )

    def process(self) -> None:
        if os.path.exists(os.path.join(self.root, "processed/all_ori.pt")):
            cached_data = torch.load(os.path.join(self.root, "processed/all_ori.pt"))
        else:
            cached_data = preprocess(
                os.path.join(self.root, "raw/all.csv"),
                niggli=True,
                primitive=False,
                graph_method="crystalnn",
                prop_list=["formation_energy_per_atom", "band_gap", "e_above_hull"],    # 3种理化性质
                use_space_group=True,
                tol=0.1,
                num_workers=32,
            )
            torch.save(cached_data, os.path.join(self.root, "all_ori.pt"))

        data_list = []

        # print(cached_data[0].keys())
        # dict_keys(['anchors', 'wyckoff_ops', 'spacegroup', 'mp_id', 'cif', 'graph_arrays', 'formation_energy_per_atom'])
        # print(cached_data[0]["graph_arrays"].keys())
        # dict_keys(['atom_types', 'frac_coords', 'cell', 'lattices', 'lengths', 'angles', 'edge_indices', 'to_jimages', 'num_atoms'])

        for data_dict in cached_data:
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
                atom_types=torch.LongTensor(atom_types), #####
                atom_types_onehot=torch.nn.functional.one_hot(
                    torch.LongTensor(atom_types), num_classes=100   # 初步one-hot编码,最大原子序数为100
                ).float(),  #####
                frac_coords=torch.Tensor(frac_coords),
                cell=torch.Tensor(cell).unsqueeze(0),
                lattices=torch.Tensor(lattices).unsqueeze(0),
                lattices_scaled=torch.Tensor(graph_arrays["lattices_scaled"]).unsqueeze(0),
                lengths=torch.Tensor(lengths).view(1, -1),
                lengths_scaled=torch.Tensor(graph_arrays["length_scaled"]).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                angles_radians=torch.Tensor(graph_arrays["angles_radians"]).view(1, -1),
                num_atoms=torch.LongTensor([num_atoms]),  #####
                num_nodes=torch.LongTensor([num_atoms]),  #####
                token_idx=torch.arange(num_atoms),
                edge_index=torch.LongTensor(graph_arrays["edge_indices"]).T,  #####
                propertys={
                    "formation_energy_per_atom": torch.tensor(data_dict["formation_energy_per_atom"]),
                    "band_gap": torch.tensor(data_dict["band_gap"]),
                    "e_above_hull": torch.tensor(data_dict["e_above_hull"]),
                },  #####
            )

            # print(data.atom_types_onehot.shape)
            # print(data.edge_index.shape)
            # 3D coordinates (NOTE do not zero-center prior to graph construction)
            # 原子笛卡尔坐标由原子分数坐标(frac_coords)与晶胞矩阵(cell)相乘得到
            data.pos = torch.einsum(
                "bi,bij->bj",
                data.frac_coords,
                torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
            )   #####
 
            # space group number
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # torch.save(data_list, os.path.join(self.processed_dir, 'mp20.pt'))
        self.save(data_list, os.path.join(self.root, "processed/mp20.pt"))

if __name__ == "__main__":
    # 运行该文件时，当前目录为UniGEM/mp20/
    # 命令行运行：python mp20.py

    from torch_geometric.loader import DataLoader
    import warnings
    from collections import Counter

    # 忽略与 CIF 解析相关的警告信息
    warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF*")
    # 实例化数据集
    dataset = MP20(root="./")   # 在unigem主目录下运行

    # 构建 DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    atom_types = set([])
    element_symbols = [
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi",        # Skipping some (e.g., 84–88)
    "Ac", "Th", "Pa", "U",  "Np", "Pu"
]   
    atom_count = []    # 统计每个样本的原子数

    # 迭代使用,获取一些mp20的统计信息
    for batch in tqdm(loader, desc="Loading batches"):

        cur_atom_types = set(batch.atom_types.unique().tolist())
        atom_types = atom_types.union(cur_atom_types)
        atom_count = atom_count + batch.num_atoms.tolist()
        
        # print(batch)
        # print(batch.batch)
        # print(batch.edge_index)
        # print(batch.propertys)  # 打印每个样本的理化性质
        # print(batch.propertys['formation_energy_per_atom']) # 字典式访问
        # break
        """
        (batch_size = 32)

        batch: 
        DataBatch(edge_index=[2, 2074], id=[32], atom_types=[280], atom_types_onehot=[280, 100],
            frac_coords=[280, 3], cell=[32, 3, 3], lattices=[32, 6], lattices_scaled=[32, 6],
            lengths=[32, 3], lengths_scaled=[32, 3], angles=[32, 3], angles_radians=[32, 3],
            num_atoms=[32], token_idx=[280],
            propertys={
                formation_energy_per_atom=[32],
                band_gap=[32],
                e_above_hull=[32],
            },
            pos=[280, 3], spacegroup=[32], num_nodes=[1], batch=[280], ptr=[33]
        )

        
        batch.batch:
        tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,
         6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,
         9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12,
        12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15,
        16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
        18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19,
        19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
        22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
        25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
        26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28,
        28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31])


        batch.edge_index:
        tensor([[ 11,   0,  11,  ..., 260, 261, 260],
        [  0,  11,   0,  ..., 261, 260, 261]])


        batch.propertys:
        {
        'formation_energy_per_atom': tensor(
                [-0.0448, -0.7831, -0.4047, -0.2193, -2.3317, -0.9138, -2.5739, -2.1768,
                -0.2267, -1.9609, -1.9334, -0.0753, -0.4097, -0.6964, -0.3284, -0.7491,
                -0.8485, -1.8593, -0.8017, -3.3351, -0.3071, -0.0757, -2.2856, -0.5970,
                -1.0132, -0.8006,  0.0157, -2.2841, -0.7595, -1.9112, -0.6293, -0.6185],
                dtype=torch.float64), 

        'band_gap': tensor(
                [0.0000, 0.0000, 0.0000, 0.0000, 3.1315, 0.0000, 0.0000, 0.0000, 0.0000,
                2.7735, 2.9553, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.6041,
                0.0000, 5.8930, 0.0000, 0.0000, 0.0000, 0.7946, 0.0000, 0.8691, 0.0000,
                4.6260, 0.0000, 0.0000, 0.0000, 0.8037], 
                dtype=torch.float64), 

        'e_above_hull': tensor(
                [0.0000, 0.0000, 0.0448, 0.0256, 0.0000, 0.0000, 0.0265, 0.0183, 0.0000,
                0.0014, 0.0766, 0.0000, 0.0015, 0.0000, 0.0000, 0.0384, 0.0174, 0.0000,
                0.0000, 0.0597, 0.0204, 0.0572, 0.0000, 0.0373, 0.0000, 0.0163, 0.0157,
                0.0000, 0.0000, 0.0192, 0.0000, 0.0555], 
                dtype=torch.float64)
        }
        """
        

    # 构建字典：元素符号 -> 原子序号
    symbol_to_number = {
        element_symbols[i-1]: i-1
        for i in atom_types
        if i <= len(element_symbols)
    }
    print("atom_types: ", atom_types)  # 打印mp20中所有存在的原子类型
    print("symbol_to_number: ", symbol_to_number)  # 打印mp20中所有存在的原子类型映射字典
    print("symbol.keys(): ", symbol_to_number.keys())  # 打印mp20中所有存在的原子类型
    
    hist = dict(Counter(atom_count))   # 用 Counter 得到分布直方图
    print({'n_nodes': hist})
    
