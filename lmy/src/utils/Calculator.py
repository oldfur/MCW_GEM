import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress
from torch_geometric.data import Data

class HTGP_Calculator(Calculator):
    """
    完全适配 PotentialTrainer 训练逻辑的 ASE Calculator
    """
    implemented_properties = ['energy', 'forces', 'stress', 'descriptors', 'weights']
    def __init__(self, model, cutoff=6.0, device='cpu', **kwargs):
        """
        :param model: 你的 HTGPModel 实例
        :param cutoff: 必须与训练时的 cutoff 一致
        :param device: 'cpu' or 'cuda'
        """
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.cutoff = cutoff
        self.device = torch.device(device)
        
        # 1. 模型设置
        self.model.to(self.device)
        self.model.eval() # 必须是 eval 模式
        
        self.capture_weights = kwargs.get("capture_weights", False)
        self.capture_descriptors = kwargs.get("capture_descriptors", False)

        # 冻结模型参数（权重），只对输入求导
        for param in self.model.parameters():
            param.requires_grad = False

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # -----------------------------------------------------------
        # 1. 数据准备 (Data Preparation)
        # -----------------------------------------------------------
        data = self._atoms_to_pyg_data(atoms)
        
        original_pos = data.pos
        original_cell = getattr(data, 'cell', None)
        displacement = None # 初始化为 None，防止报错
        
        # 判断是否需要计算应力
        is_periodic = atoms.pbc.any()
        calc_stress = 'stress' in properties and is_periodic

        # 开启梯度（这是必须的，否则无法反向传播求 Force）
        data.pos.requires_grad = True

        if calc_stress:
            # === 分支 A: 需要算应力 (构建变形图) ===
            
            # 构造虚拟应变
            displacement = torch.zeros((1, 3, 3), dtype=data.pos.dtype, device=self.device)
            displacement.requires_grad = True
            symmetric_strain = 0.5 * (displacement + displacement.transpose(-1, -2))
            strain_on_graph = symmetric_strain[0]
            
            # ⚠️ 关键：坐标变形
            # 这里 data.pos 被替换为计算图的一个节点（pos + strain）
            pos_deformed = original_pos + torch.matmul(original_pos, strain_on_graph.T)
            data.pos = pos_deformed
            
            # 晶胞变形 (如果有)
            if original_cell is not None:
                cell_deformed = original_cell + torch.matmul(original_cell, symmetric_strain)
                data.cell = cell_deformed
                
            # 记录求导目标：原始坐标 和 应变
            inputs_to_grad = [original_pos, displacement]
            
        else:
            # === 分支 B: 不需要算应力 (保持纯净) ===
            
            # ⚠️ 关键：不进行任何变形操作
            # data.pos 保持为原始的叶子节点 (Leaf Node)
            # 求导目标直接就是 data.pos
            inputs_to_grad = [data.pos]
            
            # 显式重置 cell 为原始值（防止被污染），仅用于 volume 计算
            data.cell = original_cell
        # -----------------------------------------------------------
        # 3. 前向传播 (Forward)
        # -----------------------------------------------------------
        energy = self.model(data, capture_weights=self.capture_weights, capture_descriptors=self.capture_descriptors)         
        # -----------------------------------------------------------
        # 4. 结果提取与单位转换, 加上atom_ref (Results & Units)
        # -----------------------------------------------------------
        self.results['energy'] = energy.item()

        # -----------------------------------------------------------
        # 5. 反向传播求导 (Backward)
        # -----------------------------------------------------------
        # 我们需要对 original_pos 和 displacement 求导
        inputs_to_grad = [original_pos]
        if calc_stress:
            inputs_to_grad.append(displacement)
            
        grads = torch.autograd.grad(
            outputs=energy,
            inputs=inputs_to_grad,
            retain_graph=False,
            create_graph=False # 推理时不需要二阶导
        )
        
        # --- Force ---
        # F = -dE/dx
        forces = -grads[0]
        self.results['forces'] = forces.detach().cpu().numpy()
        # --- Stress ---
        if calc_stress:
            dE_dStrain = grads[1] # (1, 3, 3)
            
            # 你的训练代码逻辑: pred_stress = dE_dStrain / vol
            # volume calculation
            volume = torch.abs(torch.det(original_cell[0]))
            
            if volume > 1e-8:
                stress_tensor = dE_dStrain / volume
                stress_np = stress_tensor.squeeze(0).detach().cpu().numpy()
                self.results['stress'] = full_3x3_to_voigt_6_stress(stress_np)
            else:
                self.results['stress'] = np.zeros(6)
        if self.capture_weights:
            self.results['weights'] = self._get_weights()
            
        if self.capture_descriptors:
            self.results['descriptors'] = self._get_descriptors()

        if self.get_charges:
            self.results['charges'] = self._get_charges()

    def _get_weights(self):
        """
        从模型各层的 PhysicsGating 模块中提取捕获的权重。
        返回: List[Dict]，列表索引对应层数
        """
        weights_per_layer = []
        
        # 遍历模型的每一层
        for i, layer in enumerate(self.model.layers):
            # 'gating' 是 ModuleDict 中的 key，对应 PhysicsGating 实例
            gating_module = layer['gating']
            
            layer_data = {}
            
            # 辅助函数：如果属性存在且不为None，转numpy
            def extract(attr_name):
                if hasattr(gating_module, attr_name):
                    val = getattr(gating_module, attr_name)
                    if val is not None:
                        return val.detach().cpu().numpy()
                return None

            # 提取你在 PhysicsGating 中定义的捕获变量
            layer_data['g0'] = extract('g0_captured')
            layer_data['g1'] = extract('g1_captured')
            layer_data['g2'] = extract('g2_captured')
            layer_data['chem_logits'] = extract('chem_logits_captured')
            layer_data['phys_logits'] = extract('phys_logits_captured')
            layer_data['scalar_basis'] = extract('scalar_basis_captured') # 如有需要可取消注释
            layer_data['p_ij'] = extract('p_ij_captured') # 如有需要可取消注释

            weights_per_layer.append(layer_data)
            
        return weights_per_layer


    def _get_descriptors(self):
        """
        从模型中提取每一层的原子特征 (h0, h1, h2)。
        你的模型代码里已经把它们存到了 self.model.all_layer_descriptors 列表里。
        """
        if not hasattr(self.model, 'all_layer_descriptors'):
            return None
            
        descriptors_numpy = []
        
        # 遍历模型保存的特征列表
        for layer_feats in self.model.all_layer_descriptors:
            layer_dict = {}
            for key, val in layer_feats.items():
                # 模型里已经做了 .detach().cpu()，这里只需要转 numpy
                if val is not None:
                    layer_dict[key] = val.numpy()
                else:
                    layer_dict[key] = None
            descriptors_numpy.append(layer_dict)
            
        return descriptors_numpy


    def _get_charges(self):
        """
        从模型中提取每一层的原子特征 (h0, h1, h2)。
        你的模型代码里已经把它们存到了 self.model.all_layer_descriptors 列表里。
        """
        if not hasattr(self.model, 'all_layer_descriptors'):
            return None
            
        descriptors_numpy = []
        
        # 遍历模型保存的特征列表
        for layer_feats in self.model.all_layer_descriptors:
            layer_dict = {}
            for key, val in layer_feats.items():
                # 模型里已经做了 .detach().cpu()，这里只需要转 numpy
                if val is not None:
                    layer_dict[key] = val.numpy()
                else:
                    layer_dict[key] = None
            descriptors_numpy.append(layer_dict)
            
        return descriptors_numpy


    def _atoms_to_pyg_data(self, atoms):
        """
        转换函数 (保持不变，除了不用压缩数据类型)
        """
        z = torch.from_numpy(atoms.get_atomic_numbers()).to(torch.long).to(self.device)
        pos = torch.from_numpy(atoms.get_positions()).to(torch.float32).to(self.device)
        
        # ASE get_cell returns [a, b, c], shape (3,3)
        # Model expects (1, 3, 3)
        if atoms.pbc.any():
            cell_np = atoms.get_cell().array
            # 再次检查体积，防止 pbc=True 但 cell 是 0 的极端情况
            if np.abs(np.linalg.det(cell_np)) > 1e-6:
                cell = torch.from_numpy(cell_np).to(torch.float32).unsqueeze(0).to(self.device)
            else:
                cell = None
        else:
            # 单分子/非周期体系，强制设为 None
            cell = None

        # Neighbor List
        i_idx, j_idx, _, S_integers = neighbor_list('ijdS', atoms, self.cutoff)
        
        edge_index = torch.tensor(np.vstack((i_idx, j_idx)), dtype=torch.long).to(self.device)
        shifts_int = torch.from_numpy(S_integers).to(torch.float32).to(self.device)
        # print(shifts_int)
        # Batch (ASE always 1 graph)
        num_atoms = len(atoms)
        batch = torch.zeros(num_atoms, dtype=torch.long).to(self.device)
        
        data = Data(
            z=z,
            pos=pos,
            cell=cell,
            edge_index=edge_index,
            shifts_int=shifts_int,
            batch=batch
        )
        # 为 Data 注入 num_graphs 属性，防止模型内部报错
        data.num_graphs = 1
        
        return data