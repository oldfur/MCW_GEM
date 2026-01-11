import typing
if not hasattr(typing, "NotRequired"):
    from typing_extensions import NotRequired
    typing.NotRequired = NotRequired
import json
from tqdm import tqdm
from mp_api.client import MPRester

import numpy as np
from pymatgen.electronic_structure.dos import Dos
from pymatgen.electronic_structure.core import Spin

API_KEY = "E3tJQOXbTZpaiDUYVOlW4kdm0NDR8X9J"
FILE_PATH = "../crystal_data_final.json"
OUTPUT_PATH = "../crystal_data_final_2.json"


def get_dos_at_fermi_from_data(d):
    """
    从 electronic_structure 路由返回的文档中计算费米能级处的 DOS
    """
    try:
        # 1. 提取核心数据
        # 在 DosData 模型中，d.dos.total 包含了能量和密度的映射
        print("d.dos.total include:", list(d.dos.total.keys())) # total is a dict
        all_spins = list(d.dos.total.keys())
        first_spin = all_spins[0]
        dos_entry = d.dos.total[first_spin]
        print("dos_entry type:", type(dos_entry))
        print("dos_entry attributes:", list(dos_entry.model_fields.keys()))
    except Exception as e:
        # print(f"转换失败: {e}")
        return None

def enrich_advanced_properties():
    # 1. 加载数据
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data_dict = {str(item['material_id']): item for item in data}
    all_ids = list(data_dict.keys())
    BATCH_SIZE = 50 
    
    with MPRester(API_KEY) as mpr:
        for i in tqdm(range(0, len(all_ids), BATCH_SIZE)):
            batch_ids = all_ids[i:i+BATCH_SIZE]
            
            # # --- 1. 补全压电数据 ---
            # try:
            #     # 关键修改：fields 中使用 "total" 而不是之前的长名字
            #     p_docs = mpr.materials.piezoelectric.search(
            #         material_ids=batch_ids, 
            #         fields=["material_id", "total", "e_ij_max"]
            #     )
            #     for p in p_docs:
            #         m_id = str(p.material_id)
            #         # 获取总压电张量矩阵
            #         if p.total is not None:
            #             # 转换为列表以存储到 JSON
            #             data_dict[m_id]["piezo_tensor"] = p.total.tolist() if hasattr(p.total, "tolist") else p.total
            #         # 获取最大压电常数标量
            #         if p.e_ij_max is not None:
            #             data_dict[m_id]["e_ij_max"] = float(p.e_ij_max)
            # except Exception:
            #     pass 

            # --- 2. 补全 DOS 数据 ---
            try:
                d_docs = mpr.materials.electronic_structure.search(
                    material_ids=batch_ids, 
                    fields=["material_id", "dos", "efermi"]
                )
                for d in d_docs:
                    m_id = str(d.material_id)
                    
                    if d.dos is not None:
                        data_dict[m_id]["efermi"] = float(d.efermi)

                        # val = get_dos_at_fermi_from_data(d)
                        # if val is not None:
                        #     data_dict[m_id]["dos_at_fermi"] = val # 费米能级处的 DOS 插值
                            
                        # if i==0:
                        #     # print(f"模型定义的字段: {d.dos.model_fields.keys()}") 
                        #     # 'total', 'elemental', 'orbital', 'magnetic_ordering'
                        #     print(f"Material ID: {m_id}, efermi: {d.efermi}")
                     
            except Exception:
                pass 

            # 每 1000 条保存临时文件
            if i > 0 and i % 1000 == 0:
                with open(FILE_PATH + ".tmp", 'w', encoding='utf-8') as f:
                    json.dump(list(data_dict.values()), f)

    # 3. 最终保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(list(data_dict.values()), f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 数据补全任务完成！")

if __name__ == "__main__":
    enrich_advanced_properties()