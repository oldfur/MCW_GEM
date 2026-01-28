import os
import typing
if not hasattr(typing, "NotRequired"):
    from typing_extensions import NotRequired
    typing.NotRequired = NotRequired
from mp_api.client import MPRester
from monty.serialization import dumpfn
from pymatgen.core import Element
from monty.serialization import dumpfn, loadfn

# ================= 配置区 =================
API_KEY = "E3tJQOXbTZpaiDUYVOlW4kdm0NDR8X9J" # 替换为我的 API Key
SAVE_FILENAME = "../mp_stable_reference_84el.json.gz" # 使用压缩格式节省空间

# total 84 elements in mp20
ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 
    'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
    'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 
    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Pu'
]
VALID_ELEMENTS = set(ELEMENTS)
BATCH_SIZE = 500  # 每批下载 500 个，防止内存溢出
# ==========================================

def download_84el_reference():
    with MPRester(API_KEY) as mpr:
        print("第一步：获取全数据库稳定材料 ID (绕过 URL 长度限制)...")
        
        # 不再传 elements 参数，只传 is_stable，这样 URL 很短
        docs = mpr.materials.summary.search(
            is_stable=True,
            fields=["material_id"]
        )
        
        all_mids = [doc.material_id for doc in docs]
        total_count = len(all_mids)
        print(f"全库共有 {total_count} 个稳定材料 ID。")

        final_entries = []
        
        print(f"第二步：分批下载并进行本地元素过滤 (每批 {BATCH_SIZE} 条)...")
        for i in range(0, total_count, BATCH_SIZE):
            batch_mids = all_mids[i : i + BATCH_SIZE]
            try:
                # 获取这一批的详细数据
                batch_entries = mpr.get_entries(batch_mids)
                
                # 本地过滤：只保留仅含 84 种元素的条目
                for entry in batch_entries:
                    # 检查该条目的所有元素是否都在你的 84 元素清单中
                    entry_elements = {str(el) for el in entry.composition.elements}
                    if entry_elements.issubset(VALID_ELEMENTS):
                        final_entries.append(entry)
                
                print(f"已处理: {i + len(batch_entries)} / {total_count} | 当前符合条件数: {len(final_entries)}", end="\r")
            except Exception as e:
                print(f"\n[跳过] 批次 {i} 下载异常: {e}")

        # 第三步：保存
        print(f"\n\n第三步：正在保存符合要求的 {len(final_entries)} 条数据...")
        dumpfn(final_entries, SAVE_FILENAME)
        print(f"--- 任务成功完成 ---")
        print(f"本地文件已生成: {SAVE_FILENAME}")

if __name__ == "__main__":
    download_84el_reference()