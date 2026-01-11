import typing
if not hasattr(typing, "NotRequired"):
    from typing_extensions import NotRequired
    typing.NotRequired = NotRequired
import json
from tqdm import tqdm
from mp_api.client import MPRester


# 配置参数
API_KEY = "E3tJQOXbTZpaiDUYVOlW4kdm0NDR8X9J"
SAVE_PATH = "../crystal_data_20_atoms.json"
MAX_SITES = 20  # 限制原子数不超过20
BATCH_SIZE = 100  # 每批处理 100 个 ID，防止请求过大或超时


def fetch_batch_data():
    all_combined_data = []
    
    with MPRester(API_KEY) as mpr:
        print(f"1. 正在检索原子数 <= {MAX_SITES} 的基础材料数据...")
        # 获取基础数据
        summary_docs = mpr.materials.summary.search(
            num_sites=(1, MAX_SITES),
            fields=[
                "material_id", "formula_pretty", "structure", 
                "bulk_modulus", "shear_modulus", "density", 
                "energy_above_hull", "band_gap", "nsites"
            ]
        )
        
        # 提取所有 ID
        all_ids = [str(d.material_id) for d in summary_docs]
        # 为了演示，我们处理前 500 个，你可以根据需要修改
        # target_ids = all_ids[:500]
        target_ids = all_ids  # 若需全部材料，取消注释此行 
        print(f"找到总计 {len(all_ids)} 个材料，准备深入采集前 {len(target_ids)} 个的详细性质...")

        # 将 summary 数据转为字典方便后续合并
        summary_dict = {str(d.material_id): d for d in summary_docs}

        # 2. 分批获取高阶性质 (Piezo 和 DOS)
        for i in range(0, len(target_ids), BATCH_SIZE):
            batch_ids = target_ids[i : i + BATCH_SIZE]
            print(f"\n正在处理批次 {i//BATCH_SIZE + 1} ({len(batch_ids)} 个材料)...")

            # --------------------
            # 批量获取压电数据
            # --------------------
            piezo_lookup = {}      
            try:
                # 保持 mpr.materials.piezoelectric.search 路径
                piezo_results = mpr.materials.piezoelectric.search(material_ids=batch_ids)
                for p in piezo_results:
                    # 有些版本使用 p.total_piezo，有些需要 p.data.total_piezo
                    # 使用 getattr 安全获取
                    val = getattr(p, 'total_piezo', None)
                    if val is None and hasattr(p, 'data'):
                        val = getattr(p.data, 'total_piezo', None)
                    
                    piezo_lookup[str(p.material_id)] = val
            except Exception as e:
                print(f"警告: 批量获取压电数据失败: {e}")

            # 修正后的批量获取 DOS 逻辑
            dos_lookup = {}
            try:
                # 使用正确的子模块路径：mpr.materials.electronic_structure
                # 注意：并不是所有材料都有 DOS 数据，这里返回的是 ElectronicStructureSummaryDoc
                es_docs = mpr.materials.electronic_structure.search(material_ids=batch_ids)
                
                for d in es_docs:
                    # 获取费米能级 (efermi)
                    efermi = getattr(d, 'efermi', None)
                    
                    # 获取完整的 DOS 对象（这需要从数据存储中动态拉取）
                    # 注意：这可能涉及到额外的网络请求，取决于 SDK 是否已缓存
                    if hasattr(d, 'dos') and d.dos is not None:
                        try:
                            dos_val = d.dos.get_interpolated_value(efermi)
                            dos_lookup[str(d.material_id)] = float(dos_val)
                        except:
                            continue
            except Exception as e:
                print(f"警告: 批量获取 DOS 失败: {e}")
            # --------------------
 
            # 3. 合并数据
            for m_id in batch_ids:
                s = summary_dict[m_id]
                entry = {
                    "material_id": m_id,
                    "formula": s.formula_pretty,
                    "num_sites":s.nsites,
                    "density": s.density,
                    "energy_above_hull": s.energy_above_hull,
                    "band_gap": s.band_gap,
                    "bulk_modulus": s.bulk_modulus,
                    "shear_modulus": s.shear_modulus,
                    "piezo_tensor": piezo_lookup.get(m_id),
                    "dos_at_fermi": dos_lookup.get(m_id),
                    "structure_dict": s.structure.as_dict()
                }
                all_combined_data.append(entry)

    # 保存
    with open(SAVE_PATH, 'w') as f:
        json.dump(all_combined_data, f, indent=4)
    print(f"\n成功！所有数据已保存至 {SAVE_PATH}")


def fetch_crystal_data():
    all_data = []
    
    with MPRester(API_KEY) as mpr:
        print(f"正在查询原子数 <= {MAX_SITES} 的材料列表...")
        
        # 1. 基础性质查询 (从 summary 接口)
        # 过滤条件：原子数 nsites 范围 [1, 20]
        docs = mpr.materials.summary.search(
            nsites=(1, MAX_SITES),
            fields=[
                "material_id", "formula_pretty", "structure", 
                "bulk_modulus", "shear_modulus", "universal_anisotropy", # 弹性相关
                "density", "num_sites", "energy_above_hull", "band_gap"
            ]
        )
        
        print(f"找到 {len(docs)} 个符合条件的材料。开始提取详细性质...")

        # 为了演示，这里取前 100 个。如需全部，请去掉 [:100]
        for doc in tqdm(docs, desc="数据合并中"):
            m_id = str(doc.material_id)
            
            # 构建单条基础数据
            entry = {
                "material_id": m_id,
                "formula": doc.formula_pretty,
                "num_sites": doc.nsites,
                "density": doc.density,
                "energy_above_hull": doc.energy_above_hull,
                "band_gap": doc.band_gap,
                "bulk_modulus": doc.bulk_modulus,
                "shear_modulus": doc.shear_modulus,
                "structure_dict": doc.structure.as_dict() # 保存为字典便于存储
            }

            # 1. 获取压电张量 (Piezoelectric Tensor) - 使用最新 API 路径
            try:
                # search 返回的是一个列表，取第一个元素即可
                piezo_docs = mpr.materials.piezoelectric.search(material_ids=[m_id])
                if piezo_docs:
                    # 最新版字段可能是 total_piezo 或 e_ij
                    entry["piezo_tensor"] = piezo_docs[0].total_piezo 
                else:
                    entry["piezo_tensor"] = None
            except Exception:
                entry["piezo_tensor"] = None

            # 2. 获取费米能级态密度 (DOS) - 同样建议使用 search 路径
            try:
                # 注意：DOS 数据非常庞大，按 ID 逐个 search 会显著拖慢速度
                dos_docs = mpr.materials.dos.search(material_ids=[m_id])
                if dos_docs:
                    dos_obj = dos_docs[0].dos
                    # 提取费米能级处的插值
                    dos_at_fermi = dos_obj.get_interpolated_value(dos_obj.efermi)
                    entry["dos_at_fermi"] = float(dos_at_fermi)
                else:
                    entry["dos_at_fermi"] = None
            except Exception:
                entry["dos_at_fermi"] = None

            all_data.append(entry)

    # 保存到本地
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n采集完成！共保存 {len(all_data)} 条数据至 {SAVE_PATH}")

if __name__ == "__main__":
    # fetch_crystal_data()
    fetch_batch_data()