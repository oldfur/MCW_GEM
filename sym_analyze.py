import os
import glob
import pandas as pd
from collections import Counter
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def analyze_crystal_folder(folder_path, result_save_path, symprec=0.1):
    """
    遍历文件夹，分析晶体空间群分布
    :param folder_path: 存放 cif 文件的文件夹路径
    :param result_save_path: 保存结果的 CSV 文件路径
    :param symprec: 对称性容差。对于生成的模型数据，建议设置在 0.1 左右
    """
    results = []
    
    # 匹配特定的文件名模式
    search_pattern = os.path.join(folder_path, "crystal_epoch_0_sample_*.cif")
    cif_files = glob.glob(search_pattern)
    
    print(f"找到 {len(cif_files)} 个文件，开始分析...")

    for file_path in cif_files:
        file_name = os.path.basename(file_path)
        try:
            # 加载结构
            struct = Structure.from_file(file_path)
            
            # 对称性分析
            sga = SpacegroupAnalyzer(struct, symprec=symprec)  # 可以尝试不同的 symprec 值以获得更稳定的结果
            
            # 获取空间群信息
            symbol = sga.get_space_group_symbol()
            number = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
            
            results.append({
                "filename": file_name,
                "spg_symbol": symbol,
                "spg_number": number,
                "crystal_system": crystal_system
            })
        except Exception as e:
            print(f"文件 {file_name} 解析失败: {e}")

    # 转换为 DataFrame 进行统计
    df = pd.DataFrame(results)
    
    # 统计空间群分布
    spg_counts = df['spg_symbol'].value_counts()
    
    print("\n--- 空间群分布统计 ---")
    print(spg_counts)
    
    # 保存结果到 CSV
    df.to_csv(result_save_path, index=False)
    print(f"\n详细数据已保存至 {result_save_path}")

    return df

if __name__ == "__main__":
    # 使用示例
    # 替换为你的 cif 文件夹路径
    folder_to_process = "../0206_2_sample_128_ve_sym_guide"
    result_save_path = "../0206_2_spacegroup_analysis_results.csv"
    analysis_df = analyze_crystal_folder(folder_to_process, result_save_path, symprec=0.3)
