import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    total = len(df)
    
    print(f"=== 数据集概览 (总计: {total} 条) ===")
    
    # 统计有效率 (非空比例)
    stats = []
    properties = ['bulk_modulus', 'shear_modulus', 'band_gap', 'energy_above_hull', 
                  'density', 'piezo_tensor', 'efermi', 'e_ij_max']
    
    for prop in properties:
        valid_count = df[prop].notnull().sum()
        stats.append({
            "性质": prop,
            "有效数量": valid_count,
            "有效率 (%)": round(valid_count / total * 100, 2)
        })
    
    print(pd.DataFrame(stats))


    # 打印 DataFrame 的各类属性与摘要信息
    print("\n=== DataFrame 属性 ===")
    print(f"形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n列类型:")
    print(df.dtypes)
    print("\n每列非空数量:")
    print(df.notnull().sum())
    # 每列非空数量:
    # material_id          80943
    # formula              80943
    # structure_dict       80943
    # num_sites            80943
    # density              80943
    # energy_above_hull    80943
    # band_gap             80943
    # efermi               47481
    # bulk_modulus         12316
    # shear_modulus        12316
    # piezo_tensor          2152
    # e_ij_max              2152
    # dos_at_fermi             0
    
    
    print("\n每列空值数量:")
    print(df.isnull().sum())
    print("\n描述性统计 (数值列):")
    print(df.describe())

    # 可视化非简单属性
    print("formula:", df['formula'].iloc[0]) 
    print("bulk_modulus:", df['bulk_modulus'].dropna().iloc[0]) 
    # {'voigt': 23.574, 'reuss': 23.574, 'vrh': 23.574}
    print("shear_modulus:", df['shear_modulus'].dropna().iloc[0]) 
    # {'voigt': 14.133, 'reuss': 12.319, 'vrh': 13.226}
    print("piezo_tensor:", df['piezo_tensor'].dropna().iloc[0])
    # [
    # [0.0,      0.0,     0.0,    0.0,     -0.1735, 0.0], 
    # [0.0,      0.0,     0.0,    -0.1735, 0.0,     0.0], 
    # [-0.1499,  -0.1499, 0.9911, 0.0,     0.0,     0.0]
    # ]
    print("structure_dict:", df['structure_dict'][0])
    # {
    #     '@module': 'pymatgen.core.structure', 
    #     '@class': 'Structure', 
    #     'charge': 0, 
    #     'lattice': 
    #         {'matrix': 
    #         [[3.48820304, -0.0,       2.01391491], 
    #         [1.16273435, 3.28870854, 2.01391491], 
    #         [0.0,        -0.0,       4.02782982]], 
    #         'pbc': [True, True, True], 
    #         'a': 4.027829901198107, 
    #         'b': 4.0278292285621005, 
    #         'c': 4.02782982, 
    #         'alpha': 59.99999514264196, 
    #         'beta': 60.00000066686326, 
    #         'gamma': 59.999994872937556, 
    #         'volume': 46.205987384126566
    #         }, 
    #     'properties': {}, 
    #     'sites': # list of sites for atoms
    #         [
    #             {
    #                 'species': [{'element': 'Ac', 'occu': 1}], 
    #                 'abc': [-0.0, -0.0, 0.0], 
    #                 'properties': {'magmom': -0.0}, 
    #                 'label': 'Ac', 
    #                 'xyz': [0.0, 0.0, 0.0]
    #             }
    #         ]
    # }

if __name__ == "__main__":
    analyze_dataset("../crystal_data.json")