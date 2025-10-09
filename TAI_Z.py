import itertools
import math
import numpy as np # 引入 NumPy

def calculate_energy(spins_np, J, h, Nx, Ny):
    """
    計算 Nx x Ny 點陣上，給定自旋組態的能量。
    採用週期性邊界條件 (PBC)。
    spins_np: NumPy 2D array，表示 Nx x Ny 的自旋組態。
              注意：現在直接接收 NumPy 陣列。
    J: 交換耦合常數
    h: 外部磁場強度
    Nx: 點陣的列數
    Ny: 點陣的行數
    """
    interaction_sum = 0
    
    # 計算所有唯一的水平鍵結 (Horizontal Bonds)
    # 使用 NumPy 陣列索引 spins_np[i, j]
    for i in range(Nx): # 遍歷每一列
        for j in range(Ny): # 遍歷每一行的自旋
            interaction_sum += spins_np[i, j] * spins_np[i, (j + 1) % Ny]
    
    # 計算所有唯一的垂直鍵結 (Vertical Bonds)
    # 使用 NumPy 陣列索引 spins_np[i, j]
    for i in range(Nx): # 遍歷每一列
        for j in range(Ny): # 遍歷每一行的自旋
            interaction_sum += spins_np[i, j] * spins_np[(i + 1) % Nx, j]
            
    # 總能量的交互作用部分
    energy = -J * interaction_sum
            
    # 加上外部磁場的作用 (Zeeman Energy)
    # 使用 np.sum() 對整個陣列求和，更高效
    energy += -h * np.sum(spins_np)
            
    return energy

def partition_function_ising_NxNy(J, h, beta, Nx, Ny):
    """
    計算 Nx x Ny Ising 模型的週期性邊界條件下的分佈函數。
    J: 交換耦合常數
    h: 外部磁場強度
    beta: 1/(k_B T) (這裡假設 k_B = 1)
    Nx: 點陣的列數
    Ny: 點陣的行數
    """
    Z = 0.0 # 初始化分佈函數
    num_spins = Nx * Ny
    
    # 使用 itertools.product 生成所有 2^(Nx*Ny) 種可能的自旋組態
    # config_1d_tuple 會是像 (-1, -1, -1, -1) 這樣的 tuple
    for config_1d_tuple in itertools.product([-1, 1], repeat=num_spins):
        # 將 1D 的組態 (tuple) 轉換為 NumPy array
        config_1d_np = np.array(config_1d_tuple)
        
        # 使用 numpy.reshape 將 1D array 轉換為 2D 點陣表示
        # 這是我們這次優化的重點！
        spins_2d_np = config_1d_np.reshape((Nx, Ny))
        
        # 計算當前組態的能量 (傳入 NumPy array)
        energy = calculate_energy(spins_2d_np, J, h, Nx, Ny)
        
        # 將 Boltzmann 因子加入分佈函數 Z
        Z += math.exp(-beta * energy)
        
    return Z

# ---

J_value = 1.0 # 交換耦合常數
h_value = 0.0 # 外部磁場強度
temperature = 1.0 # 溫度
beta_value = 1.0 / temperature # 逆溫度 (假設 k_B = 1)

print("--- 計算 2x2 點陣 ---")
Nx_2x2 = 2
Ny_2x2 = 2
Z_result_2x2 = partition_function_ising_NxNy(J_value, h_value, beta_value, Nx_2x2, Ny_2x2)
print(f"當 Nx={Nx_2x2}, Ny={Ny_2x2}, J={J_value}, h={h_value}, T={temperature} 時，2x2 Ising 模型的分佈函數 Z = {Z_result_2x2:.4f}")

print("\n--- 計算 2x3 點陣 ---")
Nx_2x3 = 2
Ny_2x3 = 3
Z_result_2x3 = partition_function_ising_NxNy(J_value, h_value, beta_value, Nx_2x3, Ny_2x3)
print(f"當 Nx={Nx_2x3}, Ny={Ny_2x3}, J={J_value}, h={h_value}, T={temperature} 時，2x3 Ising 模型的分佈函數 Z = {Z_result_2x3:.4f}")

print("\n--- 計算 3x2 點陣 (應與 2x3 相同) ---")
Nx_3x2 = 3
Ny_3x2 = 2
Z_result_3x2 = partition_function_ising_NxNy(J_value, h_value, beta_value, Nx_3x2, Ny_3x2)
print(f"當 Nx={Nx_3x2}, Ny={Ny_3x2}, J={J_value}, h={h_value}, T={temperature} 時，3x2 Ising 模型的分佈函數 Z = {Z_result_3x2:.4f}")

print("\n--- 嘗試計算 3x3 點陣 (可能還是會需要一點時間) ---")
Nx_3x3 = 3
Ny_3x3 = 3
Z_result_3x3 = partition_function_ising_NxNy(J_value, h_value, beta_value, Nx_3x3, Ny_3x3)
print(f"當 Nx={Nx_3x3}, Ny={Ny_3x3}, J={J_value}, h={h_value}, T={temperature} 時，3x3 Ising 模型的分佈函數 Z = {Z_result_3x3:.4f}")
