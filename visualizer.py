import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(json_file):
    """可视化测试结果"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'detailed_results' not in data:
        print("No detailed results available for visualization.")
        return
    
    locations = data['metadata']['configuration']['injection_locations']
    sigmas = data['metadata']['configuration']['sigmas']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Attention Fault Injection Test Results', fontsize=16, fontweight='bold')
    
    # 1. 按注入位置的检测率柱状图
    ax1 = axes[0, 0]
    location_rates = []
    location_labels = []
    
    for location in locations:
        rates_per_location = []
        for sigma in sigmas:
            result = data['detailed_results'][location][str(sigma)]
            rate = 100 * result['detection_count'] / result['total_tests']
            rates_per_location.append(rate)
        avg_rate = np.mean(rates_per_location)
        location_rates.append(avg_rate)
        location_labels.append(location)
    
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(locations)))
    bars = ax1.bar(location_labels, location_rates, color=colors)
    ax1.set_ylabel('Detection Rate (%)', fontsize=12)
    ax1.set_title('Average Detection Rate by Injection Location', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    # 2. 按注入位置和标准差的分组柱状图
    ax2 = axes[0, 1]
    x = np.arange(len(locations))
    width = 0.35
    
    rates_sigma1 = []
    rates_sigma2 = []
    
    for location in locations:
        result1 = data['detailed_results'][location][str(sigmas[0])]
        result2 = data['detailed_results'][location][str(sigmas[1])]
        rate1 = 100 * result1['detection_count'] / result1['total_tests']
        rate2 = 100 * result2['detection_count'] / result2['total_tests']
        rates_sigma1.append(rate1)
        rates_sigma2.append(rate2)
    
    bars1 = ax2.bar(x - width/2, rates_sigma1, width, label=f'σ = {sigmas[0]:.2f}', alpha=0.8)
    bars2 = ax2.bar(x + width/2, rates_sigma2, width, label=f'σ = {sigmas[1]:.2f}', alpha=0.8)
    
    ax2.set_ylabel('Detection Rate (%)', fontsize=12)
    ax2.set_title('Detection Rate by Location and Standard Deviation', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(location_labels)
    ax2.legend()
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 热力图：位置 × 标准差
    ax3 = axes[1, 0]
    heatmap_data = np.zeros((len(locations), len(sigmas)))
    
    for i, location in enumerate(locations):
        for j, sigma in enumerate(sigmas):
            result = data['detailed_results'][location][str(sigma)]
            rate = 100 * result['detection_count'] / result['total_tests']
            heatmap_data[i, j] = rate
    
    im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax3.set_xticks(np.arange(len(sigmas)))
    ax3.set_yticks(np.arange(len(locations)))
    ax3.set_xticklabels([f'σ={s:.2f}' for s in sigmas])
    ax3.set_yticklabels(location_labels)
    ax3.set_title('Detection Rate Heatmap', fontsize=13, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(locations)):
        for j in range(len(sigmas)):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=11)
    
    plt.colorbar(im, ax=ax3, label='Detection Rate (%)')
    
    # 4. 比特位分析（选择检测率最高和最低的位置）
    ax4 = axes[1, 1]
    
    # 找到检测率最高和最低的位置
    best_location = location_labels[np.argmax(location_rates)]
    worst_location = location_labels[np.argmin(location_rates)]
    
    # 使用第一个标准差的数据
    best_result = data['detailed_results'][best_location][str(sigmas[0])]
    worst_result = data['detailed_results'][worst_location][str(sigmas[0])]
    
    if 'bit_stats' in best_result and 'bit_stats' in worst_result:
        bit_range = range(1, 33)
        
        best_rates = []
        worst_rates = []
        
        for bit in bit_range:
            bit_str = str(bit)
            if bit_str in best_result['bit_stats']:
                stats = best_result['bit_stats'][bit_str]
                rate = 100 * stats['detected'] / stats['total'] if stats['total'] > 0 else 0
                best_rates.append(rate)
            else:
                best_rates.append(0)
            
            if bit_str in worst_result['bit_stats']:
                stats = worst_result['bit_stats'][bit_str]
                rate = 100 * stats['detected'] / stats['total'] if stats['total'] > 0 else 0
                worst_rates.append(rate)
            else:
                worst_rates.append(0)
        
        ax4.plot(bit_range, best_rates, 'g-o', label=f'{best_location} (best)', 
                linewidth=2, markersize=4, alpha=0.7)
        ax4.plot(bit_range, worst_rates, 'r-s', label=f'{worst_location} (worst)', 
                linewidth=2, markersize=4, alpha=0.7)
        
        ax4.set_xlabel('Bit Position', fontsize=12)
        ax4.set_ylabel('Detection Rate (%)', fontsize=12)
        ax4.set_title(f'Detection Rate by Bit Position (σ={sigmas[0]:.2f})', 
                     fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.set_ylim([0, 100])
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=24, color='gray', linestyle='--', alpha=0.5, label='Sign bit')
    else:
        ax4.text(0.5, 0.5, 'Bit statistics not available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # 保存图表
    output_filename = os.path.splitext(json_file)[0] + "_visualization.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_filename}")
    
    plt.show()

def create_bit_heatmap(json_file):
    """创建比特位×注入位置的热力图"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'detailed_results' not in data:
        print("No detailed results available.")
        return
    
    locations = data['metadata']['configuration']['injection_locations']
    sigmas = data['metadata']['configuration']['sigmas']
    
    # 使用第一个标准差
    sigma = sigmas[0]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    bit_range = range(1, 33)
    heatmap_data = np.zeros((len(locations), len(bit_range)))
    
    for i, location in enumerate(locations):
        result = data['detailed_results'][location][str(sigma)]
        if 'bit_stats' in result:
            for j, bit in enumerate(bit_range):
                bit_str = str(bit)
                if bit_str in result['bit_stats']:
                    stats = result['bit_stats'][bit_str]
                    rate = 100 * stats['detected'] / stats['total'] if stats['total'] > 0 else 0
                    heatmap_data[i, j] = rate
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(bit_range)))
    ax.set_yticks(np.arange(len(locations)))
    ax.set_xticklabels(map(str, bit_range))
    ax.set_yticklabels(locations)
    ax.set_xlabel('Bit Position', fontsize=14)
    ax.set_ylabel('Injection Location', fontsize=14)
    ax.set_title(f'Detection Rate by Injection Location and Bit Position (σ={sigma:.2f})', 
                fontsize=15, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Detection Rate (%)', fontsize=12)
    
    # 添加网格
    ax.set_xticks(np.arange(len(bit_range))-.5, minor=True)
    ax.set_yticks(np.arange(len(locations))-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 标注符号位
    ax.axvline(x=30.5, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(31, len(locations), 'Sign bit →', ha='right', va='bottom', 
           fontsize=10, color='blue', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    output_filename = os.path.splitext(json_file)[0] + "_heatmap.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Bit heatmap saved to: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    # 查找最新的结果文件
    results_dir = "results"
    pattern = os.path.join(results_dir, "**", "attention_fault_test_results_*.json")
    full_files = glob.glob(pattern, recursive=True)
    
    if not full_files:
        print("No result files found. Please run the test first:")
        print("  python attention_fault_test_final.py")
    else:
        latest_file = max(full_files, key=os.path.getmtime)
        print(f"Visualizing: {latest_file}")
        
        visualize_results(latest_file)
        create_bit_heatmap(latest_file)