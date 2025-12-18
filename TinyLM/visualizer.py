import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_tinylm_results(json_file):
    """可视化TinyLM测试结果"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'detailed_results' not in data:
        print("No detailed results available for visualization.")
        return
    
    locations = data['metadata']['test_config']['injection_locations']
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('TinyLM Fault Injection Test Results', fontsize=16, fontweight='bold')
    
    # 1. 按注入位置的检测率柱状图
    ax1 = fig.add_subplot(gs[0, :2])
    location_rates = []
    location_labels = []
    
    for location in locations:
        result = data['detailed_results'][location]
        rate = 100 * result['detection_count'] / result['total_tests']
        location_rates.append(rate)
        location_labels.append(location)
    
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(locations)))
    bars = ax1.bar(location_labels, location_rates, color=colors, alpha=0.8)
    ax1.set_ylabel('Detection Rate (%)', fontsize=12)
    ax1.set_title('Detection Rate by Injection Location', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. 检测条件说明
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    detection_condition = data['metadata']['test_config']['detection_condition']
    model_info = f"Model: TinyLM\n"
    model_info += f"Heads: {data['metadata']['model_config']['n_heads']}\n"
    model_info += f"Head dim: {data['metadata']['model_config']['head_dim']}\n"
    model_info += f"Block size: {data['metadata']['model_config']['block_size']}\n\n"
    model_info += f"Detection:\n{detection_condition}\n\n"
    model_info += f"Total tests:\n{data['metadata']['test_config']['total_tests']}"
    
    ax2.text(0.1, 0.5, model_info, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 3. 比特位分析 - 最好和最差位置
    ax3 = fig.add_subplot(gs[1, :])
    
    best_location = location_labels[np.argmax(location_rates)]
    worst_location = location_labels[np.argmin(location_rates)]
    
    best_result = data['detailed_results'][best_location]
    worst_result = data['detailed_results'][worst_location]
    
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
        
        ax3.plot(bit_range, best_rates, 'g-o', label=f'{best_location} (best: {np.mean(best_rates):.1f}%)', 
                linewidth=2.5, markersize=5, alpha=0.7)
        ax3.plot(bit_range, worst_rates, 'r-s', label=f'{worst_location} (worst: {np.mean(worst_rates):.1f}%)', 
                linewidth=2.5, markersize=5, alpha=0.7)
        
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax3.text(32, 52, '50%', fontsize=9, color='gray')
        
        ax3.set_xlabel('Bit Position', fontsize=12)
        ax3.set_ylabel('Detection Rate (%)', fontsize=12)
        ax3.set_title('Detection Rate by Bit Position (Best vs Worst Location)', 
                     fontsize=13, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=11)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=24, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)
        ax3.text(24, 95, 'Sign bit', fontsize=9, color='blue', ha='center')
    
    # 4. 比特位热力图
    ax4 = fig.add_subplot(gs[2, :])
    
    heatmap_data = np.zeros((len(locations), 32))
    
    for i, location in enumerate(locations):
        result = data['detailed_results'][location]
        if 'bit_stats' in result:
            for j, bit in enumerate(range(1, 33)):
                bit_str = str(bit)
                if bit_str in result['bit_stats']:
                    stats = result['bit_stats'][bit_str]
                    rate = 100 * stats['detected'] / stats['total'] if stats['total'] > 0 else 0
                    heatmap_data[i, j] = rate
    
    im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax4.set_xticks(np.arange(32))
    ax4.set_yticks(np.arange(len(locations)))
    ax4.set_xticklabels(map(str, range(1, 33)), fontsize=9)
    ax4.set_yticklabels(location_labels, fontsize=11)
    ax4.set_xlabel('Bit Position', fontsize=12)
    ax4.set_ylabel('Injection Location', fontsize=12)
    ax4.set_title('Detection Rate Heatmap: Location × Bit Position', 
                 fontsize=13, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Detection Rate (%)', fontsize=11)
    
    # 添加网格
    ax4.set_xticks(np.arange(32)-.5, minor=True)
    ax4.set_yticks(np.arange(len(locations))-.5, minor=True)
    ax4.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 标注符号位
    ax4.axvline(x=23.5, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    
    # 保存图表
    output_filename = json_file.replace('.json', '_visualization.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_filename}")
    
    plt.show()

def visualize_batch_variance(json_file):
    """可视化批次之间的变异性"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'detailed_results' not in data:
        print("No detailed results available.")
        return
    
    locations = data['metadata']['test_config']['injection_locations']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TinyLM Fault Injection - Batch Variability Analysis', 
                fontsize=15, fontweight='bold')
    
    for idx, location in enumerate(locations):
        ax = axes[idx // 2, idx % 2]
        result = data['detailed_results'][location]
        batch_results = result.get('batch_results', [])
        
        if not batch_results:
            ax.text(0.5, 0.5, f'No batch data for {location}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # 计算每个批次的检测率
        batch_indices = []
        batch_rates = []
        
        for batch in batch_results:
            n_detected = len(batch.get('detected_bits', []))
            n_total = len(batch.get('detected_bits', [])) + len(batch.get('undetected_bits', []))
            if n_total > 0:
                batch_indices.append(batch['batch_idx'])
                batch_rates.append(100 * n_detected / n_total)
        
        if batch_rates:
            ax.plot(batch_indices, batch_rates, 'o-', linewidth=2, markersize=6, alpha=0.7)
            
            # 添加平均线
            mean_rate = np.mean(batch_rates)
            ax.axhline(y=mean_rate, color='r', linestyle='--', linewidth=2, alpha=0.7,
                      label=f'Mean: {mean_rate:.1f}%')
            
            # 添加标准差区域
            std_rate = np.std(batch_rates)
            ax.fill_between(batch_indices, 
                           [mean_rate - std_rate] * len(batch_indices),
                           [mean_rate + std_rate] * len(batch_indices),
                           alpha=0.2, color='red', label=f'±1 std: {std_rate:.1f}%')
            
            ax.set_xlabel('Batch Index', fontsize=11)
            ax.set_ylabel('Detection Rate (%)', fontsize=11)
            ax.set_title(f'Location: {location.upper()}', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 100])
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_filename = json_file.replace('.json', '_batch_variance.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Batch variance visualization saved to: {output_filename}")
    
    plt.show()

def compare_tinylm_with_baseline(tinylm_file, baseline_file):
    """对比TinyLM和baseline的结果"""
    
    with open(tinylm_file, 'r') as f:
        tinylm_data = json.load(f)
    
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('TinyLM vs Baseline Comparison', fontsize=15, fontweight='bold')
    
    # 1. 总体对比
    ax1 = axes[0]
    
    tinylm_locations = tinylm_data['summary']['by_injection_location']
    baseline_locations = baseline_data['summary']['by_injection_location']
    
    common_locs = sorted(set(tinylm_locations.keys()) & set(baseline_locations.keys()))
    
    tinylm_rates = []
    baseline_rates = []

    if common_locs:
        x = np.arange(len(common_locs))
        width = 0.35
        
        tinylm_rates = [float(tinylm_locations[loc]['rate_percent']) for loc in common_locs]
        baseline_rates = [float(baseline_locations[loc]['rate_percent']) for loc in common_locs]
        
        bars1 = ax1.bar(x - width/2, tinylm_rates, width, label='TinyLM', alpha=0.8)
        bars2 = ax1.bar(x + width/2, baseline_rates, width, label='Baseline', alpha=0.8)
        
        ax1.set_ylabel('Detection Rate (%)', fontsize=12)
        ax1.set_title('Detection Rate by Location', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(common_locs, fontsize=11)
        ax1.legend(fontsize=11)
        ax1.set_ylim([0, 100])
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标注
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9)
    
    # 2. 差异图
    ax2 = axes[1]
    
    if common_locs:
        differences = [tinylm_rates[i] - baseline_rates[i] for i in range(len(common_locs))]
        colors = ['green' if d > 0 else 'red' for d in differences]
        
        bars = ax2.bar(common_locs, differences, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('Difference (%)', fontsize=12)
        ax2.set_title('TinyLM - Baseline Difference', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加数值标注
        for bar in bars:
            height = bar.get_height()
            label_y = height + 1 if height > 0 else height - 1
            ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:+.1f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    
    output_filename = tinylm_file.replace('.json', '_comparison.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Comparison visualization saved to: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    # 查找最新的结果文件
    results_dir = "."
    pattern = os.path.join(results_dir, "**", "tinylm_fault_test_results_*.json")
    tinylm_files = glob.glob(pattern, recursive=True)
    
    if not tinylm_files:
        print("No TinyLM result files found. Please run the test first:")
        print("  python tinylm_fault_test.py")
    else:
        latest_file = max(tinylm_files, key=os.path.getmtime)
        print(f"Visualizing: {latest_file}\n")
        
        visualize_tinylm_results(latest_file)
        visualize_batch_variance(latest_file)
        
        # 可选：与baseline比较
        baseline_files = glob.glob("attention_fault_test_results_*.json")
        if baseline_files:
            latest_baseline = max(baseline_files, key=os.path.getmtime)
            print(f"\nComparing with baseline: {latest_baseline}")
            compare_tinylm_with_baseline(latest_file, latest_baseline)