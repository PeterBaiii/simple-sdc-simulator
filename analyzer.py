import json
import glob
import os

def analyze_results(json_file):
    """分析测试结果JSON文件"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print(f"Analysis of: {os.path.basename(json_file)}")
    print("="*80)
    
    # 1. 测试配置信息
    print("\n1. Test Configuration:")
    config = data['metadata']['configuration']
    print(f"   - Batch size: {config['B']}, Heads: {config['H']}, Sequence length: {config['L']}, Head dim: {config['Dh']}")
    print(f"   - Injection index: {config['injection_index']}")
    print(f"   - Tested vectors: {config['n_vectors']}")
    print(f"   - Bit range: {config['bit_range'][0]}-{config['bit_range'][1]-1}")
    print(f"   - Total tests: {config['total_tests']}")
    
    # 2. 总体结果
    print("\n2. Overall Results:")
    overall = data['summary']['overall']
    print(f"   Detection Rate: {overall['detected']}/{overall['total']} ({overall['rate_percent']:.2f}%)")
    
    # 3. 按注入位置分析
    print("\n3. Results by Injection Location:")
    by_location = data['summary']['by_injection_location']
    
    # 排序（按检测率）
    sorted_locations = sorted(by_location.items(), key=lambda x: x[1]['rate_percent'], reverse=True)
    
    print(f"   {'Location':<12} {'Detected':<10} {'Total':<10} {'Rate':<10}")
    print(f"   {'-'*42}")
    for location, stats in sorted_locations:
        print(f"   {location:<12} {stats['detected']:<10} {stats['total']:<10} {stats['rate_percent']:>6.2f}%")
    
    # 4. 按标准差分析
    print("\n4. Results by Standard Deviation:")
    by_sigma = data['summary']['by_sigma']
    
    print(f"   {'Sigma':<12} {'Detected':<10} {'Total':<10} {'Rate':<10}")
    print(f"   {'-'*42}")
    for sigma, stats in sorted(by_sigma.items(), key=lambda x: float(x[0])):
        print(f"   {sigma:<12} {stats['detected']:<10} {stats['total']:<10} {stats['rate_percent']:>6.2f}%")
    
    # 5. 详细结果（如果有）
    if 'detailed_results' in data:
        print("\n5. Detailed Results by Location and Sigma:")
        for location in config['injection_locations']:
            for sigma in config['sigmas']:
                sigma_str = str(sigma)
                if sigma_str in data['detailed_results'][location]:
                    result = data['detailed_results'][location][sigma_str]
                    rate = 100 * result['detection_count'] / result['total_tests']
                    print(f"   {location:<12} σ={sigma:<6.4f}: {result['detection_count']:>3}/{result['total_tests']:<3} ({rate:>6.2f}%)")
                    
                    # 分析比特位分布（如果有bit_stats）
                    if 'bit_stats' in result:
                        bit_stats = result['bit_stats']
                        # 找出检测率最高和最低的比特位
                        bit_rates = {}
                        for bit, stats in bit_stats.items():
                            if stats['total'] > 0:
                                bit_rates[bit] = stats['detected'] / stats['total'] * 100
                        
                        if bit_rates:
                            best_bits = sorted(bit_rates.items(), key=lambda x: x[1], reverse=True)[:3]
                            worst_bits = sorted(bit_rates.items(), key=lambda x: x[1])[:3]
                            
                            print(f"      Best bits: {', '.join([f'{b}({r:.0f}%)' for b, r in best_bits])}")
                            print(f"      Worst bits: {', '.join([f'{b}({r:.0f}%)' for b, r in worst_bits])}")
    
    print("\n" + "="*80 + "\n")

def compare_locations(json_file):
    """比较不同注入位置的检测效果"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print("Injection Location Comparison")
    print("="*80)
    
    if 'detailed_results' not in data:
        print("No detailed results available for comparison.")
        return
    
    locations = data['metadata']['configuration']['injection_locations']
    sigmas = data['metadata']['configuration']['sigmas']
    
    # 创建比较表格
    print(f"\n{'Location':<12}", end='')
    for sigma in sigmas:
        print(f"σ={sigma:<8.4f}", end='')
    print("Average")
    print("-" * 60)
    
    for location in locations:
        print(f"{location:<12}", end='')
        rates = []
        for sigma in sigmas:
            sigma_str = str(sigma)
            result = data['detailed_results'][location][sigma_str]
            rate = 100 * result['detection_count'] / result['total_tests']
            rates.append(rate)
            print(f"{rate:>6.2f}%    ", end='')
        avg_rate = sum(rates) / len(rates)
        print(f"{avg_rate:>6.2f}%")
    
    print("\n" + "="*80 + "\n")

def find_latest_results(results_dir="results"):

    # 递归匹配 results/**/attention_fault_test_results_*.json
    full_pattern = os.path.join(results_dir, "**", "attention_fault_test_results_*.json")
    summary_pattern = os.path.join(results_dir, "**", "attention_fault_test_summary_*.json")

    full_files = glob.glob(full_pattern, recursive=True)
    summary_files = glob.glob(summary_pattern, recursive=True)

    if not full_files and not summary_files:
        print(f"No result files found in: {results_dir}")
        return None, None

    latest_full = max(full_files, key=os.path.getmtime) if full_files else None
    latest_summary = max(summary_files, key=os.path.getmtime) if summary_files else None

    return latest_full, latest_summary

if __name__ == "__main__":
    # 查找最新的结果文件
    latest_full, latest_summary = find_latest_results()
    
    if latest_summary:
        print(f"Found latest summary: {latest_summary}")
        analyze_results(latest_summary)
    
    if latest_full:
        print(f"Found latest full results: {latest_full}")
        analyze_results(latest_full)
        compare_locations(latest_full)
    
    if not latest_full and not latest_summary:
        print("No result files found. Please run the test first:")
        print("  python attention_fault_test_final.py")