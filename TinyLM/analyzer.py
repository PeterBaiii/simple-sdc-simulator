import json
import glob
import os

def analyze_tinylm_results(json_file):
    """分析TinyLM测试结果"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print(f"TinyLM Fault Injection Test - Analysis")
    print(f"File: {os.path.basename(json_file)}")
    print("="*80)
    
    # 1. 模型和测试配置
    print("\n1. Model Configuration:")
    model_config = data['metadata']['model_config']
    print(f"   - Vocabulary size: {model_config['vocab_size']}")
    print(f"   - Embedding dim: {model_config['n_embd']}")
    print(f"   - Heads: {model_config['n_heads']}, Head dim: {model_config['head_dim']}")
    print(f"   - Block size: {model_config['block_size']}")
    
    print("\n2. Test Configuration:")
    test_config = data['metadata']['test_config']
    print(f"   - Injection index: {test_config['injection_index']}")
    print(f"   - Extract position: {test_config['extract_position']}")
    print(f"   - Test batches: {test_config['test_batches']}")
    print(f"   - Bit range: {test_config['bit_range'][0]}-{test_config['bit_range'][1]-1}")
    print(f"   - Detection condition: {test_config['detection_condition']}")
    print(f"   - Total tests: {test_config['total_tests']}")
    
    print("\n3. Training Information:")
    training_info = data['metadata']['training_info']
    print(f"   - Training steps: {training_info['training_steps']}")
    print(f"   - Learning rate: {training_info['learning_rate']}")
    print(f"   - Random seed: {training_info['random_seed']}")
    
    # 2. 总体结果
    print("\n4. Overall Detection Results:")
    overall = data['summary']['overall']
    print(f"   Detection Rate: {overall['detected']}/{overall['total']} ({overall['rate_percent']:.2f}%)")
    
    # 3. 按注入位置分析
    print("\n5. Results by Injection Location:")
    by_location = data['summary']['by_injection_location']
    
    # 排序（按检测率）
    sorted_locations = sorted(by_location.items(), key=lambda x: x[1]['rate_percent'], reverse=True)
    
    print(f"   {'Location':<12} {'Detected':<10} {'Total':<10} {'Rate':<10}")
    print(f"   {'-'*42}")
    for location, stats in sorted_locations:
        print(f"   {location:<12} {stats['detected']:<10} {stats['total']:<10} {stats['rate_percent']:>6.2f}%")
    
    # 4. 详细的比特位分析
    if 'detailed_results' in data:
        print("\n6. Detailed Bit-wise Analysis:")
        
        for location in test_config['injection_locations']:
            result = data['detailed_results'][location]
            rate = 100 * result['detection_count'] / result['total_tests']
            
            print(f"\n   Location: {location.upper()}")
            print(f"   Overall: {result['detection_count']}/{result['total_tests']} ({rate:.2f}%)")
            
            # 分析比特位
            bit_stats = result['bit_stats']
            bit_rates = {}
            for bit, stats in bit_stats.items():
                if stats['total'] > 0:
                    bit_rates[bit] = stats['detected'] / stats['total'] * 100
            
            if bit_rates:
                # 最好的比特位
                best_bits = sorted(bit_rates.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"   Top 5 best bits: {', '.join([f'bit{b}({r:.0f}%)' for b, r in best_bits])}")
                
                # 最差的比特位
                worst_bits = sorted(bit_rates.items(), key=lambda x: x[1])[:5]
                print(f"   Top 5 worst bits: {', '.join([f'bit{b}({r:.0f}%)' for b, r in worst_bits])}")
                
                # 统计完全检测和完全未检测的比特位
                fully_detected = sum(1 for r in bit_rates.values() if r == 100)
                not_detected = sum(1 for r in bit_rates.values() if r == 0)
                partial = len(bit_rates) - fully_detected - not_detected
                
                print(f"   Bit distribution: {fully_detected} fully detected, {partial} partial, {not_detected} not detected")
    
    # 5. 批次变异性分析
    if 'detailed_results' in data:
        print("\n7. Batch Variability Analysis:")
        
        for location in test_config['injection_locations']:
            result = data['detailed_results'][location]
            batch_results = result.get('batch_results', [])
            
            if batch_results:
                # 计算每个批次的检测率
                batch_rates = []
                for batch in batch_results:
                    n_detected = len(batch.get('detected_bits', []))
                    n_total = len(batch.get('detected_bits', [])) + len(batch.get('undetected_bits', []))
                    if n_total > 0:
                        batch_rates.append(100 * n_detected / n_total)
                
                if batch_rates:
                    import statistics
                    mean_rate = statistics.mean(batch_rates)
                    std_rate = statistics.stdev(batch_rates) if len(batch_rates) > 1 else 0
                    min_rate = min(batch_rates)
                    max_rate = max(batch_rates)
                    
                    print(f"\n   Location: {location.upper()}")
                    print(f"   Mean detection rate: {mean_rate:.2f}% (±{std_rate:.2f}%)")
                    print(f"   Range: [{min_rate:.2f}%, {max_rate:.2f}%]")
    
    print("\n" + "="*80 + "\n")

def compare_with_baseline(tinylm_file, baseline_file=None):
    """比较TinyLM结果与基线（如果有的话）"""
    
    with open(tinylm_file, 'r') as f:
        tinylm_data = json.load(f)
    
    print("="*80)
    print("TinyLM vs Baseline Comparison")
    print("="*80)
    
    if baseline_file and os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        print(f"\nTinyLM file: {os.path.basename(tinylm_file)}")
        print(f"Baseline file: {os.path.basename(baseline_file)}")
        
        # 比较总体检测率
        tinylm_rate = tinylm_data['summary']['overall']['rate_percent']
        baseline_rate = baseline_data['summary']['overall']['rate_percent']
        
        print(f"\nOverall Detection Rate:")
        print(f"  TinyLM:   {tinylm_rate:.2f}%")
        print(f"  Baseline: {baseline_rate:.2f}%")
        print(f"  Difference: {tinylm_rate - baseline_rate:+.2f}%")
        
        # 比较各个注入位置
        print(f"\nBy Injection Location:")
        print(f"  {'Location':<12} {'TinyLM':<12} {'Baseline':<12} {'Diff':<12}")
        print(f"  {'-'*48}")
        
        tinylm_locs = tinylm_data['summary']['by_injection_location']
        baseline_locs = baseline_data['summary']['by_injection_location']
        
        common_locs = set(tinylm_locs.keys()) & set(baseline_locs.keys())
        for loc in sorted(common_locs):
            t_rate = tinylm_locs[loc]['rate_percent']
            b_rate = baseline_locs[loc]['rate_percent']
            diff = t_rate - b_rate
            print(f"  {loc:<12} {t_rate:>6.2f}%     {b_rate:>6.2f}%     {diff:>+6.2f}%")
    else:
        print("\nNo baseline file provided or file not found.")
        print("Showing TinyLM results only:")
        
        tinylm_rate = tinylm_data['summary']['overall']['rate_percent']
        print(f"\nOverall Detection Rate: {tinylm_rate:.2f}%")
        
        print(f"\nBy Injection Location:")
        for loc, stats in tinylm_data['summary']['by_injection_location'].items():
            print(f"  {loc:<12}: {stats['rate_percent']:>6.2f}%")
    
    print("\n" + "="*80 + "\n")

def find_latest_results(results_dir="."):

    # 递归匹配 results/**/attention_fault_test_results_*.json
    full_pattern = os.path.join(results_dir, "**", "tinylm_fault_test_results_*.json")
    summary_pattern = os.path.join(results_dir, "**", "tinylm_fault_test_summary_*.json")

    full_files = glob.glob(full_pattern, recursive=True)
    summary_files = glob.glob(summary_pattern, recursive=True)

    if not full_files and not summary_files:
        print("No TinyLM result files found!")
        return None, None

    latest_full = max(full_files, key=os.path.getmtime) if full_files else None
    latest_summary = max(summary_files, key=os.path.getmtime) if summary_files else None

    return latest_full, latest_summary

if __name__ == "__main__":
    # 查找最新的结果文件
    latest_full, latest_summary = find_latest_results()
    
    if latest_summary:
        print(f"Found latest summary: {latest_summary}\n")
        analyze_tinylm_results(latest_summary)
    
    if latest_full:
        print(f"Found latest full results: {latest_full}\n")
        analyze_tinylm_results(latest_full)
        
        # 可选：与baseline比较
        baseline_files = glob.glob("attention_fault_test_summary_*.json")
        if baseline_files:
            latest_baseline = max(baseline_files, key=os.path.getmtime)
            compare_with_baseline(latest_full, latest_baseline)
    
    if not latest_full and not latest_summary:
        print("No result files found. Please run the test first:")
        print("  python tinylm_fault_test.py")