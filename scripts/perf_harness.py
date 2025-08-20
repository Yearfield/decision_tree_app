#!/usr/bin/env python3
"""
Performance harness for testing key functions on synthetic data.
Times infer_branch_options and validation summaries on 10k-row dataset.
"""

import time
import pandas as pd
import os
import sys
from typing import Dict, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.tree import infer_branch_options
from logic.validate import detect_orphan_nodes, detect_missing_red_flags, compute_validation_report
from utils.helpers import validate_headers


def load_synthetic_data(filename: str = "synthetic_10k.csv") -> pd.DataFrame:
    """Load the synthetic data file."""
    file_path = os.path.join(os.path.dirname(__file__), "..", filename)
    
    if not os.path.exists(file_path):
        print(f"❌ Synthetic data file not found: {file_path}")
        print("Please run scripts/gen_synth.py first to generate the data.")
        sys.exit(1)
    
    print(f"📁 Loading synthetic data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df):,} rows with shape: {df.shape}")
    
    # Validate headers
    if not validate_headers(df):
        print("❌ Invalid headers in synthetic data")
        sys.exit(1)
    
    return df


def time_function(func, *args, **kwargs) -> Dict[str, Any]:
    """Time a function execution and return results."""
    print(f"⏱️  Timing {func.__name__}...")
    
    # Warm up run (discard)
    start_warm = time.time()
    _ = func(*args, **kwargs)
    warm_time = time.time() - start_warm
    
    # Actual timed run
    start = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start
    
    return {
        "function": func.__name__,
        "execution_time": execution_time,
        "warm_up_time": warm_time,
        "result_size": len(result) if hasattr(result, '__len__') else "N/A",
        "result_type": type(result).__name__
    }


def benchmark_infer_branch_options(df: pd.DataFrame) -> Dict[str, Any]:
    """Benchmark the infer_branch_options function."""
    print("\n" + "="*60)
    print("🌳 BENCHMARKING: infer_branch_options")
    print("="*60)
    
    result = time_function(infer_branch_options, df)
    
    print(f"✅ {result['function']} completed in {result['execution_time']:.3f} seconds")
    print(f"🔥 Warm-up time: {result['warm_up_time']:.3f} seconds")
    print(f"📊 Result: {result['result_type']} with {result['result_size']} entries")
    
    # Show some sample results
    if isinstance(result['result_size'], int) and result['result_size'] > 0:
        # Get the actual result from the function call
        actual_result = infer_branch_options(df)
        sample_keys = list(actual_result.keys())[:5]
        print(f"🔑 Sample keys: {sample_keys}")
    
    return result


def benchmark_validation_summaries(df: pd.DataFrame) -> Dict[str, Any]:
    """Benchmark the validation summary functions."""
    print("\n" + "="*60)
    print("🔍 BENCHMARKING: Validation Summaries")
    print("="*60)
    
    results = {}
    
    # Time orphan detection
    print("\n1️⃣  Testing detect_orphan_nodes...")
    orphan_result = time_function(detect_orphan_nodes, df)
    results['orphan_detection'] = orphan_result
    
    print(f"✅ Orphan detection: {orphan_result['execution_time']:.3f}s")
    print(f"📊 Found {orphan_result['result_size']} orphan nodes")
    
    # Time missing red flags detection
    print("\n2️⃣  Testing detect_missing_red_flags...")
    redflag_result = time_function(detect_missing_red_flags, df)
    results['redflag_detection'] = redflag_result
    
    print(f"✅ Red flag detection: {redflag_result['execution_time']:.3f}s")
    print(f"📊 Found {redflag_result['result_size']} missing red flags")
    
    # Time full validation report
    print("\n3️⃣  Testing compute_validation_report...")
    report_result = time_function(compute_validation_report, df)
    results['full_report'] = report_result
    
    print(f"✅ Full validation report: {report_result['execution_time']:.3f}s")
    
    return results


def run_performance_harness():
    """Run the complete performance harness."""
    print("🚀 Decision Tree Performance Harness")
    print("=" * 60)
    print("Testing key functions on 10k-row synthetic dataset")
    print("=" * 60)
    
    # Load data
    df = load_synthetic_data()
    
    # Run benchmarks
    branch_results = benchmark_infer_branch_options(df)
    validation_results = benchmark_validation_summaries(df)
    
    # Summary
    print("\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY")
    print("="*60)
    
    total_time = branch_results['execution_time']
    for result in validation_results.values():
        total_time += result['execution_time']
    
    print(f"⏱️  Total execution time: {total_time:.3f} seconds")
    print(f"🌳 Branch options: {branch_results['execution_time']:.3f}s")
    print(f"🔍 Orphan detection: {validation_results['orphan_detection']['execution_time']:.3f}s")
    print(f"🚨 Red flag detection: {validation_results['redflag_detection']['execution_time']:.3f}s")
    print(f"📋 Full validation report: {validation_results['full_report']['execution_time']:.3f}s")
    
    # Performance assessment
    if total_time < 10:
        print("\n✅ EXCELLENT: All operations completed in under 10 seconds!")
    elif total_time < 30:
        print("\n🟡 GOOD: All operations completed in under 30 seconds")
    elif total_time < 60:
        print("\n🟠 ACCEPTABLE: All operations completed in under 1 minute")
    else:
        print("\n🔴 SLOW: Operations took over 1 minute - may need optimization")
    
    print(f"\n📈 Dataset size: {len(df):,} rows")
    print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    return {
        'branch_options': branch_results,
        'validation': validation_results,
        'total_time': total_time,
        'dataset_size': len(df)
    }


if __name__ == "__main__":
    run_performance_harness()
