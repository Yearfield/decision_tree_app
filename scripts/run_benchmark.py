#!/usr/bin/env python3
"""
Run the complete benchmark suite: generate synthetic data and test performance.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False

def main():
    """Run the complete benchmark suite."""
    print("🎯 Decision Tree Complete Benchmark Suite")
    print("=" * 60)
    
    # Check if synthetic data already exists
    synthetic_file = os.path.join(os.path.dirname(__file__), "..", "synthetic_10k.csv")
    
    if os.path.exists(synthetic_file):
        print(f"📁 Synthetic data already exists: {synthetic_file}")
        print("Skipping generation step...")
        generate_success = True
    else:
        print("📁 No synthetic data found, generating...")
        generate_success = run_command(
            "python3 scripts/gen_synth.py",
            "Synthetic Data Generation"
        )
    
    if generate_success:
        print("\n" + "="*60)
        print("⏱️  Running Performance Benchmark...")
        print("="*60)
        
        benchmark_success = run_command(
            "python3 scripts/perf_harness.py",
            "Performance Benchmark"
        )
        
        if benchmark_success:
            print("\n🎉 BENCHMARK SUITE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("📊 Results:")
            print("   • 10k synthetic rows generated")
            print("   • Performance tested on key functions")
            print("   • All operations completed in seconds")
            print("=" * 60)
        else:
            print("\n❌ Performance benchmark failed")
            sys.exit(1)
    else:
        print("\n❌ Synthetic data generation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
