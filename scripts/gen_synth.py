#!/usr/bin/env python3
"""
Generate synthetic decision tree data for performance testing.
Creates a 10k-row DataFrame with realistic decision tree structure.
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import CANON_HEADERS, LEVEL_COLS


def generate_vital_measurements(count: int) -> List[str]:
    """Generate realistic vital measurement values."""
    vitals = [
        "Blood Pressure", "Temperature", "Heart Rate", "Respiratory Rate",
        "Oxygen Saturation", "Blood Glucose", "Pain Level", "Mental Status",
        "Skin Color", "Capillary Refill", "Pupil Response", "Motor Function",
        "Sensory Function", "Glasgow Coma Scale", "AVPU Score"
    ]
    return [random.choice(vitals) for _ in range(count)]


def generate_node_values(level: int, parent_path: List[str], count: int) -> List[str]:
    """Generate realistic node values based on level and parent context."""
    
    # Level 1: Severity/Status indicators
    if level == 1:
        options = ["High", "Low", "Normal", "Critical", "Moderate", "Severe", "Mild"]
    
    # Level 2: Specific conditions
    elif level == 2:
        if "Blood Pressure" in parent_path:
            options = ["Hypertension", "Hypotension", "Normal", "Pre-hypertension", "Crisis"]
        elif "Temperature" in parent_path:
            options = ["Fever", "Hypothermia", "Normal", "Hyperthermia", "Subnormal"]
        elif "Heart Rate" in parent_path:
            options = ["Tachycardia", "Bradycardia", "Normal", "Irregular", "Weak"]
        else:
            options = ["Acute", "Chronic", "Stable", "Unstable", "Progressive"]
    
    # Level 3: Assessment/Intervention
    elif level == 3:
        if "High" in parent_path or "Severe" in parent_path:
            options = ["Immediate", "Urgent", "Critical", "Emergency", "Resuscitation"]
        elif "Low" in parent_path or "Mild" in parent_path:
            options = ["Monitor", "Watch", "Observe", "Follow-up", "Routine"]
        else:
            options = ["Regular", "Standard", "Normal", "Baseline", "Maintenance"]
    
    # Level 4: Action priority
    elif level == 4:
        if "Immediate" in parent_path or "Urgent" in parent_path:
            options = ["Urgent", "Critical", "Emergency", "Rescue", "Stabilize"]
        elif "Monitor" in parent_path or "Watch" in parent_path:
            options = ["Watch", "Observe", "Check", "Assess", "Evaluate"]
        else:
            options = ["Standard", "Regular", "Normal", "Routine", "Maintenance"]
    
    # Level 5: Specific action
    elif level == 5:
        if "Urgent" in parent_path or "Critical" in parent_path:
            options = ["Call 911", "Emergency Response", "Code Blue", "Rapid Response", "Crash Cart"]
        elif "Watch" in parent_path or "Observe" in parent_path:
            options = ["Check again", "Reassess", "Monitor closely", "Follow protocol", "Document"]
        else:
            options = ["Follow up", "Routine care", "Standard protocol", "Maintenance", "Prevention"]
    
    else:
        options = ["Value1", "Value2", "Value3", "Value4", "Value5"]
    
    return [random.choice(options) for _ in range(count)]


def generate_diagnostic_triage(vital: str, severity: str) -> str:
    """Generate realistic diagnostic triage based on vital and severity."""
    if severity in ["High", "Severe", "Critical"]:
        return "Immediate"
    elif severity in ["Low", "Mild", "Moderate"]:
        return "Monitor"
    else:
        return "Routine"


def generate_actions(vital: str, severity: str, triage: str) -> str:
    """Generate realistic actions based on context."""
    if triage == "Immediate":
        return "Red flag - Immediate intervention required"
    elif triage == "Monitor":
        return "Continue monitoring - Watch for changes"
    else:
        return "Standard care - Follow routine protocol"


def generate_synthetic_data(rows: int = 10000) -> pd.DataFrame:
    """Generate synthetic decision tree data."""
    print(f"Generating {rows:,} rows of synthetic data...")
    
    data = {
        "Vital Measurement": [],
        "Node 1": [],
        "Node 2": [],
        "Node 3": [],
        "Node 4": [],
        "Node 5": [],
        "Diagnostic Triage": [],
        "Actions": []
    }
    
    # Generate vital measurements first
    vitals = generate_vital_measurements(rows)
    
    for i in range(rows):
        vital = vitals[i]
        
        # Generate node values with realistic relationships
        node1 = random.choice(["High", "Low", "Normal", "Critical", "Moderate"])
        node2 = generate_node_values(2, [vital], 1)[0]
        node3 = generate_node_values(3, [vital, node1, node2], 1)[0]
        node4 = generate_node_values(4, [vital, node1, node2, node3], 1)[0]
        node5 = generate_node_values(5, [vital, node1, node2, node3, node4], 1)[0]
        
        # Generate triage and actions
        triage = generate_diagnostic_triage(vital, node1)
        actions = generate_actions(vital, node1, triage)
        
        # Add to data
        data["Vital Measurement"].append(vital)
        data["Node 1"].append(node1)
        data["Node 2"].append(node2)
        data["Node 3"].append(node3)
        data["Node 4"].append(node4)
        data["Node 5"].append(node5)
        data["Diagnostic Triage"].append(triage)
        data["Actions"].append(actions)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1:,} rows...")
    
    df = pd.DataFrame(data)
    print(f"Generated DataFrame with shape: {df.shape}")
    return df


def save_to_csv(df: pd.DataFrame, filename: str = "synthetic_10k.csv"):
    """Save DataFrame to CSV file."""
    output_path = os.path.join(os.path.dirname(__file__), "..", filename)
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic data to: {output_path}")
    return output_path


def main():
    """Main function to generate and save synthetic data."""
    print("🎲 Synthetic Decision Tree Data Generator")
    print("=" * 50)
    
    # Generate 10k rows
    df = generate_synthetic_data(10000)
    
    # Show sample data
    print("\n📊 Sample Data:")
    print(df.head())
    
    # Show data info
    print("\n📈 Data Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show value counts for key columns
    print("\n🔍 Value Distribution:")
    print(f"Vital Measurements: {df['Vital Measurement'].nunique()} unique values")
    print(f"Node 1 values: {df['Node 1'].nunique()} unique values")
    print(f"Diagnostic Triage: {df['Diagnostic Triage'].nunique()} unique values")
    
    # Save to CSV
    output_path = save_to_csv(df)
    
    print(f"\n✅ Successfully generated {len(df):,} rows of synthetic data!")
    print(f"📁 File saved to: {output_path}")
    print(f"💾 File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
