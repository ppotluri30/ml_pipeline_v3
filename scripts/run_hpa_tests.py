#!/usr/bin/env python3
"""
HPA Performance Testing Suite
Runs Locust headless tests and collects performance metrics
"""

import subprocess
import re
import time
import csv
from datetime import datetime

# Test scenarios
TESTS = [
    {"users": 10, "spawn": 2, "time": 30, "desc": "Light baseline"},
    {"users": 10, "spawn": 2, "time": 60, "desc": "Light sustained"},
    {"users": 25, "spawn": 5, "time": 30, "desc": "Medium quick"},
    {"users": 25, "spawn": 5, "time": 60, "desc": "Medium sustained"},
    {"users": 25, "spawn": 5, "time": 120, "desc": "Medium extended"},
    {"users": 50, "spawn": 10, "time": 30, "desc": "Heavy quick"},
    {"users": 50, "spawn": 10, "time": 60, "desc": "Heavy sustained"},
    {"users": 50, "spawn": 10, "time": 120, "desc": "Heavy extended"},
    {"users": 100, "spawn": 10, "time": 30, "desc": "Extreme quick"},
    {"users": 100, "spawn": 10, "time": 60, "desc": "Extreme sustained"},
    {"users": 100, "spawn": 10, "time": 120, "desc": "Extreme extended"},
]

OUTPUT_FILE = "reports/hpa_performance/HPA_PERFORMANCE_RESULTS.csv"

def run_locust_test(users, spawn_rate, duration):
    """Run a Locust test via kubectl exec"""
    cmd = [
        "kubectl", "exec", 
        "--stdin=false",
        "--tty=false",
        "deployment/locust-master", "--",
        "sh", "-c",
        f"cd /home/locust && locust -f locustfile.py --headless "
        f"--host=http://inference:8000 -u {users} -r {spawn_rate} -t {duration}s"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 60)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"  WARNING: Test timed out after {duration + 60} seconds")
        return ""
    except Exception as e:
        print(f"  ERROR: {e}")
        return ""

def parse_metrics(output):
    """Extract metrics from Locust output"""
    metrics = {
        "total_requests": 0,
        "failures": 0,
        "avg_latency": 0,
        "p95_latency": 0,
        "throughput": 0.0
    }
    
    # Find final "Aggregated" stats line
    # Format: "Aggregated   159   0(0.00%) |  147  73  559  120 |  5.60   0.00"
    agg_match = re.search(
        r'Aggregated\s+(\d+)\s+(\d+)\([^\)]+\)\s+\|\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\|\s+([\d\.]+)',
        output
    )
    if agg_match:
        metrics["total_requests"] = int(agg_match.group(1))
        metrics["failures"] = int(agg_match.group(2))
        metrics["avg_latency"] = int(agg_match.group(3))
        metrics["throughput"] = float(agg_match.group(4))
    
    # Find P95 from percentiles table (6th column)
    # Format: "Aggregated   120  120  140  150  170  180  210  210  210  210  210   44"
    p95_match = re.search(
        r'Aggregated\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)',
        output
    )
    if p95_match:
        metrics["p95_latency"] = int(p95_match.group(1))
    
    return metrics

def main():
    print("=" * 80)
    print("HPA PERFORMANCE TESTING SUITE")
    print("=" * 80)
    print(f"\nTest Matrix: {len(TESTS)} scenarios\n")
    
    # Create CSV file
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "TestID", "Users", "SpawnRate", "Duration", "Description",
            "TotalRequests", "Failures", "AvgLatency", "P95Latency", "Throughput"
        ])
    
    results = []
    for i, test in enumerate(TESTS, 1):
        print("=" * 80)
        print(f"TEST {i} of {len(TESTS)}: {test['desc']}")
        print(f"Users: {test['users']} | Spawn: {test['spawn']}/s | Duration: {test['time']}s")
        print("=" * 80)
        
        # Run test
        print("Running Locust test...")
        output = run_locust_test(test['users'], test['spawn'], test['time'])
        
        if not output:
            print("  ERROR: No output from test")
            continue
        
        # Parse metrics
        print("Parsing results...")
        metrics = parse_metrics(output)
        
        # Display results
        print(f"  Requests: {metrics['total_requests']} | Failures: {metrics['failures']}")
        print(f"  Avg Latency: {metrics['avg_latency']}ms | P95: {metrics['p95_latency']}ms")
        print(f"  Throughput: {metrics['throughput']:.2f} req/s")
        
        # Save to CSV
        row = [
            i, test['users'], test['spawn'], test['time'], test['desc'],
            metrics['total_requests'], metrics['failures'],
            metrics['avg_latency'], metrics['p95_latency'], metrics['throughput']
        ]
        results.append(row)
        
        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        print()
    
    print("=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_FILE}\n")
    
    # Display summary table
    print("SUMMARY:")
    print(f"{'ID':<4} {'Users':<6} {'Spawn':<6} {'Time':<6} {'Requests':<10} {'Failures':<10} {'Avg(ms)':<8} {'P95(ms)':<8} {'RPS':<8}")
    print("-" * 80)
    for row in results:
        print(f"{row[0]:<4} {row[1]:<6} {row[2]:<6} {row[3]:<6} {row[5]:<10} {row[6]:<10} {row[7]:<8} {row[8]:<8} {row[9]:<8.2f}")

if __name__ == "__main__":
    main()
