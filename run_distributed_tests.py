"""
Run distributed Locust tests against varying inference replica counts.
This script uses the Locust HTTP API to control the master and run headless tests.
"""

import requests
import time
import json
import subprocess
import sys
from pathlib import Path

LOCUST_MASTER_URL = "http://localhost:8089"
RESULTS_DIR_4W = Path("locust/results/distributed_v1")
RESULTS_DIR_8W = Path("locust/results/distributed_v1_8workers")

def wait_for_locust_ready():
    """Wait for Locust master to be ready."""
    for i in range(30):
        try:
            resp = requests.get(f"{LOCUST_MASTER_URL}/")
            if resp.status_code == 200:
                print("✓ Locust master is ready")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    print("✗ Locust master not ready after 60s")
    return False

def get_stats():
    """Get current stats from Locust."""
    try:
        resp = requests.get(f"{LOCUST_MASTER_URL}/stats/requests")
        return resp.json()
    except:
        return None

def start_test(users, spawn_rate):
    """Start a Locust test."""
    data = {
        "user_count": users,
        "spawn_rate": spawn_rate,
        "host": "http://inference-lb"
    }
    resp = requests.post(f"{LOCUST_MASTER_URL}/swarm", data=data)
    return resp.status_code == 200

def stop_test():
    """Stop the current Locust test."""
    try:
        resp = requests.get(f"{LOCUST_MASTER_URL}/stop")
        return resp.status_code == 200
    except:
        return False

def reset_stats():
    """Reset Locust statistics."""
    try:
        resp = requests.get(f"{LOCUST_MASTER_URL}/stats/reset")
        return resp.status_code == 200
    except:
        return False

def scale_inference(replicas):
    """Scale inference containers."""
    print(f"\n🔧 Scaling inference to {replicas} replicas...")
    cmd = f"docker compose up -d --scale inference={replicas} --no-recreate inference"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Scaled to {replicas} inference replicas")
        time.sleep(10)  # Wait for health checks
        return True
    else:
        print(f"✗ Failed to scale: {result.stderr}")
        return False

def run_test_sequence(replica_count, users, duration_seconds, results_dir, test_name):
    """Run a single test configuration."""
    print(f"\n{'='*60}")
    print(f"🧪 Test: {test_name}")
    print(f"   Inference replicas: {replica_count}")
    print(f"   Users: {users}")
    print(f"   Duration: {duration_seconds}s")
    print(f"{'='*60}")
    
    # Reset stats before starting
    reset_stats()
    time.sleep(2)
    
    # Start the test
    spawn_rate = min(users // 10, 50)  # Reasonable spawn rate
    print(f"▶ Starting test: {users} users, spawn rate {spawn_rate}/s...")
    if not start_test(users, spawn_rate):
        print("✗ Failed to start test")
        return False
    
    # Wait for ramp-up
    ramp_up_time = users / spawn_rate
    print(f"⏳ Ramping up ({ramp_up_time:.1f}s)...")
    time.sleep(ramp_up_time + 5)
    
    # Run for duration
    print(f"🏃 Running test for {duration_seconds}s...")
    time.sleep(duration_seconds)
    
    # Get final stats
    stats = get_stats()
    if stats:
        print(f"\n📊 Test Results:")
        for stat in stats.get('stats', []):
            if stat['name'] == 'Aggregated':
                print(f"   Total Requests: {stat['num_requests']}")
                print(f"   Failures: {stat['num_failures']} ({stat.get('fail_ratio', 0)*100:.2f}%)")
                print(f"   RPS: {stat.get('current_rps', 0):.2f}")
                print(f"   Median: {stat.get('median_response_time', 0)}ms")
                print(f"   P95: {stat.get('ninetyfifth_response_time', 0)}ms")
                print(f"   P99: {stat.get('ninetyninth_response_time', 0)}ms")
                break
    
    # Stop test
    print("⏹ Stopping test...")
    stop_test()
    time.sleep(5)
    
    return True

def main():
    print("🚀 Starting Distributed Locust Test Suite")
    print("="*60)
    
    # Wait for Locust to be ready
    if not wait_for_locust_ready():
        sys.exit(1)
    
    # Test configurations
    replica_counts = [1, 2, 4, 8]
    user_counts = [200, 400, 800]
    test_duration = 120  # 2 minutes
    
    # Phase 1: 4 workers (already running)
    print("\n" + "="*60)
    print("📍 PHASE 1: Tests with 4 Locust Workers")
    print("="*60)
    
    for replicas in replica_counts:
        if not scale_inference(replicas):
            continue
            
        for users in user_counts:
            test_name = f"replicas{replicas}_u{users}"
            run_test_sequence(
                replicas,
                users, 
                test_duration,
                RESULTS_DIR_4W,
                test_name
            )
            time.sleep(10)  # Cool down between tests
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  - {RESULTS_DIR_4W}")
    print("\n💡 Note: Parse CSV files from Locust web UI or /stats/requests endpoint")

if __name__ == "__main__":
    main()
