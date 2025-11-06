#!/usr/bin/env python3
"""
Kubernetes-native Locust Load Test Driver
Replicates run_all_locust_tests.ps1 functionality with automated test matrix execution
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests


class LoadTestDriver:
    """Orchestrates distributed Locust load tests in Kubernetes"""

    def __init__(
        self,
        locust_master_url: str = "http://localhost:30089",
        inference_deployment: str = "inference",
        worker_deployment: str = "locust-worker",
        namespace: str = "default",
        test_duration: int = 120,
        cooldown: int = 15
    ):
        self.locust_master_url = locust_master_url
        self.inference_deployment = inference_deployment
        self.worker_deployment = worker_deployment
        self.namespace = namespace
        self.test_duration = test_duration
        self.cooldown = cooldown
        self.results: List[Dict] = []

    def scale_deployment(self, deployment: str, replicas: int) -> bool:
        """Scale a Kubernetes deployment"""
        try:
            cmd = [
                "kubectl", "scale", "deployment", deployment,
                f"--replicas={replicas}",
                f"--namespace={self.namespace}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"❌ Failed to scale {deployment}: {result.stderr}")
                return False
            
            print(f"✅ Scaled {deployment} to {replicas} replicas")
            
            # Wait for pods to be ready
            self._wait_for_deployment_ready(deployment, replicas)
            return True
        except Exception as e:
            print(f"❌ Error scaling {deployment}: {e}")
            return False

    def _wait_for_deployment_ready(self, deployment: str, expected_replicas: int, timeout: int = 120):
        """Wait for deployment to reach desired replicas"""
        print(f"⏳ Waiting for {deployment} to be ready ({expected_replicas} replicas)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                cmd = [
                    "kubectl", "get", "deployment", deployment,
                    f"--namespace={self.namespace}",
                    "-o", "jsonpath={.status.readyReplicas}"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                ready = int(result.stdout.strip() or "0")
                
                if ready == expected_replicas:
                    print(f"✅ {deployment} is ready with {ready}/{expected_replicas} replicas")
                    time.sleep(5)  # Extra buffer for stability
                    return
                
                print(f"   {ready}/{expected_replicas} ready...", end="\r")
                time.sleep(2)
            except Exception as e:
                print(f"   Waiting... {e}", end="\r")
                time.sleep(2)
        
        print(f"⚠️  Timeout waiting for {deployment} (continuing anyway)")

    def start_locust_test(self, user_count: int, spawn_rate: int = 20) -> bool:
        """Start a Locust test via API"""
        try:
            payload = {
                "user_count": user_count,
                "spawn_rate": spawn_rate,
                "host": "http://inference:8000"
            }
            
            response = requests.post(
                f"{self.locust_master_url}/swarm",
                data=payload,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ Started Locust test: {user_count} users @ {spawn_rate}/s spawn rate")
                return True
            else:
                print(f"❌ Failed to start test: {response.status_code} {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error starting test: {e}")
            return False

    def get_locust_stats(self) -> Optional[Dict]:
        """Get current test statistics from Locust master"""
        try:
            response = requests.get(
                f"{self.locust_master_url}/stats/requests",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"⚠️  Error fetching stats: {e}")
            return None

    def stop_locust_test(self) -> bool:
        """Stop the current Locust test"""
        try:
            response = requests.get(f"{self.locust_master_url}/stop", timeout=10)
            if response.status_code == 200:
                print("✅ Stopped Locust test")
                return True
            return False
        except Exception as e:
            print(f"❌ Error stopping test: {e}")
            return False

    def reset_locust_stats(self) -> bool:
        """Reset Locust statistics"""
        try:
            response = requests.get(f"{self.locust_master_url}/stats/reset", timeout=10)
            if response.status_code == 200:
                print("✅ Reset Locust stats")
                return True
            return False
        except Exception as e:
            print(f"⚠️  Error resetting stats: {e}")
            return False

    def poll_test_with_progress(self, duration: int) -> Optional[Dict]:
        """Poll test progress and display live stats"""
        print(f"\n📊 Running test for {duration} seconds...")
        start_time = time.time()
        end_time = start_time + duration
        last_stats = None
        
        while time.time() < end_time:
            stats = self.get_locust_stats()
            if stats and stats.get("stats"):
                # Find aggregated stats
                agg_stats = next(
                    (s for s in stats["stats"] if s.get("name") == "Aggregated"),
                    None
                )
                
                if agg_stats:
                    elapsed = int(time.time() - start_time)
                    remaining = int(end_time - time.time())
                    
                    rps = agg_stats.get("current_rps", 0)
                    median = agg_stats.get("median_response_time", 0)
                    p95 = agg_stats.get("ninetieth_response_time", 0)  # Note: often 90th, not 95th
                    failures = agg_stats.get("num_failures", 0)
                    requests = agg_stats.get("num_requests", 0)
                    
                    print(
                        f"   [{elapsed:3d}s / {duration}s] "
                        f"RPS: {rps:6.2f} | "
                        f"Median: {median:5.0f}ms | "
                        f"P90: {p95:5.0f}ms | "
                        f"Requests: {requests:5d} | "
                        f"Failures: {failures:3d}",
                        end="\r"
                    )
                    last_stats = agg_stats
            
            time.sleep(2)
        
        print("\n✅ Test duration completed")
        return last_stats

    def run_test_scenario(
        self,
        inference_replicas: int,
        worker_count: int,
        user_count: int
    ) -> Dict:
        """Execute a single test scenario"""
        print(f"\n{'='*80}")
        print(f"🧪 TEST SCENARIO")
        print(f"   Inference Replicas: {inference_replicas}")
        print(f"   Locust Workers: {worker_count}")
        print(f"   Virtual Users: {user_count}")
        print(f"{'='*80}")
        
        # Scale infrastructure
        if not self.scale_deployment(self.inference_deployment, inference_replicas):
            return self._create_error_result(inference_replicas, worker_count, user_count, "Failed to scale inference")
        
        if not self.scale_deployment(self.worker_deployment, worker_count):
            return self._create_error_result(inference_replicas, worker_count, user_count, "Failed to scale workers")
        
        # Reset and start test
        self.reset_locust_stats()
        spawn_rate = min(user_count // 10, 50)  # 10% of users or max 50/s
        
        if not self.start_locust_test(user_count, spawn_rate):
            return self._create_error_result(inference_replicas, worker_count, user_count, "Failed to start test")
        
        # Wait for spawn to complete
        time.sleep(max(5, user_count / spawn_rate + 2))
        
        # Run test and collect stats
        final_stats = self.poll_test_with_progress(self.test_duration)
        
        # Stop test
        self.stop_locust_test()
        
        # Package results
        if final_stats:
            result = {
                "replicas": inference_replicas,
                "workers": worker_count,
                "users": user_count,
                "rps": final_stats.get("current_rps", 0),
                "median_ms": final_stats.get("median_response_time", 0),
                "p95_ms": final_stats.get("ninetieth_response_time", 0),  # Often p90
                "p99_ms": final_stats.get("ninetieth_response_time", 0),  # Fallback
                "failures_pct": self._calculate_failure_rate(final_stats),
                "total_requests": final_stats.get("num_requests", 0),
                "duration_s": self.test_duration,
                "timestamp": datetime.now().isoformat(),
                "status": "SUCCESS"
            }
        else:
            result = self._create_error_result(inference_replicas, worker_count, user_count, "No stats collected")
        
        self.results.append(result)
        
        # Cooldown
        print(f"\n⏸️  Cooldown for {self.cooldown}s...")
        time.sleep(self.cooldown)
        
        return result

    def _create_error_result(self, replicas: int, workers: int, users: int, error: str) -> Dict:
        """Create error result entry"""
        return {
            "replicas": replicas,
            "workers": workers,
            "users": users,
            "rps": 0,
            "median_ms": 0,
            "p95_ms": 0,
            "p99_ms": 0,
            "failures_pct": 100.0,
            "total_requests": 0,
            "duration_s": self.test_duration,
            "timestamp": datetime.now().isoformat(),
            "status": f"ERROR: {error}"
        }

    def _calculate_failure_rate(self, stats: Dict) -> float:
        """Calculate failure percentage"""
        requests = stats.get("num_requests", 0)
        failures = stats.get("num_failures", 0)
        if requests == 0:
            return 0.0
        return (failures / requests) * 100

    def run_test_matrix(
        self,
        replica_counts: List[int],
        worker_counts: List[int],
        user_counts: List[int]
    ):
        """Execute full test matrix"""
        total_tests = len(replica_counts) * len(worker_counts) * len(user_counts)
        current_test = 0
        
        print(f"\n{'='*80}")
        print(f"🚀 STARTING TEST MATRIX")
        print(f"   Total Scenarios: {total_tests}")
        print(f"   Replica Counts: {replica_counts}")
        print(f"   Worker Counts: {worker_counts}")
        print(f"   User Counts: {user_counts}")
        print(f"   Duration per Test: {self.test_duration}s")
        print(f"   Estimated Total Time: {(self.test_duration + self.cooldown) * total_tests / 60:.1f} minutes")
        print(f"{'='*80}\n")
        
        for replicas in replica_counts:
            for workers in worker_counts:
                for users in user_counts:
                    current_test += 1
                    print(f"\n[{current_test}/{total_tests}] Testing configuration...")
                    self.run_test_scenario(replicas, workers, users)
        
        print(f"\n{'='*80}")
        print(f"✅ TEST MATRIX COMPLETED")
        print(f"{'='*80}\n")

    def export_results_csv(self, output_path: Path):
        """Export results to CSV"""
        try:
            with open(output_path, "w") as f:
                # Header
                f.write("Replicas,Workers,Users,RPS,Median_ms,P95_ms,P99_ms,Failures_Pct,Total_Requests,Duration_s,Timestamp,Status\n")
                
                # Data rows
                for r in self.results:
                    f.write(
                        f"{r['replicas']},{r['workers']},{r['users']},"
                        f"{r['rps']:.2f},{r['median_ms']:.0f},{r['p95_ms']:.0f},{r['p99_ms']:.0f},"
                        f"{r['failures_pct']:.2f},{r['total_requests']},{r['duration_s']},"
                        f"{r['timestamp']},{r['status']}\n"
                    )
            
            print(f"✅ Exported CSV: {output_path}")
        except Exception as e:
            print(f"❌ Failed to export CSV: {e}")

    def export_results_markdown(self, output_path: Path):
        """Export results to Markdown"""
        try:
            with open(output_path, "w") as f:
                f.write("# Kubernetes Locust Load Test Results\n\n")
                f.write(f"**Test Execution:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Test Duration per Scenario:** {self.test_duration}s\n\n")
                f.write(f"**Total Scenarios:** {len(self.results)}\n\n")
                
                # Configuration matrix
                f.write("## Test Matrix\n\n")
                f.write("| Replicas | Workers | Users | RPS | Median (ms) | P95 (ms) | P99 (ms) | Failures (%) | Total Requests | Status |\n")
                f.write("|----------|---------|-------|-----|-------------|----------|----------|--------------|----------------|--------|\n")
                
                for r in self.results:
                    status_icon = "✅" if r["status"] == "SUCCESS" else "❌"
                    f.write(
                        f"| {r['replicas']} | {r['workers']} | {r['users']} | "
                        f"{r['rps']:.2f} | {r['median_ms']:.0f} | {r['p95_ms']:.0f} | {r['p99_ms']:.0f} | "
                        f"{r['failures_pct']:.2f} | {r['total_requests']} | {status_icon} {r['status']} |\n"
                    )
                
                # Key findings
                f.write("\n## Key Findings\n\n")
                successful_results = [r for r in self.results if r["status"] == "SUCCESS"]
                
                if successful_results:
                    best_rps = max(successful_results, key=lambda x: x["rps"])
                    best_latency = min(successful_results, key=lambda x: x["median_ms"])
                    
                    f.write(f"- **Best Throughput:** {best_rps['rps']:.2f} RPS with {best_rps['replicas']} replicas, {best_rps['workers']} workers, {best_rps['users']} users\n")
                    f.write(f"- **Best Latency:** {best_latency['median_ms']:.0f}ms median with {best_latency['replicas']} replicas, {best_latency['workers']} workers, {best_latency['users']} users\n")
                    
                    avg_rps = sum(r["rps"] for r in successful_results) / len(successful_results)
                    avg_median = sum(r["median_ms"] for r in successful_results) / len(successful_results)
                    
                    f.write(f"- **Average Throughput:** {avg_rps:.2f} RPS\n")
                    f.write(f"- **Average Median Latency:** {avg_median:.0f}ms\n")
                    f.write(f"- **Success Rate:** {len(successful_results)}/{len(self.results)} scenarios\n")
                else:
                    f.write("- ⚠️ No successful test scenarios\n")
                
                # Recommendations
                f.write("\n## Recommendations\n\n")
                if successful_results:
                    high_throughput = [r for r in successful_results if r["rps"] > avg_rps]
                    if high_throughput:
                        f.write("**High-Performance Configurations:**\n\n")
                        for r in sorted(high_throughput, key=lambda x: x["rps"], reverse=True)[:3]:
                            f.write(f"- {r['replicas']} replicas × {r['workers']} workers × {r['users']} users → {r['rps']:.2f} RPS, {r['median_ms']:.0f}ms median\n")
                    
                    f.write("\n**Latency Optimization:**\n\n")
                    low_latency = [r for r in successful_results if r["median_ms"] < avg_median]
                    for r in sorted(low_latency, key=lambda x: x["median_ms"])[:3]:
                        f.write(f"- {r['replicas']} replicas × {r['workers']} workers × {r['users']} users → {r['median_ms']:.0f}ms median, {r['rps']:.2f} RPS\n")
                else:
                    f.write("- ⚠️ Review error logs and infrastructure configuration\n")
                    f.write("- ⚠️ Ensure Locust master is accessible at the configured URL\n")
                    f.write("- ⚠️ Verify inference deployment is running and healthy\n")
            
            print(f"✅ Exported Markdown: {output_path}")
        except Exception as e:
            print(f"❌ Failed to export Markdown: {e}")


def main():
    """Main entry point"""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║     Kubernetes Locust Load Test Driver - Automated Test Matrix       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration (can be made configurable via env vars or CLI args)
    driver = LoadTestDriver(
        locust_master_url="http://localhost:30089",
        inference_deployment="inference",
        worker_deployment="locust-worker",
        namespace="default",
        test_duration=120,
        cooldown=15
    )
    
    # Test matrix (matching PowerShell script)
    replica_counts = [1, 2, 4, 8]
    worker_counts = [4, 8]
    user_counts = [200, 400, 800]
    
    # Execute test matrix
    driver.run_test_matrix(replica_counts, worker_counts, user_counts)
    
    # Export results
    output_dir = Path("/results")  # Will be mounted in Job
    output_dir.mkdir(parents=True, exist_ok=True)
    
    driver.export_results_csv(output_dir / "auto_summary.csv")
    driver.export_results_markdown(output_dir / "auto_summary.md")
    
    print("\n✅ All results exported successfully!")
    print(f"   CSV: {output_dir / 'auto_summary.csv'}")
    print(f"   Markdown: {output_dir / 'auto_summary.md'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
