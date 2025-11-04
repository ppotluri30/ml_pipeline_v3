#!/usr/bin/env python3
"""
Stress Test Script for FLTS Inference Service
Progressively tests with 10, 50, and 100 concurrent users
"""

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from collections import defaultdict
import sys

# Test configuration
INFERENCE_URL = "http://localhost/predict"
PAYLOAD_FILE = "payload-valid.json"

class LoadTestResults:
    def __init__(self, concurrent_users):
        self.concurrent_users = concurrent_users
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies = []
        self.errors = defaultdict(int)
        self.start_time = None
        self.end_time = None
        self.status_codes = defaultdict(int)
    
    def add_result(self, success, latency_ms, status_code=None, error=None):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.latencies.append(latency_ms)
        else:
            self.failed_requests += 1
            if error:
                self.errors[str(error)] += 1
        if status_code:
            self.status_codes[status_code] += 1
    
    def calculate_percentile(self, percentile):
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * percentile / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    def print_summary(self):
        duration_sec = (self.end_time - self.start_time)
        rps = self.successful_requests / duration_sec if duration_sec > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Load Test Results - {self.concurrent_users} Concurrent Users")
        print(f"{'='*70}")
        print(f"Total Requests:      {self.total_requests}")
        print(f"Successful:          {self.successful_requests} ({self.successful_requests/self.total_requests*100:.1f}%)")
        print(f"Failed:              {self.failed_requests} ({self.failed_requests/self.total_requests*100:.1f}%)")
        print(f"Duration:            {duration_sec:.2f} seconds")
        print(f"Requests/sec:        {rps:.2f}")
        print(f"\nLatency Statistics (ms):")
        print(f"  Min:               {min(self.latencies) if self.latencies else 0:.0f}")
        print(f"  Max:               {max(self.latencies) if self.latencies else 0:.0f}")
        print(f"  Mean:              {statistics.mean(self.latencies) if self.latencies else 0:.0f}")
        print(f"  Median (p50):      {self.calculate_percentile(50):.0f}")
        print(f"  p95:               {self.calculate_percentile(95):.0f}")
        print(f"  p99:               {self.calculate_percentile(99):.0f}")
        
        print(f"\nStatus Codes:")
        for code, count in sorted(self.status_codes.items()):
            print(f"  {code}: {count}")
        
        if self.errors:
            print(f"\nErrors:")
            for error, count in self.errors.items():
                print(f"  {error}: {count}")
        print(f"{'='*70}\n")
        
        return {
            "concurrent_users": self.concurrent_users,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "duration_sec": duration_sec,
            "rps": rps,
            "latency_min": min(self.latencies) if self.latencies else 0,
            "latency_max": max(self.latencies) if self.latencies else 0,
            "latency_mean": statistics.mean(self.latencies) if self.latencies else 0,
            "latency_p50": self.calculate_percentile(50),
            "latency_p95": self.calculate_percentile(95),
            "latency_p99": self.calculate_percentile(99),
            "status_codes": dict(self.status_codes),
            "errors": dict(self.errors)
        }


async def send_request(session, payload, results, semaphore):
    """Send a single prediction request"""
    async with semaphore:
        start_time = time.perf_counter()
        try:
            async with session.post(
                INFERENCE_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                await response.read()  # Consume response body
                latency_ms = (time.perf_counter() - start_time) * 1000
                results.add_result(
                    success=response.status == 200,
                    latency_ms=latency_ms,
                    status_code=response.status
                )
                return response.status
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            results.add_result(False, latency_ms, error="TimeoutError")
            return None
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            results.add_result(False, latency_ms, error=type(e).__name__)
            return None


async def run_load_test(concurrent_users, requests_per_user, payload):
    """Run load test with specified concurrency"""
    print(f"\n{'='*70}")
    print(f"Starting Load Test: {concurrent_users} concurrent users, {requests_per_user} requests each")
    print(f"Total requests: {concurrent_users * requests_per_user}")
    print(f"Target URL: {INFERENCE_URL}")
    print(f"{'='*70}\n")
    
    results = LoadTestResults(concurrent_users)
    semaphore = asyncio.Semaphore(concurrent_users)
    
    results.start_time = time.perf_counter()
    
    async with aiohttp.ClientSession() as session:
        # Create all tasks
        tasks = []
        for user in range(concurrent_users):
            for req in range(requests_per_user):
                task = asyncio.create_task(send_request(session, payload, results, semaphore))
                tasks.append(task)
        
        # Execute all tasks
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending {len(tasks)} requests...")
        await asyncio.gather(*tasks)
    
    results.end_time = time.perf_counter()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Test complete!")
    
    return results.print_summary()


def load_payload(filename):
    """Load test payload from file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Payload file '{filename}' not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in payload file: {e}")
        sys.exit(1)


async def main():
    """Run progressive load tests"""
    payload = load_payload(PAYLOAD_FILE)
    
    print("\n" + "="*70)
    print("FLTS Inference Service - Progressive Load Test")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Endpoint: {INFERENCE_URL}")
    print(f"Payload: {PAYLOAD_FILE}")
    print("="*70)
    
    # Test configurations: (concurrent_users, requests_per_user)
    test_configs = [
        (10, 2),   # 10 users × 2 requests = 20 total
        (50, 2),   # 50 users × 2 requests = 100 total
        (100, 2),  # 100 users × 2 requests = 200 total
    ]
    
    all_results = []
    
    for concurrent_users, requests_per_user in test_configs:
        result = await run_load_test(concurrent_users, requests_per_user, payload)
        all_results.append(result)
        
        # Brief pause between test phases
        if concurrent_users < 100:
            print(f"\nWaiting 10 seconds before next test phase...")
            await asyncio.sleep(10)
    
    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70)
    print(f"{'Users':<8} {'RPS':<10} {'p50(ms)':<10} {'p95(ms)':<10} {'p99(ms)':<10} {'Success%':<10}")
    print("-"*70)
    
    for result in all_results:
        success_pct = result['successful_requests'] / result['total_requests'] * 100 if result['total_requests'] > 0 else 0
        print(f"{result['concurrent_users']:<8} "
              f"{result['rps']:<10.2f} "
              f"{result['latency_p50']:<10.0f} "
              f"{result['latency_p95']:<10.0f} "
              f"{result['latency_p99']:<10.0f} "
              f"{success_pct:<10.1f}")
    
    print("="*70)
    
    # Identify degradation points
    print("\nPERFORMANCE ANALYSIS:")
    for i, result in enumerate(all_results):
        if i == 0:
            continue
        prev_result = all_results[i-1]
        
        # Check latency increase
        latency_increase = ((result['latency_p95'] - prev_result['latency_p95']) / prev_result['latency_p95'] * 100) if prev_result['latency_p95'] > 0 else 0
        
        # Check success rate decrease
        prev_success = prev_result['successful_requests'] / prev_result['total_requests'] * 100
        curr_success = result['successful_requests'] / result['total_requests'] * 100
        success_decrease = prev_success - curr_success
        
        print(f"\n{prev_result['concurrent_users']} → {result['concurrent_users']} users:")
        print(f"  - p95 latency change: {latency_increase:+.1f}%")
        print(f"  - Success rate change: {-success_decrease:+.1f}%")
        
        if latency_increase > 50:
            print(f"  ⚠️  WARNING: Significant latency degradation detected!")
        if success_decrease > 5:
            print(f"  ⚠️  WARNING: Error rate increasing!")
    
    print("\n" + "="*70)
    print("Load test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
