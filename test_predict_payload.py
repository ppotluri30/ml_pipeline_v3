#!/usr/bin/env python3
"""Test script to send a valid predict payload and diagnose timestamp parsing"""

import datetime as dt
import json
import requests

def generate_test_payload():
    """Generate a test payload with unique timestamps"""
    base_time = dt.datetime(2025, 11, 8, 1, 0, 0)
    rows = 30
    
    # Generate unique timestamps (2-minute intervals)
    times = [(base_time + dt.timedelta(minutes=i*2)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(rows)]
    
    # Create data matching expected schema
    data = {
        "time": times,
        "down": [5000000.0 + i * 10000 for i in range(rows)],
        "up": [1000.0 + i * 10 for i in range(rows)],
        "rnti_count": [2000.0 + i for i in range(rows)],
        "mcs_down": [10.0 + (i % 5) for i in range(rows)],
        "mcs_down_var": [50.0 + i * 0.5 for i in range(rows)],
        "mcs_up": [12.0 + (i % 4) for i in range(rows)],
        "mcs_up_var": [40.0 + i * 0.4 for i in range(rows)],
        "rb_down": [0.05 + i * 0.001 for i in range(rows)],
        "rb_down_var": [1e-7 + i * 1e-9 for i in range(rows)],
        "rb_up": [0.01 + i * 0.0005 for i in range(rows)],
        "rb_up_var": [5e-8 + i * 1e-9 for i in range(rows)],
    }
    
    payload = {
        "index_col": "time",
        "data": data,
        "inference_length": 1
    }
    
    return payload

def test_inference_api(host="http://localhost:8000"):
    """Send test payload to inference API"""
    payload = generate_test_payload()
    
    print("=" * 80)
    print("TEST PAYLOAD SUMMARY")
    print("=" * 80)
    print(f"Rows: {len(payload['data']['time'])}")
    print(f"Unique timestamps: {len(set(payload['data']['time']))}")
    print(f"First 5 timestamps: {payload['data']['time'][:5]}")
    print(f"Last 5 timestamps: {payload['data']['time'][-5:]}")
    print(f"Payload size: {len(json.dumps(payload))} bytes")
    print()
    
    try:
        print(f"Sending POST to {host}/predict...")
        response = requests.post(f"{host}/predict", json=payload, timeout=10)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
        if response.status_code == 200:
            print("✅ SUCCESS")
            result = response.json()
            if "predictions" in result:
                print(f"Predictions: {len(result['predictions'])} values")
        else:
            print(f"❌ FAILED: {response.status_code}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_inference_api(host)
