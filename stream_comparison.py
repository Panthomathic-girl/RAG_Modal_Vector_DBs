import requests
import time
import json

def measure_streaming_performance(url, query, num_runs=3):
    """
    Measures streaming performance for an SSE endpoint.
    Returns dict with avg TTFT, total time, inter-chunk time.
    """
    encoded_query = requests.utils.quote(query)
    full_url = f"{url}?query={encoded_query}"
    
    times = {
        'ttft': [],  # Time to first token
        'total_time': [],
        'chunk_times': []  # List of lists for inter-chunk times
    }
    
    for run in range(num_runs):
        start_time = time.time()
        first_chunk_time = None
        prev_chunk_time = None
        chunk_intervals = []
        
        try:
            with requests.get(full_url, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        current_time = time.time()
                        
                        # Parse SSE line
                        if line.startswith(b'data: '):
                            data_str = line.decode('utf-8')[6:]
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            
                            # Record first chunk
                            if first_chunk_time is None and data.get('chunk'):
                                first_chunk_time = current_time - start_time
                            
                            # Record inter-chunk intervals
                            if prev_chunk_time is not None:
                                chunk_intervals.append(current_time - prev_chunk_time)
                            prev_chunk_time = current_time
                
                total_time = time.time() - start_time
                
                times['ttft'].append(first_chunk_time or total_time)
                times['total_time'].append(total_time)
                times['chunk_times'].append(chunk_intervals)
        
        except Exception as e:
            print(f"Error in run {run+1} for {url}: {str(e)}")
            continue
    
    # Compute averages
    if times['ttft']:
        avg_ttft = sum(times['ttft']) / len(times['ttft'])
        avg_total = sum(times['total_time']) / len(times['total_time'])
        all_intervals = [interval for run_intervals in times['chunk_times'] for interval in run_intervals]
        avg_inter_chunk = sum(all_intervals) / len(all_intervals) if all_intervals else 0
    else:
        avg_ttft = float('nan')
        avg_total = float('nan')
        avg_inter_chunk = float('nan')
    
    return {
        'avg_ttft': avg_ttft,
        'avg_total_time': avg_total,
        'avg_inter_chunk_time': avg_inter_chunk,
        'successful_runs': len(times['ttft'])
    }

# Test parameters (customize as needed)
query = "Explain the theory of relativity in simple terms"  # Sample query
endpoint1 = "http://3.7.148.21:8000/chat/stream"  # From stream_test.html
endpoint2 = "http://127.0.0.1:8000/chat/stream2"  # From stream2.html
num_runs = 3  # Number of test runs for averaging

# Run tests
print("Testing endpoint 1 (stream_test.html backend): " + endpoint1)
results1 = measure_streaming_performance(endpoint1, query, num_runs)

print("\nTesting endpoint 2 (stream2.html backend): " + endpoint2)
results2 = measure_streaming_performance(endpoint2, query, num_runs)

# Output comparison
print("\nPerformance Comparison:")
print(f"Endpoint 1 ({endpoint1}):")
print(f"  - Avg TTFT: {results1['avg_ttft']:.3f}s")
print(f"  - Avg Total Time: {results1['avg_total_time']:.3f}s")
print(f"  - Avg Inter-Chunk Time: {results1['avg_inter_chunk_time']:.3f}s")
print(f"  - Successful Runs: {results1['successful_runs']}/{num_runs}")

print(f"\nEndpoint 2 ({endpoint2}):")
print(f"  - Avg TTFT: {results2['avg_ttft']:.3f}s")
print(f"  - Avg Total Time: {results2['avg_total_time']:.3f}s")
print(f"  - Avg Inter-Chunk Time: {results2['avg_inter_chunk_time']:.3f}s")
print(f"  - Successful Runs: {results2['successful_runs']}/{num_runs}")

# Conclusion with file reference
if results1['successful_runs'] > 0 and results2['successful_runs'] > 0:
    if results1['avg_total_time'] < results2['avg_total_time']:
        print("\nConclusion: stream_test.html is faster overall (lower average total time).")
    elif results2['avg_total_time'] < results1['avg_total_time']:
        print("\nConclusion: stream2.html is faster overall (lower average total time).")
    else:
        print("\nConclusion: Both stream_test.html and stream2.html have similar performance.")
else:
    print("\nConclusion: Unable to compare due to connection issues. Check if servers are running.")
    print("Theoretical Note: Based on network alone (remote vs local), stream2.html (localhost) is expected to be faster due to zero latency.")