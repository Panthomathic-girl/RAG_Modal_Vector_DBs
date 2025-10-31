import requests
import time
import json
import statistics
import difflib  # For content diff

def stream_and_capture(url, query, num_runs=3):
    encoded_query = requests.utils.quote(query)
    full_url = f"{url}?query={encoded_query}"
    
    results = []
    for run in range(num_runs):
        start_time = time.time()
        first_chunk_time = None
        chunks = []
        full_text = ''
        support_message = None
        error = None
        intervals = []
        prev_time = start_time
        
        try:
            with requests.get(full_url, stream=True, timeout=30) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        curr_time = time.time()
                        if first_chunk_time is None:
                            first_chunk_time = curr_time - start_time
                        intervals.append(curr_time - prev_time)
                        prev_time = curr_time
                        
                        data_str = line.decode('utf-8').strip()
                        if data_str.startswith('data: '):
                            data_json = data_str[6:]
                            try:
                                data = json.loads(data_json)
                                if 'chunk' in data:
                                    chunk_text = data['chunk']
                                    chunks.append(chunk_text)
                                    full_text += chunk_text
                                if 'message' in data:
                                    full_text = data['message']
                                if 'supportMessage' in data:
                                    support_message = data['supportMessage']
                                if 'error' in data:
                                    error = data['error']
                            except json.JSONDecodeError:
                                continue
            total_time = time.time() - start_time
        except Exception as e:
            error = str(e)
            total_time = time.time() - start_time
        
        avg_interval = statistics.mean(intervals[1:]) if len(intervals) > 1 else 0
        
        results.append({
            'ttft': first_chunk_time or total_time,
            'total_time': total_time,
            'avg_chunk_interval': avg_interval,
            'full_text': full_text,
            'support_message': support_message,
            'error': error,
            'raw_chunks': chunks
        })
    
    # Averages
    avg_ttft = statistics.mean([r['ttft'] for r in results]) if results else float('nan')
    avg_total = statistics.mean([r['total_time'] for r in results]) if results else float('nan')
    avg_interval = statistics.mean([i for r in results for i in [r['avg_chunk_interval']]]) if results else float('nan')
    
    return {
        'avg_ttft': avg_ttft,
        'avg_total': avg_total,
        'avg_interval': avg_interval,
        'sample_result': results[0] if results else None,  # First run as sample
        'all_results': results
    }

# Query and endpoints
query = "what is the ad booking flow of the customer"
endpoint1 = "http://3.7.148.21:8000/chat/stream"  # stream_test.html
endpoint2 = "http://127.0.0.1:8000/chat/stream2"  # stream2.html

# Run
res1 = stream_and_capture(endpoint1, query)
res2 = stream_and_capture(endpoint2, query)

# Print
print("=== stream_test.html (Remote) ===")
print(f"Avg TTFT: {res1['avg_ttft']:.3f}s, Avg Total: {res1['avg_total']:.3f}s, Avg Interval: {res1['avg_interval']:.3f}s")
print("Sample Full Text:", res1['sample_result']['full_text'] if res1['sample_result'] else "Empty")
print("Sample Support:", json.dumps(res1['sample_result']['support_message'], indent=2) if res1['sample_result'] else "None")
print("Sample Error:", res1['sample_result']['error'] if res1['sample_result'] else "None")

print("\n=== stream2.html (Local) ===")
print(f"Avg TTFT: {res2['avg_ttft']:.3f}s, Avg Total: {res2['avg_total']:.3f}s, Avg Interval: {res2['avg_interval']:.3f}s")
print("Sample Full Text:", res2['sample_result']['full_text'] if res2['sample_result'] else "Empty")
print("Sample Support:", json.dumps(res2['sample_result']['support_message'], indent=2) if res2['sample_result'] else "None")
print("Sample Error:", res2['sample_result']['error'] if res2['sample_result'] else "None")

# Diff content if no errors
if res1['sample_result'] and res2['sample_result'] and not res1['sample_result']['error'] and not res2['sample_result']['error']:
    diff = difflib.unified_diff(
        res1['sample_result']['full_text'].splitlines(),
        res2['sample_result']['full_text'].splitlines(),
        fromfile='stream_test.html',
        tofile='stream2.html'
    )
    print("\nContent Diff:")
    print('\n'.join(diff))

# Conclusion
if res1['sample_result']['error'] or res2['sample_result']['error']:
    print("\nConclusion: Errors detected (e.g., quota for stream2)—fix backend. stream_test is working but slower.")
elif res2['avg_total'] < res1['avg_total'] and len(res2['sample_result']['full_text']) >= len(res1['sample_result']['full_text']):
    print("\nConclusion: stream2.html is faster and better (lower time, equal/more detailed content).")
else:
    print("\nConclusion: stream_test.html is faster or similar—check content relevance manually.")
