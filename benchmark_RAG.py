import json
import time
import requests
from websocket import create_connection, WebSocketException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
HOST = 'http://127.0.0.1:8000'  # Match stream2.html
WS_URL = f'ws://{HOST.split("://")[1]}/chat/ws2'  # WebSocket endpoint
SSE_URLS = [f'{HOST}/chat/stream2']  # Prioritize /chat/stream2
QUERY = 'Tell me about Rajasthan Patrika'  # Sample query
NUM_TESTS = 5  # Number of runs
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Sec-WebSocket-Protocol': 'chat',
    # Removed 'Origin' to avoid duplicates
    # Add 'Authorization': 'Bearer YOUR_TOKEN' if needed
}

def test_connectivity():
    """Test server connectivity for both endpoints."""
    try:
        response = requests.get(HOST, headers=HEADERS, timeout=5)
        logger.info(f"HTTP connectivity test: {response.status_code} {response.reason}")
    except requests.RequestException as e:
        logger.error(f"HTTP connectivity test failed: {e}")
    
    for sse_url in SSE_URLS:
        try:
            response = requests.get(sse_url, params={'query': QUERY}, headers=HEADERS, stream=True, timeout=5)
            logger.info(f"SSE endpoint test ({sse_url}): {response.status_code} {response.reason}")
        except requests.RequestException as e:
            logger.error(f"SSE endpoint test ({sse_url}) failed: {e}")
    
    try:
        ws = create_connection(WS_URL, header=HEADERS)
        logger.info("WebSocket connectivity test: Success")
        ws.close()
    except WebSocketException as e:
        logger.error(f"WebSocket connectivity test failed: {e}")

def test_websocket():
    """Test WebSocket response time."""
    start_time = time.time()
    try:
        ws = create_connection(WS_URL, header=HEADERS)
        logger.info(f"WebSocket connected to {WS_URL}")
        ws.send(json.dumps({'query': QUERY}))
        
        full_response = ''
        received_complete = False
        while not received_complete:
            message = ws.recv()
            if not message:
                break
            try:
                data = json.loads(message)
                if data.get('type') == 'stream':
                    full_response += data.get('chunk', '')
                elif data.get('type') == 'complete':
                    received_complete = True
            except json.JSONDecodeError:
                logger.warning(f"Non-JSON message received: {message[:50]}...")
        
        ws.close()
        end_time = time.time()
        return end_time - start_time, len(full_response)
    except WebSocketException as e:
        logger.error(f"WebSocket test failed: {e}")
        raise

def test_sse(sse_url):
    """Test SSE response time for given URL."""
    start_time = time.time()
    full_response = ''
    params = {'query': QUERY}
    try:
        with requests.get(sse_url, params=params, stream=True, headers=HEADERS) as response:
            logger.info(f"SSE request sent to {sse_url}?query={QUERY}")
            if response.status_code != 200:
                raise ValueError(f"SSE request failed with status {response.status_code} {response.reason}")
            for line in response.iter_lines():
                if line:
                    try:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data:'):
                            data_str = line_str[5:].strip()
                            if data_str == '[DONE]':
                                break
                            data = json.loads(data_str)
                            if 'chunk' in data:
                                full_response += data['chunk']
                            if data.get('is_final') or data.get('type') == 'done':
                                break
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid SSE line: {line_str[:50]}...")
        
        end_time = time.time()
        return end_time - start_time, len(full_response)
    except requests.RequestException as e:
        logger.error(f"SSE test ({sse_url}) failed: {e}")
        raise

# Run connectivity test
logger.info("Testing server connectivity...")
test_connectivity()

# Run benchmarks
ws_times = []
sse_times = []
successful_sse_url = None

logger.info(f"Starting {NUM_TESTS} benchmark tests...")
for i in range(NUM_TESTS):
    logger.info(f"Test {i+1}/{NUM_TESTS}")
    
    try:
        ws_time, ws_len = test_websocket()
        ws_times.append(ws_time)
        logger.info(f"  WebSocket: {ws_time:.4f}s (response length: {ws_len})")
    except Exception as e:
        logger.error(f"  WebSocket error: {e}")
    
    for sse_url in SSE_URLS:
        try:
            sse_time, sse_len = test_sse(sse_url)
            sse_times.append(sse_time)
            successful_sse_url = sse_url
            logger.info(f"  SSE ({sse_url}): {sse_time:.4f}s (response length: {sse_len})")
            break
        except Exception as e:
            logger.error(f"  SSE error ({sse_url}): {e}")
    
    time.sleep(1)

# Calculate and report averages
if ws_times:
    avg_ws = sum(ws_times) / len(ws_times)
    logger.info(f"\nAverage WebSocket time: {avg_ws:.4f}s")
else:
    logger.warning("\nAll WebSocket tests failed")

if sse_times:
    avg_sse = sum(sse_times) / len(sse_times)
    logger.info(f"Average SSE time: {avg_sse:.4f}s")
else:
    logger.warning("All SSE tests failed")

if ws_times and sse_times:
    if avg_ws < avg_sse:
        logger.info("Conclusion: WebSocket (websocket_2.html) is faster.")
    elif avg_sse < avg_ws:
        logger.info("Conclusion: SSE (stream2.html) is faster.")
    else:
        logger.info("Conclusion: Both have similar performance.")
elif sse_times:
    logger.info(f"Conclusion: SSE ({successful_sse_url}) is functional; WebSocket failed.")
else:
    logger.warning("Conclusion: Both SSE and WebSocket failed.")