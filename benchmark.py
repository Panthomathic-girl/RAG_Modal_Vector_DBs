import time
import statistics
import requests
import json
import websocket  # pip install websocket-client
import logging
from urllib.parse import quote

# Configure logging to match benchmark_RAG.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
URL_BASE = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/chat/ws"
SSE_URL = f"{URL_BASE}/chat/stream"
QUERY = "Tell me about Rajasthan Patrika"  # Matches your logs
NUM_TESTS = 50  # Number of test runs
MAX_RETRIES = 3  # For 429 handling
BACKOFF_BASE = 2  # Seconds for exponential backoff
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Sec-WebSocket-Protocol': 'chat',
}

def test_connectivity():
    """Test server connectivity for both endpoints."""
    try:
        response = requests.get(URL_BASE, headers=HEADERS, timeout=5)
        logger.info(f"HTTP connectivity test: {response.status_code} {response.reason}")
    except requests.RequestException as e:
        logger.error(f"HTTP connectivity test failed: {e}")

    try:
        response = requests.get(f"{SSE_URL}?query={quote(QUERY)}", headers=HEADERS, stream=True, timeout=5)
        logger.info(f"SSE endpoint test ({SSE_URL}): {response.status_code} {response.reason}")
    except requests.RequestException as e:
        logger.error(f"SSE endpoint test ({SSE_URL}) failed: {e}")

    try:
        ws = websocket.create_connection(WS_URL, header=HEADERS)
        logger.info("WebSocket connectivity test: Success")
        ws.close()
    except websocket.WebSocketException as e:
        logger.error(f"WebSocket connectivity test failed: {e}")

def test_sse():
    """Test SSE response time and response length."""
    start = time.time()
    full_response = ""
    try:
        with requests.get(f"{SSE_URL}?query={quote(QUERY)}", stream=True, headers=HEADERS) as response:
            logger.info(f"SSE request sent to {SSE_URL}?query={quote(QUERY)}")
            if response.status_code != 200:
                raise ValueError(f"SSE request failed with status {response.status_code} {response.reason}")
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8').lstrip("data: ").strip()
                    try:
                        data = json.loads(decoded)
                        if 'chunk' in data:
                            full_response += data['chunk']
                        if 'message' in data or data.get('type') == 'done' or decoded == '[DONE]':
                            break
                        if 'error' in data and '429' in data['error']:
                            raise Exception("429 Quota Exceeded")
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid SSE line: {decoded[:50]}...")
        end = time.time()
        return end - start, len(full_response)
    except Exception as e:
        logger.error(f"SSE test failed: {e}")
        raise

def test_websocket():
    """Test WebSocket response time and response length."""
    start = time.time()
    full_response = ""
    try:
        ws = websocket.create_connection(WS_URL, header=HEADERS)
        logger.info(f"WebSocket connected to {WS_URL}")
        ws.send(json.dumps({"query": QUERY}))
        while True:
            msg = ws.recv()
            if not msg:
                break
            try:
                data = json.loads(msg)
                if data.get('type') == 'stream':
                    full_response += data.get('chunk', '')
                if data.get('type') == 'complete':
                    break
                if data.get('type') == 'error' and '429' in data.get('message', ''):
                    raise Exception("429 Quota Exceeded")
            except json.JSONDecodeError:
                logger.warning(f"Non-JSON message received: {msg[:50]}...")
        ws.close()
        end = time.time()
        return end - start, len(full_response)
    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")
        raise

if __name__ == "__main__":
    # Run connectivity test
    logger.info("Testing server connectivity...")
    test_connectivity()

    # Run benchmarks
    sse_times = []
    ws_times = []
    sse_lengths = []
    ws_lengths = []

    logger.info(f"Starting {NUM_TESTS} benchmark tests...")
    for i in range(NUM_TESTS):
        logger.info(f"Test {i+1}/{NUM_TESTS}")
        
        # Test SSE
        for attempt in range(MAX_RETRIES):
            try:
                sse_time, sse_len = test_sse()
                sse_times.append(sse_time)
                sse_lengths.append(sse_len)
                logger.info(f"  SSE ({SSE_URL}): {sse_time:.4f}s (response length: {sse_len})")
                break
            except Exception as e:
                if "429" in str(e):
                    wait_time = BACKOFF_BASE ** attempt
                    logger.info(f"  SSE attempt {attempt+1}/{MAX_RETRIES}: 429 error, waiting {wait_time}s")
                    time.sleep(wait_time)
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"  SSE test {i+1} failed after retries")
                else:
                    logger.error(f"  SSE error: {e}")
                    break

        # Test WebSocket
        for attempt in range(MAX_RETRIES):
            try:
                ws_time, ws_len = test_websocket()
                ws_times.append(ws_time)
                ws_lengths.append(ws_len)
                logger.info(f"  WebSocket ({WS_URL}): {ws_time:.4f}s (response length: {ws_len})")
                break
            except Exception as e:
                if "429" in str(e):
                    wait_time = BACKOFF_BASE ** attempt
                    logger.info(f"  WebSocket attempt {attempt+1}/{MAX_RETRIES}: 429 error, waiting {wait_time}s")
                    time.sleep(wait_time)
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"  WebSocket test {i+1} failed after retries")
                else:
                    logger.error(f"  WebSocket error: {e}")
                    break
        
        time.sleep(1)  # Avoid overwhelming the server

    # Calculate and report averages
    if ws_times:
        avg_ws = statistics.mean(ws_times)
        logger.info(f"\nAverage WebSocket time: {avg_ws:.4f}s")
    else:
        logger.warning("\nAll WebSocket tests failed")

    if sse_times:
        avg_sse = statistics.mean(sse_times)
        logger.info(f"Average SSE time: {avg_sse:.4f}s")
    else:
        logger.warning("All SSE tests failed")

    # Report conclusion
    if ws_times and sse_times:
        if avg_ws < avg_sse:
            logger.info("Conclusion: WebSocket (websocket_test.html) is faster.")
        elif avg_sse < avg_ws:
            logger.info("Conclusion: SSE (stream_test.html) is faster.")
        else:
            logger.info("Conclusion: Both have similar performance.")
    elif sse_times:
        logger.info("Conclusion: SSE (stream_test.html) is functional; WebSocket failed.")
    else:
        logger.warning("Conclusion: Both SSE and WebSocket failed.")