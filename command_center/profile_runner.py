import os
import time
import sys
import threading
from pathlib import Path

# Enable profiling
os.environ["GRIDSIGHT_PROFILE"] = "1"
os.environ["ASTRAM_USE_PLACEHOLDER"] = "0"  # Ensure we use the real model if available

import uvicorn
from streamlit.testing.v1 import AppTest

def run_profiling():
    print("Starting Streamlit AppTest profiling...")
    app_path = str(Path(__file__).parent / "frontend" / "app.py")
    
    # Init AppTest
    start_time = time.perf_counter()
    at = AppTest.from_file(app_path).run(timeout=60)
    print(f"Initial Page Load (App.py): {time.perf_counter() - start_time:.2f}s")
    
    # Navigate to Incident Command Center
    print("Navigating to Incident Command Center...")
    page2_path = str(Path(__file__).parent / "frontend" / "pages" / "2_Incident_Command_Center.py")
    start_time = time.perf_counter()
    at_p2 = AppTest.from_file(page2_path).run(timeout=60)
    print(f"Page 2 Initial Load: {time.perf_counter() - start_time:.2f}s")

    # Simulate Rapid Clicks to test Cache
    # We find the selectbox for incidents
    if len(at_p2.selectbox) > 0:
        # First click
        print("Clicking Incident 1 (Cold cache)...")
        start_time = time.perf_counter()
        at_p2.selectbox[0].select_index(1).run(timeout=60)
        print(f"Click 1 Rerun Time: {time.perf_counter() - start_time:.2f}s")

        # Second click
        print("Clicking Incident 2...")
        start_time = time.perf_counter()
        at_p2.selectbox[0].select_index(2).run(timeout=60)
        print(f"Click 2 Rerun Time: {time.perf_counter() - start_time:.2f}s")

        # Third click (Same incident to test predict cache and mapmyindia cache)
        print("Clicking Incident 2 AGAIN (Hot cache)...")
        start_time = time.perf_counter()
        at_p2.selectbox[0].select_index(2).run(timeout=60)
        print(f"Click 3 Rerun Time (Hot): {time.perf_counter() - start_time:.2f}s")
        
        # Fourth click
        print("Clicking Incident 3...")
        start_time = time.perf_counter()
        at_p2.selectbox[0].select_index(3).run(timeout=60)
        print(f"Click 4 Rerun Time: {time.perf_counter() - start_time:.2f}s")
    else:
        print("Selectbox not found in Page 2")

    print("Profiling complete. Stats dumped to cache/profiling_results.json.")
    # Exit to dump stats
    os._exit(0)

if __name__ == "__main__":
    run_profiling()
