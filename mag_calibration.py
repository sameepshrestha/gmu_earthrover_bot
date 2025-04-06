import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import numpy as np
from botreceiver import BotReceiver
from botController import BotController

# Base URL for bot communication
BASE_URL = "http://localhost:8000"

def evaluate_circle_fit(x_adjusted, y_adjusted):
    """
    Evaluate how circular a scatter plot is by calculating the variance of distances from the center.
    
    Args:
        x_adjusted, y_adjusted: Mean-adjusted magnetometer data.
    
    Returns:
        Variance of distances (lower is more circular).
    """
    distances = np.sqrt(x_adjusted**2 + y_adjusted**2)
    return np.var(distances)

def main():
    # Initialize bot communication
    botreceiver = BotReceiver(BASE_URL)
    botcontroller = BotController(BASE_URL)
    
    # Data collection parameters
    collection_duration = 60  # seconds
    sample_interval = 0.1     # seconds (adjust based on hardware response)
    
    # Lists to store data
    timestamps = []
    mags0_list = []
    mags1_list = []
    mags2_list = []
    mags3_list = []
    
    # Collect data while spinning the robot
    print("Starting data collection... Spin the robot.")
    start_time = time.time()
    while time.time() - start_time < collection_duration:
        try:
            # Command the robot to spin (adjust angular velocity as needed)
            botcontroller.send_control_command(0.0, 0.11)  # Linear=0, Angular=0.11 rad/s
            data = botreceiver.fetch_bot_data()
            timestamp = data["timestamp"]
            mags = data["mags"][0]  # Assuming mags is a list of 4 values
            timestamps.append(timestamp)
            mags0_list.append(mags[0])
            mags1_list.append(mags[1])
            mags2_list.append(mags[2])
            mags3_list.append(mags[3])
            time.sleep(sample_interval)
        except Exception as e:
            print(f"Error during data collection: {e}")
            break
    
    print("Data collection complete.")
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'mags0': mags0_list,
        'mags1': mags1_list,
        'mags2': mags2_list,
        'mags3': mags3_list
    })


    
    # Calculate means for centering
    avg_mags0 = df['mags0'].mean()
    avg_mags1 = df['mags1'].mean()
    avg_mags2 = df['mags2'].mean()
    
    # Mean-adjusted data
    mags0_adjusted = df['mags0'] - avg_mags0
    mags1_adjusted = df['mags1'] - avg_mags1
    mags2_adjusted = df['mags2'] - avg_mags2
    
    # Calculate scaling factors (max deviation)
    max_mags0 = max(mags0_adjusted.max(), -mags0_adjusted.min())
    max_mags1 = max(mags1_adjusted.max(), -mags1_adjusted.min())
    max_mags2 = max(mags2_adjusted.max(), -mags2_adjusted.min())

    
    # Evaluate circularity for each pair
    pairs = [
        ('mags0', 'mags1', mags0_adjusted, mags1_adjusted, avg_mags0, max_mags0, avg_mags1, max_mags1),
        ('mags1', 'mags2', mags1_adjusted, mags2_adjusted, avg_mags1, max_mags1, avg_mags2, max_mags2),
        ('mags0', 'mags2', mags0_adjusted, mags2_adjusted, avg_mags0, max_mags0, avg_mags2, max_mags2)
    ]
    
    best_pair = None
    min_rmse = float('inf')
    
    for name_x, name_y, x_adj, y_adj, avg_x, max_x, avg_y, max_y in pairs:
        rmse = evaluate_circle_fit(x_adj, y_adj)
        print(f"RMSE for {name_x} vs {name_y}: {rmse}")
        if rmse < min_rmse:
            min_rmse = rmse
            best_pair = (name_x, name_y, x_adj, y_adj, avg_x, max_x, avg_y, max_y)

    
    # Unpack the best pair
    name_x, name_y, x_adj, y_adj, avg_x, max_x, avg_y, max_y = best_pair
    print(f"Selected axes: {name_x} and {name_y}")
    selected_x, selected_y = name_x, name_y

    
    return name_x, name_y, avg_x, max_x, avg_y, max_y

if __name__ == "__main__":
    main()