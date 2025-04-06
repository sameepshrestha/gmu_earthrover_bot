import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from botreceiver import BotReceiver
from bot_controller import BotController
import time

# Base URL for bot communication
BASE_URL = "http://localhost:8000"

def fit_circle(x, y):
    """
    Fit a circle to the data points using least squares optimization.

    Args:
        x (array-like): X coordinates of the data points.
        y (array-like): Y coordinates of the data points.

    Returns:
        tuple: (rmse, xc, yc, R) where:
            - rmse: Root Mean Square Error of the circle fit.
            - xc, yc: Coordinates of the circle center.
            - R: Radius of the fitted circle.
    """
    x = np.array(x)
    y = np.array(y)

    def calc_R(xc, yc):
        """Calculate distances from points to center (xc, yc)."""
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_2(c):
        """Residuals function for least squares: distances minus their mean."""
        Ri = calc_R(c[0], c[1])
        return Ri - Ri.mean()

    # Initial estimate for the center is the mean of the data
    center_estimate = (np.mean(x), np.mean(y))
    result = least_squares(f_2, center_estimate)
    xc, yc = result.x

    # Calculate distances and radius for the optimized center
    Ri = calc_R(xc, yc)
    R = Ri.mean()

    # Compute RMSE
    residuals = Ri - R
    rmse = np.sqrt(np.sum(residuals**2) / len(x))

    return rmse, xc, yc, R

def evaluate_circle_fit(x_adjusted, y_adjusted):
    """
    Evaluate the circularity of a pair of axes using circle fitting.

    Args:
        x_adjusted (pd.Series): Mean-adjusted data for the X axis.
        y_adjusted (pd.Series): Mean-adjusted data for the Y axis.

    Returns:
        float: RMSE of the circle fit (lower indicates a better fit to a circle).
    """
    rmse, _, _, _ = fit_circle(x_adjusted, y_adjusted)
    return rmse

def calibrate_magnetometer(botreceiver,botcontroller):
    """
    Collect magnetometer data, select the best pair of axes using circle fitting,
    and return their calibration parameters.

    Returns:
        tuple: (name_x, name_y, avg_x, max_x, avg_y, max_y) where:
            - name_x, name_y: Names of the selected axes.
            - avg_x, avg_y: Means of the selected axes.
            - max_x, max_y: Maximum deviations for scaling.
    """

    # Data collection parameters
    collection_duration = 150  # seconds
    sample_interval = 0.1    # seconds (adjust based on hardware response)

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
            # Command the robot to spin
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

    # Calculate scaling factors (maximum deviation)
    max_mags0 = max(mags0_adjusted.max(), -mags0_adjusted.min())
    max_mags1 = max(mags1_adjusted.max(), -mags1_adjusted.min())
    max_mags2 = max(mags2_adjusted.max(), -mags2_adjusted.min())

    # Define pairs to evaluate
    pairs = [
        (0, 1, mags0_adjusted, mags1_adjusted, avg_mags0, max_mags0, avg_mags1, max_mags1),
        (1, 2, mags1_adjusted, mags2_adjusted, avg_mags1, max_mags1, avg_mags2, max_mags2),
        (0, 2, mags0_adjusted, mags2_adjusted, avg_mags0, max_mags0, avg_mags2, max_mags2)
    ]

    best_pair = None
    min_rmse = float('inf')

    # Evaluate each pair using circle fitting
    for name_x, name_y, x_adj, y_adj, avg_x, max_x, avg_y, max_y in pairs:
        rmse, xc, yc, R = fit_circle(x_adj, y_adj)
        # print(f"For {name_x} vs {name_y}: RMSE={rmse:.2f}, center=({xc:.2f}, {yc:.2f}), radius={R:.2f}")
        if rmse < min_rmse:
            min_rmse = rmse
            best_pair = (name_x, name_y, x_adj, y_adj, avg_x, max_x, avg_y, max_y)

    # Unpack the best pair
    name_x, name_y, _, _, avg_x, max_x, avg_y, max_y = best_pair
    # print(f"Selected axes: {name_x} and {name_y}")

    return name_x, name_y, avg_x, max_x, avg_y, max_y

def main():
    """
    Main function to run the magnetometer calibration.
    """

    botreceiver = BotReceiver(BASE_URL)
    botcontroller = BotController(BASE_URL)
    name_x, name_y, avg_x, max_x, avg_y, max_y = calibrate_magnetometer(botreceiver, botcontroller)
    print(f"Calibration complete. Selected axes: {name_x}, {name_y}")
    print(f"X-axis mean: {avg_x}, max deviation: {max_x}")
    print(f"Y-axis mean: {avg_y}, max deviation: {max_y}")

if __name__ == "__main__":
    main()