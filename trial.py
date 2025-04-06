import math

def calculate_bearing(point1, point2):
    """
    Calculate the bearing from point2 to point1 with East as 0°.
    
    Parameters:
        point1: Tuple (lat1, lon1) - Latitude and Longitude of target point in degrees.
        point2: Tuple (lat2, lon2) - Latitude and Longitude of robot position in degrees.
    
    Returns:
        Bearing in degrees with East as 0°.
    """
    lat1, lon1 = point1
    lat2, lon2 = point2

    # Convert degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon1 - lon2)

    # Compute bearing
    y = math.sin(delta_lambda) * math.cos(phi1)
    x = math.cos(phi2) * math.sin(phi1) - math.sin(phi2) * math.cos(phi1) * math.cos(delta_lambda)
    bearing_rad = math.atan2(y, x)

    # Convert to degrees and adjust to East = 0° system
    bearing_deg = math.degrees(bearing_rad)
    bearing_east =  (bearing_deg - 90 + 360) % 360 % 360

    return round(bearing_east, 2)

def get_bearing(bearing, data_orientation):
    delta_angle = (bearing - data_orientation + 360) % 360
    if delta_angle > 180:
        delta_angle -= 360

    steering = 0.1
    print(delta_angle, "delta angle", bearing, data_orientation)
    if abs(delta_angle) >= 30:
        Recalibrate = True
        if delta_angle < 0: # turn left
            steering *= -1 
        else:
            steering *= 1
    return steering

if __name__ == "__main__":
    # Example usage
    go_go_steering = get_bearing(181, 0)
    print(f"{go_go_steering = } degrees.")