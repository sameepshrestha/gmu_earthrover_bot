import math

# Coordinates in degrees
lat1, lon1 = 38.827616000000006, -77.3058606  # Target
lat2, lon2 = 38.827667236328125, -77.30570220947266  # Robot

# Convert to radians
phi1 = math.radians(lat1)
phi2 = math.radians(lat2)
delta_lambda = math.radians(lon1 - lon2)

# Bearing in radians
y = math.sin(delta_lambda) * math.cos(phi1)
x = math.cos(phi2) * math.sin(phi1) - math.sin(phi2) * math.cos(phi1) * math.cos(delta_lambda)
bearing_rad = math.atan2(y, x)

# Convert to degrees and normalize
bearing_deg = math.degrees(bearing_rad)
bearing_east = (90 - bearing_deg + 360) % 360

print(round(bearing_east, 2))
