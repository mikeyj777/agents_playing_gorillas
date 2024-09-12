import numpy as np

class Consts:
    GRAVITY = 9.8  # m/s^2
    AIR_DENSITY = 1.225  # kg/m^3 (typical at sea level)
    BANANA_DRAG_COEFFICIENT = 0.04 # Drag coefficient for a sphere
    BANANA_CROSS_SECTIONAL_AREA = 0.05  # Cross-sectional area of banana (approximation)
    BANANA_MASS = 0.15  # Mass of the banana (in kg)
    TRAJECTORY_TIME_STEP = 0.1
    IMPACT_TOLERANCE = 1.0
    X_BOUNDS = {
        'min': 0,
        'max': 300
    }
    # X_BOUNDS = {
    #     'min': -np.inf,
    #     'max': np.inf
    # }
    Y_BOUNDS = {
        'min': 0,
        'max': 300
    }
    # Y_BOUNDS = {
    #     'min': -np.inf,
    #     'max': np.inf
    # }
    MIN_SPEED = 1e-8