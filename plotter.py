import pandas as pd
import matplotlib.pyplot as plt

def plot_trajectory(trajectory_data):
    trajectory_df = pd.DataFrame(trajectory_data)
    x = trajectory_df['position_x']
    y = trajectory_df['position_y']
    plt.scatter(x, y)
    plt.show()