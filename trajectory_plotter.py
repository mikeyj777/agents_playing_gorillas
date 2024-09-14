import pandas as pd
import matplotlib.pyplot as plt

def plot_trajectory(trajectory_df):
    plt.ion()
    plt.figure(1)
    plt.clf()
    x = trajectory_df['position_x']
    y = trajectory_df['position_y']
    plt.scatter(x, y)
    plt.show()
    plt.pause(0.01)