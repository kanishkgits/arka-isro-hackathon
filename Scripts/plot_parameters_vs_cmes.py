import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from matplotlib.colors import LogNorm
import numpy as np

def plot(data, cme_times):
    plt.figure(figsize=(20, 5)) 

    # Plot number density
    plt.plot(data['time'], data['proton_density'], label='Proton Density', linewidth = 0.5)
    plt.plot(data['time'], data['alpha_density'], label='Alpha Density', linewidth = 0.5)

    # Plot CME times as vertical lines
    for idx, t in enumerate(cme_times):
        plt.axvline(x=t, color='red', linestyle='--', linewidth=0.8, label="CME Event" if idx == 0 else "")

    # Formatting
    plt.xlabel("Time")
    plt.ylabel("Number Density [#/cm³]")

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 5)) 

    # Plot number density
    plt.plot(data['time'], data['proton_bulk_speed'], label='Proton Bulk Speed', linewidth = 0.5)
    plt.plot(data['time'], data['alpha_bulk_speed'], label='Alpha Bulk Speed', linewidth = 0.5)

    # Plot CME times as vertical lines
    for idx, t in enumerate(cme_times):
        plt.axvline(x=t, color='red', linestyle='--', linewidth=0.8, label="CME Event" if idx == 0 else "")

    # Formatting
    plt.xlabel("Time")
    plt.ylabel("Speed [km/s]")

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 5)) 

    # Plot number density
    plt.plot(data['time'], data['proton_thermal'], label='Proton Thermal', linewidth = 0.5)
    plt.plot(data['time'], data['alpha_thermal'], label='Alpha Thermal', linewidth = 0.5)

    # Plot CME times as vertical lines
    for idx, t in enumerate(cme_times):
        plt.axvline(x=t, color='red', linestyle='--', linewidth=0.8, label="CME Event" if idx == 0 else "")

    # Formatting
    plt.xlabel("Time")
    plt.ylabel("Thermal speed [Km/sec]")

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 5)) 
    plt.plot(data['time'], data['proton_xvelocity'], label='Proton X-velocity', linewidth = 0.5)
    for idx, t in enumerate(cme_times):
        plt.axvline(x=t, color='red', linestyle='--', linewidth=0.8, label="CME Event" if idx == 0 else "")
    plt.xlabel("Time")
    plt.ylabel("Speed [km/s]")
    plt.show()

    plt.figure(figsize=(20, 5)) 
    plt.plot(data['time'], data['proton_yvelocity'], label='Proton Y-velocity', linewidth = 0.5)
    for idx, t in enumerate(cme_times):
        plt.axvline(x=t, color='red', linestyle='--', linewidth=0.8, label="CME Event" if idx == 0 else "")
    plt.xlabel("Time")
    plt.ylabel("Speed [km/s]")
    plt.show()

    plt.figure(figsize=(20, 5)) 
    plt.plot(data['time'], data['proton_zvelocity'], label='Proton Z-velocity', linewidth = 0.5)
    for idx, t in enumerate(cme_times):
        plt.axvline(x=t, color='red', linestyle='--', linewidth=0.8, label="CME Event" if idx == 0 else "")
    plt.xlabel("Time")
    plt.ylabel("Speed [km/s]")
    plt.show()