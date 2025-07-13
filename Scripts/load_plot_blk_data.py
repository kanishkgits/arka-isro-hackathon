import numpy as np
from spacepy.pycdf import CDF
import datetime as dt
import glob

#loading bulk data from cdf files
def load_blk_variables(DATA_PATH, INVALID = -1e31):
    all_data = {
        'time': [],
        'proton_density': [],
        'proton_bulk_speed': [],
        'proton_thermal': [],
        'alpha_density': [],
        'alpha_bulk_speed': [],
        'alpha_thermal': [],
        'proton_xvelocity': [],
        'proton_yvelocity': [],
        'proton_zvelocity': [],
    }

    for file in sorted(glob.glob(DATA_PATH)):
        with CDF(file) as cdf:
            try:
                time = np.array([dt.datetime.fromtimestamp(e.timestamp(), dt.UTC) for e in cdf['epoch_for_cdf_mod'][:]])
                variables = {
                    'proton_density': np.array(cdf['proton_density'][:]),
                    'proton_bulk_speed': np.array(cdf['proton_bulk_speed'][:]),
                    'proton_thermal': np.array(cdf['proton_thermal'][:]),
                    'alpha_density': np.array(cdf['alpha_density'][:]),
                    'alpha_bulk_speed': np.array(cdf['alpha_bulk_speed'][:]),
                    'alpha_thermal': np.array(cdf['alpha_thermal'][:]),
                    'proton_xvelocity': np.array(cdf['proton_xvelocity'][:]),
                    'proton_yvelocity': np.array(cdf['proton_yvelocity'][:]),
                    'proton_zvelocity': np.array(cdf['proton_zvelocity'][:]),
                }
            except KeyError as e:
                print(f"Missing expected variable in {file}: {e}")
                continue

            # Create a valid data mask (based on proton vars)
            valid_mask = (
                (variables['proton_density'] != INVALID) &
                (variables['proton_bulk_speed'] != INVALID) &
                (variables['proton_thermal'] != INVALID) &
                (variables['alpha_density'] != INVALID) &
                (variables['alpha_bulk_speed'] != INVALID) &
                (variables['alpha_thermal'] != INVALID) &
                (variables['proton_xvelocity'] != INVALID) &
                (variables['proton_yvelocity'] != INVALID) &
                (variables['proton_zvelocity'] != INVALID)
            )

            all_data['time'].extend(time[valid_mask])
            for key in variables:
                all_data[key].extend(variables[key][valid_mask])

    # Convert all lists to NumPy arrays
    for key in all_data:
        all_data[key] = np.array(all_data[key])
    for key in all_data:
        if(key == 'time'):
            continue
        all_data[key] = moving_average(all_data[key], 10)
        all_data[key] = min_max_scale(all_data[key])
    all_data['time'] =all_data['time'][9:]

    return all_data

#scaling and smoothing
def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def min_max_scale(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

#plotting
import matplotlib.pyplot as plt

def plot_blk_parameters(data):
    plots = [
        ('proton_bulk_speed', 'Proton Bulk Speed'),
        ('alpha_bulk_speed', 'Alpha Bulk Speed', 'tab:orange'),
        ('proton_density', 'Proton Density'),
        ('alpha_density', 'Alpha Density', 'tab:orange'),
        ('proton_thermal', 'Proton Thermal'),
        ('alpha_thermal', 'Alpha Thermal', 'tab:orange'),
        ('proton_xvelocity', 'Proton X velocity'),
        ('proton_yvelocity', 'Proton Y velocity'),
        ('proton_zvelocity', 'Proton Z velocity'),
    ]

    for plot in plots:
        key, title = plot[0], plot[1]
        color = plot[2] if len(plot) == 3 else None

        plt.figure(figsize=(20, 5))
        plt.plot(data['time'], data[key], label=title, linewidth=0.5, color=color)
        plt.title(title)
        plt.xlabel("Time")
        plt.show()
