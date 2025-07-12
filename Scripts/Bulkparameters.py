import numpy as np
from spacepy.pycdf import CDF
import datetime as dt
import glob

def load_blk_variables(DATA_PATH, INVALID):
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
        all_data[key] = moving_average(all_data[key], 50)
        all_data[key] = min_max_scale(all_data[key])
    all_data['time'] =all_data['time'][49:]

    return all_data

def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def min_max_scale(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

