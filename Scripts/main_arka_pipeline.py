import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath('..'))
from Scripts.load_plot_blk_data import load_blk_variables
from Scripts.load_plot_cactus_data import parse_cactus_file, plot_cme
from Scripts.train_cme_classifier import train_cme_prediction_model, evaluate_cme_windows,  save_model
import seaborn as sns
import numpy as np
import pandas as pd
from Scripts.visualize_model_outputs import plot_weighted_signal

#The cool intro logo
def draw_arka_graph():

    letters = {
        'A': [
            [(0, 0), (0.5, 1), (1, 0)],        
            [(0.25, 0.5), (0.75, 0.5)]         
        ],
        'R': [
            [(0, 0), (0, 1)],
            [(0, 1), (0.6, 1), (0.6, 0.75), (0, 0.5)],
            [(0, 0.5), (0.6, 0)]
        ],
        'K': [
            [(0, 0), (0, 1)],
            [(0, 0.5), (0.6, 1)],
            [(0, 0.5), (0.6, 0)]
        ]
    }

    plt.figure(figsize=(12, 6))
    spacing = 2.0 
    color_cycle = ['blue', 'orange']
    line_width = 3

    for idx, letter in enumerate("ARKA"):
        strokes = letters[letter]
        x_offset = idx * spacing * 1.5
        for stroke_idx, stroke in enumerate(strokes):
            for i in range(len(stroke) - 1):
                (x0, y0) = stroke[i]
                (x1, y1) = stroke[i + 1]
                plt.plot(
                    [x0 + x_offset, x1 + x_offset],
                    [y0, y1],
                    color=color_cycle[(stroke_idx + i) % len(color_cycle)],
                    linewidth=line_width
                )

    plt.axis('equal')
    plt.axis('off')
    plt.show()

#plotting some bulk parameters(checkout the Bulk_Visualized notebook)
def bulk_plots():

    trainingData = load_blk_variables('data/Training_data/*.cdf')
    untrainedData = load_blk_variables('data/Untrained_data/*.cdf')
    data = {}
    for key in trainingData:
        data[key] = np.concatenate([trainingData[key], untrainedData[key]])

    plt.figure(figsize=(20, 5)) 
    plt.plot(data['time'], data['proton_bulk_speed'], label='Proton Bulk Speed', linewidth = 0.5)
    plt.title("Proton Bulk Speed")
    plt.xlabel("Time")
    plt.show()

    plt.figure(figsize=(20, 5)) 
    plt.plot(data['time'], data['proton_density'], label='Proton Density', linewidth = 0.5, color = 'orange')
    plt.title("Proton Density")
    plt.xlabel("Time")
    plt.show()

    plt.figure(figsize=(20, 5)) 
    plt.plot(data['time'], data['proton_thermal'], label='Proton Thermal', linewidth = 0.5)
    plt.title("Proton Thermal")
    plt.xlabel("Time")
    plt.show()

#plotting the Cactus CMEs
def cme_plots():
    cme_times = (parse_cactus_file('data/cmesept2024.txt') + 
                 parse_cactus_file('data/cmeoct2024.txt') + 
                 parse_cactus_file('data/cmenov2024.txt') +
                 parse_cactus_file('data/cmedec2024.txt') +
                 parse_cactus_file('data/cmejan2025.txt') +
                 parse_cactus_file('data/cmefeb2025.txt') +
                 parse_cactus_file('data/cmemar2025.txt') +
                 parse_cactus_file('data/cmeapril2025.txt')
                )
    plot_cme(cme_times)

def plot_training_set():    
    data = load_blk_variables('data/Training_data/*.cdf', -1e31)
    cme_times = (parse_cactus_file('data/cmesept2024.txt') + 
                 parse_cactus_file('data/cmeoct2024.txt') + 
                 parse_cactus_file('data/cmenov2024.txt') +
                 parse_cactus_file('data/cmedec2024.txt')
                )
    df, clf, weights = train_cme_prediction_model(data, cme_times)
    save_model(clf)
    plot_weighted_signal(df,threshold = 0.8)
    
    print("Generating report...")
    evaluate_cme_windows(df, cme_times)

def plot_untrained_set():
    from Scripts.evaluate_model_performance import test
    data = load_blk_variables('data/Untrained_data/*.cdf', -1e31)  
    cme_times = (parse_cactus_file('data/cmejan2025.txt') +
                 parse_cactus_file('data/cmefeb2025.txt') +
                 parse_cactus_file('data/cmemar2025.txt') +
                 parse_cactus_file('data/cmeapril2025.txt')
                )
    df = test(data, cme_times)
    plot_weighted_signal(df,threshold = 0.8)

    print("Generating report...")
    evaluate_cme_windows(df, cme_times)