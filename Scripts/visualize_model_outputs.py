import matplotlib.pyplot as plt
import seaborn as sns

def plot_weighted_signal(df, title="Weighted Signal vs Time", max_points=50000, threshold = 0.75):
    # Limit size to avoid crashes
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42).sort_values('time')

    # Setup plotting style
    sns.set_style("whitegrid")
    plt.figure(figsize=(40, 8))

    # Plot weighted signal
    plt.plot(df['time'], df['weighted_signal'], label='Weighted Signal', color='#007acc', linewidth=0.8)

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(df['weighted_signal'], height=threshold, distance=20)  # increase height and distance

    for p in peaks:
        plt.axvline(df.iloc[p]['time'], color='orange', linestyle='--', linewidth = 0.5)
    
    # vertical lines for actual CME labels
    for idx, t in enumerate(df[df['cme_label'] == 1]['time']):
        plt.axvline(x=t, color='red', linestyle='--', linewidth=0.8, alpha=0.5,
                    label='Actual CME' if idx == 0 else "")

    plt.axhline(y=threshold, color='gray', linestyle='--', label=f'Threshold = {threshold}')

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Weighted Signal")
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()
