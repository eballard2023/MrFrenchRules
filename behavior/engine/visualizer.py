import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json
import time

class WinstonMonitor:
    def __init__(self, threshold=1.5):
        self.labels = ['V', 'A', 'D', 'O', 'C', 'E', 'A', 'N']
        self.threshold = threshold
        
        plt.ion()
        # 2 graph
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.5, bottom=0.18)

    def update(self):
        try:
            with open('data/current_state.json', 'r') as f:
                data = json.load(f)
            
            # --- GRAPH 1 : Z-SCORES (Instant) ---
            self.ax1.clear()
            z_scores = data['z_scores']
            colors_z = ['#e74c3c' if abs(z) >= self.threshold else '#3498db' for z in z_scores]
            self.ax1.bar(self.labels, z_scores, color=colors_z)
            self.ax1.axhline(y=self.threshold, color='r', linestyle='--', alpha=0.3)
            self.ax1.axhline(y=-self.threshold, color='r', linestyle='--', alpha=0.3)
            self.ax1.set_ylim(-4, 4)
            self.ax1.set_title(f"INSTANT DRIFT (Z-SCORES) - Status: {data['status']}")

            # Legend for bar colors
            legend_elements = [
                Patch(facecolor='#3498db', label='|z| < threshold (within baseline)'),
                Patch(facecolor='#e74c3c', label='|z| ≥ threshold (strong drift)'),
            ]
            self.ax1.legend(handles=legend_elements, loc='upper right', fontsize='x-small')


            # --- GRAPH 2 : CUSUM (Accumulation) ---
            self.ax2.clear()
            # combination of positive and negative for the view (positive in blue, negative in orange for example)
            c_pos = data['cusum_pos']
            c_neg = [-x for x in data['cusum_neg']] # symetry for the view
            
            self.ax2.bar(self.labels, c_pos, color='#2ecc71', alpha=0.7, label='Positive Accumulation')
            self.ax2.bar(self.labels, c_neg, color='#f39c12', alpha=0.7, label='Negative Accumulation')
            
            # Threshold CUSUM
            self.ax2.axhline(y=self.threshold, color='red', linewidth=2, label='TRIGGER LINE')
            self.ax2.axhline(y=-self.threshold, color='red', linewidth=2)
            
            self.ax2.set_ylim(-self.threshold * 1.5, self.threshold * 1.5)
            self.ax2.set_title("CUMULATIVE SUM (CUSUM) - Energy towards Trigger")
            self.ax2.legend(loc='upper right', fontsize='x-small')

            plt.pause(0.1)
        except Exception:
            pass


def run_viz():
    monitor = WinstonMonitor()
    while True:
        monitor.update()
        time.sleep(0.5)