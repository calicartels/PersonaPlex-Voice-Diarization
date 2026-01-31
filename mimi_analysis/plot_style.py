import matplotlib.pyplot as plt
import seaborn as sns

def setup_style():
    """Configure matplotlib and seaborn for professional plots"""
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14

COLOR_PALETTE = sns.color_palette("deep")
HEATMAP_CMAP = "viridis"
SIMILARITY_CMAP = "RdYlGn"

