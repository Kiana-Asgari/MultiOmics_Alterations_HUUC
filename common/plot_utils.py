import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import numpy as np

def set_seaborn_style(style: str = "white", 
                      context: str = "paper",
                      font_scale: float = 1.2,
                      use_latex: bool = True,
                      figsize: tuple = (8, 6)):
   
    if use_latex:
        try:
            plt.rcParams.update({
                'text.usetex': True,
                'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}',
                'font.family': 'serif',
                'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
                'font.sans-serif': ['Computer Modern Sans', 'Arial', 'DejaVu Sans'],
                'font.monospace': ['Computer Modern Typewriter', 'Courier', 'DejaVu Sans Mono'],
                'mathtext.fontset': 'cm',
                'mathtext.rm': 'serif',
                'mathtext.cal': 'serif',
                'mathtext.it': 'serif:italic',
                'mathtext.bf': 'serif:bold',
            })
        except Exception as e:
            print(f"Warning: LaTeX rendering not available: {e}")
            print("Falling back to standard font rendering")
            use_latex = False
    
    # Set seaborn style and context
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    
    # Configure matplotlib parameters for publication quality
    plt.rcParams.update({
        # Figure settings
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        'figure.autolayout': True,
        
        # Axes settings
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'axes.grid': False,
        
        # Tick settings
        'xtick.major.size': 6,
        'xtick.major.width': 1.5,
        'xtick.minor.size': 3,
        'xtick.minor.width': 1.0,
        'ytick.major.size': 6,
        'ytick.major.width': 1.5,
        'ytick.minor.size': 3,
        'ytick.minor.width': 1.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Font settings
        'font.size': 6,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'figure.titlesize': 8,
        
        # Line settings
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'lines.markeredgewidth': 0.5,
        
        # Legend settings
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.borderpad': 0.5,
        'legend.columnspacing': 1.0,
        
        # Grid settings
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
    
    })
    
    # Set a professional color palette
    sns.set_palette("husl", n_colors=10)
    


def apply_spine_styling(ax):
    """
    Apply professional spine styling to a matplotlib axes object.
    This function handles spine styling that might not be available via rcParams.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to style
    """
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Style remaining spines
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Set spine colors
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')


def get_publication_colors(n_colors: int = 10, palette_type: str = "qualitative"):
    if palette_type == "qualitative":
        # Colorblind-friendly qualitative palette
        face_colors = ['salmon', 'lightskyblue', 'grey'] 
        edge_colors = ['red', 'blue', 'grey']
        return face_colors, edge_colors
    elif palette_type == "diverging":
        # Red-Blue diverging palette
        colors = ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf',
                 '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
    elif palette_type == "sequential":
        # Blue sequential palette
        colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
                 '#4292c6', '#2171b5', '#08519c', '#08306b']
    else:
        # Default to seaborn husl
        colors = sns.color_palette("husl", n_colors)
        return colors
    
    # Return the requested number of colors, cycling if needed
    if n_colors <= len(colors):
        return colors[:n_colors]
    else:
        # Cycle through colors if more are needed
        return [colors[i % len(colors)] for i in range(n_colors)]


def create_publication_legend(ax, loc: str = 'lower right', 
                             bbox_to_anchor: tuple = None,
                             ncol: int = 1,
                             frameon: bool = True,
                             fancybox: bool = True,
                             shadow: bool = False,
                             fontsize: int = 10):
    legend = ax.legend(
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=frameon,
        fancybox=fancybox,
        shadow=shadow,
        fontsize=fontsize,
        borderpad=0.5,
        columnspacing=1.0,
        labelspacing=0.5
    )
    
    # Style the legend frame
    if frameon:
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_linewidth(1.0)
        frame.set_alpha(0.5)  # Completely transparent background
    
    return legend


def save_publication_figure(fig, filename: str, 
                           formats: list = ['pdf', 'png'],
                           transparent: bool = False,
                           bbox_inches: str = 'tight',
                           pad_inches: float = 0.1):
    
    for fmt in formats:
        if fmt.lower() == 'pdf':
            # PDF is vector format, no DPI needed
            fig.savefig(f"{filename}.pdf", 
                       format='pdf',
                       bbox_inches=bbox_inches,
                       pad_inches=pad_inches,
                       transparent=transparent,
                       facecolor='white' if not transparent else 'none')
        elif fmt.lower() in ['png', 'jpg', 'jpeg', 'tiff']:
            fig.savefig(f"{filename}.{fmt}", 
                       format=fmt,
                       bbox_inches=bbox_inches,
                       pad_inches=pad_inches,
                       transparent=transparent,
                       facecolor='white' if not transparent else 'none')
        else:
            print(f"Warning: Unsupported format '{fmt}', skipping...")
    
    print(f"Figure saved as: {', '.join([f'{filename}.{fmt}' for fmt in formats])}")


def reset_matplotlib_defaults():
    """
    Reset matplotlib to default settings.
    Useful for switching between different styling contexts.
    """
    plt.rcdefaults()
    print("Matplotlib settings reset to defaults")


def annotate_significant_features(ax: plt.Axes, 
                                plot_df: pd.DataFrame, 
                                significant_mask: pd.Series,
                                fontsize: int = 24,
                                bbox_facecolor: str = 'white',
                                bbox_alpha: float = 0.5,
                                arrow_alpha: float = 0.8,
                                max_chars_per_line: int = 22) -> list:
    """Automatically annotate significant features with smart positioning and square boxes."""

    significant_features = plot_df[significant_mask]
    texts = []
    for _, row in significant_features.iterrows():
        # Wrap text to fit in square boxes
        feature_name = row['feature']
        if 'c0' in feature_name.lower():
            border_color = 'blue'
            bbox_facecolor = 'blue'
        elif row['log2_fc'] > 0:
            border_color = 'red'
            bbox_facecolor = 'white'
        else:
            border_color = 'blue'
            bbox_facecolor = 'white'

        wrapped_text = wrap_text_to_fit(feature_name, max_chars_per_line)
        text = ax.annotate(
            wrapped_text,
            xy=(row['log2_fc'], row['neg_log10_p']),
            xytext=(row['log2_fc'] + 0.2* np.sign(row['log2_fc']),
                    row['neg_log10_p'] - 0.2*np.sign(row['neg_log10_p'])),  # Box starts here
            fontsize=fontsize,
            bbox=dict(
                boxstyle='round,pad=0.1',
                facecolor=bbox_facecolor,
                edgecolor=border_color,
                alpha=bbox_alpha,
                linewidth=1
            ),
            arrowprops=dict(
                arrowstyle='->',
                color=border_color,
                alpha=arrow_alpha,
                connectionstyle='arc3,rad=0.1'
            ),
            ha='center',
            va='center'
        )
        texts.append(text)
    
    # Auto-adjust text positions to avoid overlaps
    adjust_text(texts, ax=ax, force_points=(1.5, 1.5), force_text=(3, 3), expand_points=(7, 7), expand_text=(7, 7))
    
    return texts


def wrap_text_to_fit(text: str, max_chars_per_line: int) -> str:
    """Wrap text to fit within a specified number of characters per line."""
    if len(text) <= max_chars_per_line:
        return text
    
    import re
    # Replace '-' and '_' with space
    clean_text = text.replace('-', ' ').replace('_', ' ')
    n = len(clean_text)
    if n <= max_chars_per_line:
        lines = [clean_text]
    else:
        # Split from the middle regardless
        split_idx = n // 2
        # Find nearest space to the middle to split, prefer after mid
        if clean_text[split_idx] != ' ':
            lines = [clean_text[:split_idx]+'-', clean_text[split_idx+1:]]
        else:
            lines = [clean_text[:split_idx], clean_text[split_idx+1:]]

    
    return '\n'.join(lines)