"""
Visualize LLM Judge results as stacked bar charts.

This script reads llm_judge_results files and creates visualizations
showing the relevance distribution across 100 queries.
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def extract_timestamp_from_filename(filename: str) -> str:
    """
    Extract timestamp from llm_judge_results filename.

    Args:
        filename: Filename like "llm_judge_results_20251110_054017.txt"

    Returns:
        Timestamp string like "20251110_054017"
    """
    match = re.search(r'llm_judge_results_(\d{8}_\d{6})\.txt', filename)
    if match:
        return match.group(1)
    return "unknown"


def parse_llm_judge_results(file_path: Path) -> tuple[list[int], list[int], list[int]]:
    """
    Parse llm_judge_results file and extract counts.

    Args:
        file_path: Path to the llm_judge_results file

    Returns:
        Tuple of three lists: (relevant_counts, somewhat_relevant_counts, irrelevant_counts)
    """
    relevant_counts = []
    somewhat_relevant_counts = []
    irrelevant_counts = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            parts = line.split()
            if len(parts) == 3:
                # New format: relevant somewhat_relevant irrelevant
                relevant = int(parts[0])
                somewhat_relevant = int(parts[1])
                irrelevant = int(parts[2])
            elif len(parts) == 2:
                # Old format: relevant irrelevant (no "somewhat relevant" category)
                # Treat as relevant and irrelevant only
                relevant = int(parts[0])
                somewhat_relevant = 0
                irrelevant = int(parts[1])
            else:
                print(f"Warning: Line {line_num} has unexpected format: {line}")
                continue

            relevant_counts.append(relevant)
            somewhat_relevant_counts.append(somewhat_relevant)
            irrelevant_counts.append(irrelevant)

    return relevant_counts, somewhat_relevant_counts, irrelevant_counts


def plot_llm_judge_results(file_path: Path):
    """
    Read llm_judge_results file and create a stacked bar chart visualization.

    The chart shows 100 stacked bars with:
    - Dark green for relevant results
    - Light green for somewhat relevant results
    - Red for irrelevant results

    The chart is saved in the same folder as the input file with the same timestamp.

    Args:
        file_path: Path to the llm_judge_results file
    """
    # Parse the results
    relevant, somewhat_relevant, irrelevant = parse_llm_judge_results(file_path)

    if len(relevant) == 0:
        print(f"Error: No data found in {file_path}")
        return

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 8))

    # Query indices (1-100 or however many we have)
    query_indices = np.arange(1, len(relevant) + 1)

    # Define colors
    color_relevant = '#1a5f2a'        # Dark green
    color_somewhat = '#90ee90'         # Light green
    color_irrelevant = '#dc143c'       # Red

    # Create stacked bars
    bar_width = 0.8

    # Bottom bars: irrelevant (red)
    bars1 = ax.bar(query_indices, irrelevant, bar_width,
                   label='Irrelevant', color=color_irrelevant)

    # Middle bars: somewhat relevant (light green) - stacked on top of irrelevant
    bars2 = ax.bar(query_indices, somewhat_relevant, bar_width,
                   bottom=irrelevant, label='Somewhat Relevant',
                   color=color_somewhat)

    # Top bars: relevant (dark green) - stacked on top of somewhat relevant
    bottom_for_relevant = np.array(irrelevant) + np.array(somewhat_relevant)
    bars3 = ax.bar(query_indices, relevant, bar_width,
                   bottom=bottom_for_relevant, label='Relevant',
                   color=color_relevant)

    # Customize the chart
    ax.set_xlabel('Query Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Results', fontsize=12, fontweight='bold')
    ax.set_title('LLM Judge Results: Relevance Distribution Across 100 Queries',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 10)
    ax.set_xlim(0, len(relevant) + 1)

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Extract timestamp from filename and create output path
    timestamp = extract_timestamp_from_filename(file_path.name)
    output_file = file_path.parent / f"llm_judge_visualization_{timestamp}.png"

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")

    # Close the figure to free memory
    plt.close(fig)


if __name__ == "__main__":
    # Example usage
    results_file = "../../../data/q100_output/run_20251110_072451/llm_judge_results_20251110_072859.txt"
    results_file = Path(results_file)
    
    if results_file.exists():
        plot_llm_judge_results(results_file)
    else:
        print(f"Error: File not found: {results_file}")
        print("Please update the path in the script.")
