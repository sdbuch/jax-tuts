import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def moving_average(data, window_size):
    """
    Calculate the moving average of the data with the specified window size.

    Args:
        data: NumPy array of data
        window_size: Size of the moving average window

    Returns:
        NumPy array of smoothed data
    """
    if window_size <= 1:
        return data

    # Create a valid convolution (output length = input length - window_size + 1)
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(data, weights, mode="valid")

    # Pad the beginning to make output same length as input
    padding = np.full(window_size - 1, np.nan)
    return np.concatenate([padding, smoothed])


def extract_meta_param(filename):
    """
    Extract a unique parameter from a META filename for labeling.

    Args:
        filename: The META filename to analyze

    Returns:
        str: A string representing the unique meta parameter (chunk length or other)
    """
    # Extract the chunk length as the primary identifier
    chunk_len = re.search(r"cl(\d+)", filename)
    if chunk_len:
        return f"cl={chunk_len.group(1)}"

    # Fall back to stride length if chunk length isn't available
    stride_len = re.search(r"stl(\d+)", filename)
    if stride_len:
        return f"stl={stride_len.group(1)}"

    # Fall back to inner learning rate
    ilr = re.search(r"ilr(\d+\.\d+e[-+]?\d+)", filename)
    if ilr:
        return f"ilr={ilr.group(1)}"

    # If no specific parameter found, just use 'META'
    return "META"


def decode_filename(filename):
    """
    Decode the filename to extract key parameters for the plot title.

    Args:
        filename: The filename to decode

    Returns:
        str: A human-readable title based on key parameters
    """
    # Extract the key parameters we want to display
    noise_coeff = re.search(r"nc(\d+\.\d+)", filename)
    seq_len = re.search(r"sl(\d+)", filename)
    word_len = re.search(r"wl(\d+)", filename)

    # Build the title
    title_parts = []

    if noise_coeff:
        title_parts.append(f"Noise coeff {noise_coeff.group(1)}")

    if seq_len:
        title_parts.append(f"Seq len {seq_len.group(1)}")

    if word_len:
        title_parts.append(f"Word len {word_len.group(1)}")

    return ", ".join(title_parts)


def find_matched_runs(directory_path):
    """
    Find matched META and non-META runs in the directory.

    Args:
        directory_path: Path to the directory containing .npy files

    Returns:
        dict: Dictionary of matched runs with non-META prefix as key and file info as values
    """
    npy_files = glob(os.path.join(directory_path, "*.npy"))

    # Group files by whether they contain META or not
    meta_files = [f for f in npy_files if "META" in f]
    non_meta_files = [f for f in npy_files if "META" not in f]

    matched_runs = {}

    # For each non-META file, find all corresponding META files
    for non_meta_file in non_meta_files:
        non_meta_basename = os.path.basename(non_meta_file)
        non_meta_prefix = os.path.splitext(non_meta_basename)[0]

        # Look for all META files that match the prefix structure
        matching_meta_files = []

        for meta_file in meta_files:
            meta_basename = os.path.basename(meta_file)
            meta_prefix = os.path.splitext(meta_basename)[0]

            # Check if the META file is a match for the non-META file
            if non_meta_prefix in meta_prefix and "_META_" in meta_prefix:
                # Extract a parameter to differentiate this META variant
                meta_param = extract_meta_param(meta_prefix)

                matching_meta_files.append(
                    {"file": meta_file, "prefix": meta_prefix, "param": meta_param}
                )

        # Only add to matched runs if we found at least one META match
        if matching_meta_files:
            matched_runs[non_meta_prefix] = {
                "non_meta": non_meta_file,
                "meta_runs": matching_meta_files,
            }

    return matched_runs


def plot_matched_runs(directory_path, smoothing_window=50):
    """
    Find matched META and non-META runs, plot them together, and save as .png files.

    Args:
        directory_path: Path to the directory containing .npy files
        smoothing_window: Window size for moving average smoothing (default: 50)
    """
    # Set up nice plot style
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),  # Slightly larger figure for more META runs
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.5,
        }
    )

    # Library of visually distinct colors for META runs
    # Using a colorblind-friendly palette
    meta_colors = [
        "#d62728",  # red
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
    ]

    # Find matched runs
    matched_runs = find_matched_runs(directory_path)

    if not matched_runs:
        print("No matched META and non-META runs found.")
        return

    # Process each matched set
    for prefix, files in matched_runs.items():
        non_meta_file = files["non_meta"]
        meta_runs = files["meta_runs"]

        # Load the non-META losses array
        non_meta_losses = np.load(non_meta_file)
        smoothed_non_meta = moving_average(non_meta_losses, smoothing_window)

        # Create the plot
        plt.figure()

        # Plot non-META data - raw and smoothed
        plt.plot(non_meta_losses, color="#1f77b4", alpha=0.15, label=None)
        plt.plot(
            np.arange(len(smoothed_non_meta)),
            smoothed_non_meta,
            color="#1f77b4",
            label="TF",
        )

        # Plot each META run with a different color
        for i, meta_run in enumerate(meta_runs):
            meta_file = meta_run["file"]
            meta_prefix = meta_run["prefix"]
            meta_param = meta_run["param"]

            # Get the color for this META run (cycle through the colors)
            color = meta_colors[i % len(meta_colors)]

            # Load and smooth the META losses
            meta_losses = np.load(meta_file)
            smoothed_meta = moving_average(meta_losses, smoothing_window)

            # Plot raw and smoothed META data
            plt.plot(meta_losses, color=color, alpha=0.15, label=None)
            plt.plot(
                np.arange(len(smoothed_meta)),
                smoothed_meta,
                color=color,
                label=f"TTT-TF ({meta_param})",
            )

        # Use the non-META filename for basic parameters
        decoded_title = decode_filename(prefix)

        # Add labels and title
        plt.xlabel("Training Steps")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Training Loss Comparison: {decoded_title}")

        # Set y-axis to logarithmic scale for better visualization
        plt.yscale("log")

        # Add legend with better placement for multiple entries
        plt.legend(loc="best", framealpha=0.9)

        # Add grid for better readability
        plt.grid(True, alpha=0.3)

        # Ensure tight layout
        plt.tight_layout()

        # Save the plot with a new combined filename
        output_path = os.path.join(directory_path, f"{prefix}_multi_comparison.png")
        plt.savefig(output_path, dpi=300)
        print(f"Saved multi-comparison plot to {output_path}")

        # Close the figure to free memory
        plt.close()


if __name__ == "__main__":
    import argparse

    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Plot matched META and non-META loss curves"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing .npy files (default: current directory)",
    )
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        default=50,
        help="Smoothing window size (default: 50)",
    )

    args = parser.parse_args()

    plot_matched_runs(args.directory, args.window)
    print("Done!")
