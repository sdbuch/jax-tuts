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
    chunk_len = re.search(r"cl(\d+)", filename)

    # Build the title
    title_parts = []

    if noise_coeff:
        title_parts.append(f"Noise coeff {noise_coeff.group(1)}")

    if seq_len:
        title_parts.append(f"Seq len {seq_len.group(1)}")

    if word_len:
        title_parts.append(f"Word len {word_len.group(1)}")

    if chunk_len:
        title_parts.append(f"Chunk len {chunk_len.group(1)}")

    return ", ".join(title_parts)


def find_matched_runs(directory_path):
    """
    Find matched META and non-META runs in the directory.

    Args:
        directory_path: Path to the directory containing .npy files

    Returns:
        dict: Dictionary of matched runs with non-META prefix as key and both file paths as values
    """
    npy_files = glob(os.path.join(directory_path, "*.npy"))

    # Group files by whether they contain META or not
    meta_files = [f for f in npy_files if "META" in f]
    non_meta_files = [f for f in npy_files if "META" not in f]

    matched_runs = {}

    # For each non-META file, check if there's a corresponding META file
    for non_meta_file in non_meta_files:
        non_meta_basename = os.path.basename(non_meta_file)
        non_meta_prefix = os.path.splitext(non_meta_basename)[0]

        # Look for a META file that matches the prefix structure
        for meta_file in meta_files:
            meta_basename = os.path.basename(meta_file)
            meta_prefix = os.path.splitext(meta_basename)[0]

            # Check if the META file is a match for the non-META file
            # The META file should contain everything in the non-META file plus _META_ and additional params
            if non_meta_prefix in meta_prefix and "_META_" in meta_prefix:
                matched_runs[non_meta_prefix] = {
                    "non_meta": non_meta_file,
                    "meta": meta_file,
                    "meta_prefix": meta_prefix,
                }
                break

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
            "figure.figsize": (10, 6),
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.5,
        }
    )

    # Find matched runs
    matched_runs = find_matched_runs(directory_path)

    if not matched_runs:
        print("No matched META and non-META runs found.")
        return

    # Process each matched pair
    for prefix, files in matched_runs.items():
        # Load the losses arrays
        non_meta_losses = np.load(files["non_meta"])
        meta_losses = np.load(files["meta"])

        # Calculate smoothed losses
        smoothed_non_meta = moving_average(non_meta_losses, smoothing_window)
        smoothed_meta = moving_average(meta_losses, smoothing_window)

        # Use the META run filename for decoding to capture chunk_len
        decoded_title = decode_filename(files["meta_prefix"])

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

        # Plot META data - raw and smoothed
        plt.plot(meta_losses, color="#d62728", alpha=0.15, label=None)
        plt.plot(
            np.arange(len(smoothed_meta)),
            smoothed_meta,
            color="#d62728",
            label="TTT-TF",
        )

        # Add labels and title
        plt.xlabel("Training Steps")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Training Loss Comparison: {decoded_title}")

        # Set y-axis to logarithmic scale for better visualization
        # plt.yscale("log")

        # Add legend
        plt.legend()

        # Add grid for better readability
        plt.grid(True, alpha=0.3)

        # Ensure tight layout
        plt.tight_layout()

        # Save the plot with a new combined filename
        output_path = os.path.join(directory_path, f"{prefix}_comparison.png")
        plt.savefig(output_path, dpi=300)
        print(f"Saved comparison plot to {output_path}")

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
