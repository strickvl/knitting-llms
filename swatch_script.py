from transformers import AutoModel
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ModelSwatchGenerator:
    def __init__(self, grid_size=10, percentile_thresholds=(25, 50, 75)):
        """
        Initialize the swatch pattern generator.

        Args:
            grid_size (int): Size of the grid (grid_size x grid_size)
            percentile_thresholds (tuple): Three percentile values for creating four color bands
        """
        self.grid_size = grid_size
        self.percentile_thresholds = percentile_thresholds

    def load_model_embeddings(self, model_name):
        """
        Load embeddings from a Hugging Face model.

        Args:
            model_name (str): Name of the model from Hugging Face (e.g., 'prajjwal1/bert-tiny')

        Returns:
            numpy.ndarray: The embedding weights
        """
        print(f"Loading model: {model_name}")
        model = AutoModel.from_pretrained(model_name)

        # Handle different model architectures
        if hasattr(model, "embeddings"):
            # BERT-style models
            embedding_weights = model.embeddings.word_embeddings.weight
        elif hasattr(model, "wte"):
            # GPT-style models
            embedding_weights = model.wte.weight
        else:
            raise ValueError(
                "Couldn't find embedding layer. This model architecture might not be supported."
            )

        return embedding_weights.detach().numpy()

    def generate_pattern(self, model_name):
        """
        Generate a knitting pattern from a model's embeddings.

        Args:
            model_name (str): Name of the model from Hugging Face

        Returns:
            tuple: (color_grid, grid_counts, pca)
                - color_grid: 2D array with values 0, 1, 2, 3 for lightest to darkest blue
                - grid_counts: Raw counts of embeddings in each grid cell
                - pca: Fitted PCA object
        """
        # Load and reduce dimensionality
        embedding_weights = self.load_model_embeddings(model_name)
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embedding_weights)

        # Create grid
        x_min, x_max = reduced_embeddings[:, 0].min(), reduced_embeddings[:, 0].max()
        y_min, y_max = reduced_embeddings[:, 1].min(), reduced_embeddings[:, 1].max()

        x_bins = np.linspace(x_min, x_max, self.grid_size + 1)
        y_bins = np.linspace(y_min, y_max, self.grid_size + 1)

        # Count points in each bin
        grid_counts = np.zeros((self.grid_size, self.grid_size))
        for x, y in reduced_embeddings:
            x_idx = np.digitize(x, x_bins) - 1
            y_idx = np.digitize(y, y_bins) - 1
            if x_idx == self.grid_size:
                x_idx -= 1
            if y_idx == self.grid_size:
                y_idx -= 1
            grid_counts[y_idx, x_idx] += 1

        # Create color coding with 4 shades
        low_threshold = np.percentile(grid_counts, self.percentile_thresholds[0])
        mid_threshold = np.percentile(grid_counts, self.percentile_thresholds[1])
        high_threshold = np.percentile(grid_counts, self.percentile_thresholds[2])

        color_grid = np.zeros_like(grid_counts)
        color_grid[grid_counts > low_threshold] = 1
        color_grid[grid_counts > mid_threshold] = 2
        color_grid[grid_counts > high_threshold] = 3

        return color_grid, grid_counts, pca

    def plot_pattern(self, color_grid, model_name, save_path=None):
        """
        Visualize the knitting pattern.

        Args:
            color_grid (numpy.ndarray): Grid of color values (0, 1, 2, 3)
            model_name (str): Name of the model (for title)
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 10))

        # Custom colormap for four shades of blue
        colors = ["#E3F2FD", "#90CAF9", "#2196F3", "#0D47A1"]
        cmap = plt.cm.colors.ListedColormap(colors)

        # Plot the pattern
        sns.heatmap(
            color_grid,
            cmap=cmap,
            cbar=False,
            square=True,
            xticklabels=False,
            yticklabels=False,
        )

        # Add title and labels
        plt.title(f"Knitting Pattern for {model_name}\n(Lightest=0 to Darkest=3)")

        if save_path:
            # Create outputs directory if it doesn't exist
            os.makedirs("outputs", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Saved pattern to {save_path}")
        else:
            plt.show()

        plt.close()

    def print_pattern_stats(self, color_grid):
        """Print statistics about the pattern."""
        print("\nPattern Statistics:")
        print(f"Lightest blue (0):  {np.sum(color_grid == 0)} cells")
        print(f"Light blue (1):     {np.sum(color_grid == 1)} cells")
        print(f"Medium blue (2):    {np.sum(color_grid == 2)} cells")
        print(f"Dark blue (3):      {np.sum(color_grid == 3)} cells")


def main():
    # List of models to compare
    models = [
        "prajjwal1/bert-tiny",  # Very small BERT model
        "gpt2",  # Small GPT-2 model
    ]

    generator = ModelSwatchGenerator(grid_size=10)

    # Generate patterns for each model
    for model_name in models:
        print(f"\nProcessing {model_name}...")
        color_grid, counts, pca = generator.generate_pattern(model_name)

        # Print statistics
        generator.print_pattern_stats(color_grid)

        # Plot and save
        save_path = os.path.join(
            "outputs", f"knitting_pattern_{model_name.replace('/', '_')}.png"
        )
        generator.plot_pattern(color_grid, model_name, save_path)

        # Print first few rows as knitting instructions
        print("\nFirst few rows of knitting instructions:")
        for i in range(min(3, len(color_grid))):
            row = color_grid[i]
            instruction = " ".join(
                [
                    "Lightest"
                    if x == 0
                    else "Light"
                    if x == 1
                    else "Medium"
                    if x == 2
                    else "Dark"
                    for x in row
                ]
            )
            print(f"Row {i+1}: {instruction}")


if __name__ == "__main__":
    main()
