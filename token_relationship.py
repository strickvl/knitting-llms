import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os


class TokenRelationshipVisualizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Words with strong semantic relationships and opposites
        self.words = [
            "sun",
            "moon",
            "star",  # celestial
            "fire",
            "ice",
            "water",  # elements
            "cat",
            "dog",
            "bird",  # animals
            "good",
            "evil",
            "neutral",  # concepts
            "love",
            "hate",
            "peace",  # emotions
        ]

    def get_layer_embeddings(self, layer_idx=-1):
        """Get embeddings from a specific transformer layer."""
        inputs = self.tokenizer(self.words, return_tensors="pt", padding=True)

        with torch.no_grad():
            # Get all hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Extract embeddings from specified layer
            layer_embeddings = hidden_states[layer_idx]
            # Take first token for each word
            word_embeddings = np.array([emb[0].numpy() for emb in layer_embeddings])

            # Normalize embeddings
            scaler = StandardScaler()
            word_embeddings = scaler.fit_transform(word_embeddings)

            return word_embeddings

    def create_pattern(self):
        """Create patterns using embeddings from different layers."""
        # Get embeddings from first, middle, and last layers
        num_layers = (
            len(self.model.encoder.layer)
            if hasattr(self.model, "encoder")
            else len(self.model.h)
            if hasattr(self.model, "h")
            else 12
        )

        layer_indices = [0, num_layers // 2, -1]
        patterns = {}
        matrices = {}

        for layer_idx in layer_indices:
            embeddings = self.get_layer_embeddings(layer_idx)
            sim_matrix = cosine_similarity(embeddings)

            # Normalize similarity matrix to enhance contrast
            sim_matrix = (sim_matrix - sim_matrix.min()) / (
                sim_matrix.max() - sim_matrix.min()
            )

            # Create three-level pattern using dynamic thresholds
            pattern = np.zeros_like(sim_matrix)
            thresholds = np.percentile(sim_matrix, [33, 66])
            pattern[sim_matrix > thresholds[0]] = 1
            pattern[sim_matrix > thresholds[1]] = 2

            layer_name = f"layer_{layer_idx}"
            patterns[layer_name] = pattern
            matrices[layer_name] = sim_matrix

        return patterns, matrices

    def visualize(self, save_path=None):
        patterns, matrices = self.create_pattern()

        # Create figure with two rows: similarities and patterns
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Token Patterns for {self.model_name}", fontsize=16)

        # Colors for knitting pattern
        colors = ["#F8F9FA", "#ADB5BD", "#495057"]  # Light, medium, dark grays
        knit_cmap = plt.cm.colors.ListedColormap(colors)

        for idx, (layer_name, (pattern, matrix)) in enumerate(
            zip(patterns.keys(), zip(patterns.values(), matrices.values()))
        ):
            # Plot similarity matrix
            sns.heatmap(
                matrix,
                ax=axes[0, idx],
                cmap="viridis",
                xticklabels=self.words,
                yticklabels=self.words,
            )
            axes[0, idx].set_title(f"Similarities - {layer_name}")
            axes[0, idx].set_xticklabels(self.words, rotation=45, ha="right")

            # Plot knitting pattern
            sns.heatmap(
                pattern,
                ax=axes[1, idx],
                cmap=knit_cmap,
                xticklabels=self.words,
                yticklabels=self.words,
            )
            axes[1, idx].set_title(f"Knitting Pattern - {layer_name}")
            axes[1, idx].set_xticklabels(self.words, rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            os.makedirs("outputs", exist_ok=True)
            plt.savefig(
                os.path.join("outputs", save_path), bbox_inches="tight", dpi=300
            )
            print(f"Saved visualization to {os.path.join('outputs', save_path)}")
        else:
            plt.show()

        plt.close()

        # Print knitting instructions for the final layer pattern
        final_pattern = patterns[list(patterns.keys())[-1]]
        self._print_knitting_instructions(final_pattern)

    def _print_knitting_instructions(self, pattern):
        """Print detailed knitting instructions for the pattern."""
        print("\nKnitting Instructions:")
        print("\nStitch Guide:")
        print("  Light (0): Knit stitch in main color")
        print("  Medium (1): Purl stitch in main color or knit in contrast color")
        print("  Dark (2): Seed stitch or knit in second contrast color")

        # Group words into their semantic categories
        word_groups = [self.words[i : i + 3] for i in range(0, len(self.words), 3)]

        for group_idx, group in enumerate(word_groups):
            print(f"\nSection {group_idx + 1} - {group}:")
            for i, word1 in enumerate(group):
                row_idx = group_idx * 3 + i
                row_pattern = pattern[row_idx]

                print(f"\nRow {row_idx + 1} ({word1}):")
                stitches = []
                current_stitch = row_pattern[0]
                count = 1

                # Generate compressed instructions (e.g., "3 light, 2 dark")
                for stitch in row_pattern[1:]:
                    if stitch == current_stitch:
                        count += 1
                    else:
                        stitch_name = (
                            "light"
                            if current_stitch == 0
                            else "medium"
                            if current_stitch == 1
                            else "dark"
                        )
                        stitches.append(f"{count} {stitch_name}")
                        current_stitch = stitch
                        count = 1

                # Add the last group
                stitch_name = (
                    "light"
                    if current_stitch == 0
                    else "medium"
                    if current_stitch == 1
                    else "dark"
                )
                stitches.append(f"{count} {stitch_name}")

                print("  " + ", ".join(stitches))


def main():
    models = ["bert-base-uncased", "gpt2", "distilroberta-base"]

    for model_name in models:
        print(f"\nProcessing {model_name}...")
        visualizer = TokenRelationshipVisualizer(model_name)
        visualizer.visualize(f"token_patterns_{model_name.replace('/', '_')}.png")


if __name__ == "__main__":
    main()
