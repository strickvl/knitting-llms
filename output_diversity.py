import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
import warnings


@dataclass
class PatternConfig:
    size: int = 20  # Increased from 10 to 20
    num_tokens: int = 10
    color_scheme: List[str] = None
    model_colors: dict = field(
        default_factory=lambda: {
            "bert_base_uncased": ["#E3F2FD", "#64B5F6", "#1565C0"],  # Blues
            "gpt2_xl": ["#E8F5E9", "#81C784", "#2E7D32"],  # Greens
            "t5_11b": ["#F3E5F5", "#BA68C8", "#7B1FA2"],  # Purples
            "gpt_j_6B": ["#FFF3E0", "#FFB74D", "#EF6C00"],  # Oranges
            "bloom_7b1": ["#FCE4EC", "#F06292", "#C2185B"],  # Pinks
            "Llama_2_13b_hf": ["#FFF3E0", "#FFB74D", "#F57C00"],  # Deep Oranges
            "falcon_7b": ["#F1F8E9", "#AED581", "#689F38"],  # Light Greens
            "default": ["#E9ECEF", "#6C757D", "#212529"],  # Grayscale
        }
    )

    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = self.model_colors["default"]


class LMPatternGenerator:
    def __init__(self, model_name: str, prompt: str = "Nothing will come of nothing"):
        """Initialize the pattern generator with a specific model and prompt."""
        self.model_name = model_name
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # Try different model types based on architecture
            try:
                if "t5" in model_name.lower():
                    from transformers import T5ForConditionalGeneration

                    self.model = T5ForConditionalGeneration.from_pretrained(
                        model_name, output_attentions=True, output_hidden_states=True
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, output_attentions=True, output_hidden_states=True
                    )
            except ValueError:
                try:
                    from transformers import AutoModel

                    self.model = AutoModel.from_pretrained(
                        model_name, output_attentions=True, output_hidden_states=True
                    )
                except Exception as e:
                    raise ValueError(f"Could not load model {model_name}: {str(e)}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt = prompt
        self.config = PatternConfig()

        # Set color scheme based on model
        base_model_name = model_name.split("/")[-1].replace(
            "-", "_"
        )  # Handle Hugging Face paths
        if base_model_name in self.config.model_colors:
            self.config.color_scheme = self.config.model_colors[base_model_name]

    def get_model_outputs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Get token probabilities, attention patterns, and hidden states from the model."""
        inputs = self.tokenizer(self.prompt, return_tensors="pt")

        with torch.no_grad():
            # Handle T5 models differently
            if "t5" in self.model_name.lower():
                # For T5, we need to provide decoder input ids
                decoder_input_ids = self.tokenizer("", return_tensors="pt")["input_ids"]
                outputs = self.model(
                    **inputs,
                    decoder_input_ids=decoder_input_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                )
                # Use encoder outputs for pattern generation
                attention = (
                    outputs.encoder_attentions[-1][0]
                    if outputs.encoder_attentions
                    else torch.eye(inputs["input_ids"].shape[1]).unsqueeze(0)
                )
                hidden_states = (
                    outputs.encoder_hidden_states
                    if outputs.encoder_hidden_states
                    else [outputs.encoder_last_hidden_state]
                )
                # Create a simplified probability distribution from encoder hidden states
                last_hidden = outputs.encoder_last_hidden_state[0]
                probs = torch.softmax(last_hidden @ last_hidden.T, dim=-1)
            else:
                # Standard handling for other models
                outputs = self.model(
                    **inputs, output_attentions=True, output_hidden_states=True
                )

                # Handle different model architectures
                if hasattr(outputs, "logits"):
                    logits = outputs.logits[0]  # [seq_len, vocab_size]
                    probs = torch.softmax(logits, dim=-1)
                else:
                    # For models without logits, use last hidden state
                    last_hidden = outputs.last_hidden_state[0]  # [seq_len, hidden_size]
                    # Create a simplified probability distribution from hidden states
                    probs = torch.softmax(last_hidden @ last_hidden.T, dim=-1)

                # Get attention patterns from last layer if available
                if hasattr(outputs, "attentions") and outputs.attentions:
                    attention = outputs.attentions[-1][0]
                else:
                    # Create dummy attention pattern if not available
                    seq_len = inputs["input_ids"].shape[1]
                    attention = torch.eye(seq_len).unsqueeze(0)  # [1, seq_len, seq_len]

                hidden_states = (
                    outputs.hidden_states
                    if hasattr(outputs, "hidden_states")
                    else [outputs.last_hidden_state]
                )

        return probs, attention, hidden_states

    def create_token_pattern(self, probs: torch.Tensor, size: int) -> np.ndarray:
        """Create a pattern based on token probabilities."""
        # For T5 and other models that might have smaller vocab/probability distributions
        num_tokens = min(self.config.num_tokens, probs.shape[-1])

        top_probs, _ = torch.topk(probs, num_tokens, dim=-1)
        mean_probs = top_probs.mean(-1)  # [seq_len]
        pattern = (
            mean_probs.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )  # [1, 1, seq_len, 1]
        pattern = torch.nn.functional.interpolate(
            pattern, size=(size, size), mode="bilinear", align_corners=False
        )
        return pattern.squeeze().numpy()

    def create_attention_pattern(
        self, attention: torch.Tensor, size: int
    ) -> np.ndarray:
        """Create a pattern based on attention weights."""
        avg_attention = attention.mean(0)
        pattern = (
            torch.nn.functional.interpolate(
                avg_attention.unsqueeze(0).unsqueeze(0),
                size=(size, size),
                mode="bilinear",
            )
            .squeeze()
            .numpy()
        )
        return pattern

    def create_layer_interaction_pattern(
        self, hidden_states: List[torch.Tensor], size: int
    ) -> np.ndarray:
        """Create a pattern based on layer interactions."""
        layer_indices = [1, len(hidden_states) // 2, -1]
        selected_states = [hidden_states[i][0].mean(1) for i in layer_indices]
        interactions = torch.corrcoef(torch.stack(selected_states))
        pattern = (
            torch.nn.functional.interpolate(
                interactions.unsqueeze(0).unsqueeze(0),
                size=(size, size),
                mode="bilinear",
            )
            .squeeze()
            .numpy()
        )
        return pattern

    def combine_patterns(
        self, patterns: List[np.ndarray], weights: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine multiple patterns into a final knitting pattern."""
        if weights is None:
            weights = [1.0 / len(patterns)] * len(patterns)

        normalized_patterns = []
        for pattern in patterns:
            norm_pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
            normalized_patterns.append(norm_pattern)

        combined = sum(w * p for w, p in zip(weights, normalized_patterns))
        combined = (combined - combined.min()) / (combined.max() - combined.min())

        knit_pattern = np.zeros_like(combined)
        knit_pattern[combined > np.percentile(combined, 33)] = 1
        knit_pattern[combined > np.percentile(combined, 66)] = 2

        return knit_pattern, combined

    def generate_pattern(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the final knitting pattern."""
        probs, attention, hidden_states = self.get_model_outputs()

        token_pattern = self.create_token_pattern(probs, self.config.size)
        attention_pattern = self.create_attention_pattern(attention, self.config.size)
        layer_pattern = self.create_layer_interaction_pattern(
            hidden_states, self.config.size
        )

        patterns = [token_pattern, attention_pattern, layer_pattern]
        weights = [0.4, 0.3, 0.3]  # Adjustable weights

        return self.combine_patterns(patterns, weights)

    def calculate_pattern_complexity(self, pattern: np.ndarray) -> str:
        """Calculate pattern complexity based on color changes and distribution.

        Returns:
            str: Difficulty rating (Beginner, Intermediate, or Advanced)
        """
        # Count color changes within rows
        color_changes = 0
        for row in pattern:
            changes = np.sum(np.diff(row) != 0)
            color_changes += changes

        avg_changes_per_row = color_changes / len(pattern)

        if avg_changes_per_row < 3:
            return "Beginner"
        elif avg_changes_per_row < 6:
            return "Intermediate"
        else:
            return "Advanced"

    def get_color_name(self, hex_color: str) -> str:
        """Get a human-readable color name for a hex color code."""
        # Basic color mapping for common colors
        color_names = {
            "#E3F2FD": "Light Blue",
            "#64B5F6": "Medium Blue",
            "#1565C0": "Dark Blue",
            "#E8F5E9": "Light Green",
            "#81C784": "Medium Green",
            "#2E7D32": "Dark Green",
            "#F3E5F5": "Light Purple",
            "#BA68C8": "Medium Purple",
            "#7B1FA2": "Dark Purple",
            "#FFF3E0": "Light Orange",
            "#FFB74D": "Medium Orange",
            "#EF6C00": "Dark Orange",
            "#FCE4EC": "Light Pink",
            "#F06292": "Medium Pink",
            "#C2185B": "Dark Pink",
            "#E9ECEF": "Light Gray",
            "#6C757D": "Medium Gray",
            "#212529": "Dark Gray",
        }
        return color_names.get(hex_color, "Custom Color")

    def generate_markdown_instructions(
        self, pattern: np.ndarray, save_path: Optional[str] = None
    ) -> str:
        """Generate detailed markdown instructions for the knitting pattern."""
        # Calculate pattern complexity
        difficulty = self.calculate_pattern_complexity(pattern)

        # Create markdown content
        md_content = [
            f"# Knitting Pattern Generated from {self.model_name}",
            f"\n## Pattern Information",
            f"- **Model**: {self.model_name}",
            f'- **Prompt**: "{self.prompt}"',
            f"- **Size**: {self.config.size}x{self.config.size} stitches",
            f"- **Difficulty**: {difficulty}",
            f"\n## Materials",
            "- Yarn in three colors (see color scheme below)",
            "- Knitting needles appropriate for your chosen yarn",
            "- Stitch markers (optional)",
            "- Row counter (recommended)",
            "- Tapestry needle for weaving in ends",
            f"\n## Gauge",
            "- Gauge is not critical for this pattern, but aim for:",
            "- 20-22 stitches and 28-30 rows = 4 inches (10 cm) in stockinette stitch",
            "- Use needle size appropriate for your chosen yarn to achieve a fabric you like",
            f"\n## Color Scheme",
        ]

        # Add color information with names
        for i, color in enumerate(self.config.color_scheme):
            color_name = self.get_color_name(color)
            md_content.append(f"- Color {i + 1}: {color_name} ({color})")

        md_content.extend(
            [
                f"\n## Pattern Notes",
                "- Pattern is worked flat (back and forth)",
                "- Odd-numbered rows (right side): Work from right to left",
                "- Even-numbered rows (wrong side): Work from left to right",
                "- The pattern is worked in stockinette stitch unless otherwise specified",
                "- Carry unused colors loosely along the back of work",
                "- Check gauge and adjust needle size accordingly",
                "- For best results, always work a gauge swatch before starting",
                f"\n## Row-by-Row Instructions",
            ]
        )

        # Add row-by-row instructions with more detail
        for i in range(len(pattern)):
            md_content.append(f"\n### Row {i + 1}")
            current_color = None
            count = 0
            instructions = []

            # For even-numbered rows (wrong side), reverse the order
            row = pattern[i][::-1] if (i + 1) % 2 == 0 else pattern[i]

            for value in row:
                color_num = int(value) + 1
                if color_num != current_color:
                    if current_color is not None:
                        color_name = self.get_color_name(
                            self.config.color_scheme[current_color - 1]
                        )
                        instructions.append(
                            f"Color {current_color} ({color_name}) for {count} stitches"
                        )
                    current_color = color_num
                    count = 1
                else:
                    count += 1

            # Add the last group
            if current_color is not None:
                color_name = self.get_color_name(
                    self.config.color_scheme[current_color - 1]
                )
                instructions.append(
                    f"Color {current_color} ({color_name}) for {count} stitches"
                )

            direction = "left to right" if (i + 1) % 2 == 0 else "right to left"
            md_content.append(
                f"Work {direction} as follows: " + " | ".join(instructions)
            )

        # Add tips and image reference
        md_content.extend(
            [
                f"\n## Tips",
                "- Use stitch markers between color changes to help track your progress",
                "- Weave in ends as you go to minimize finishing work",
                "- Block your finished piece to even out the stitches",
                "- Take a picture of each row as you complete it to track your progress",
                "- Consider using bobbins for each color to prevent tangling",
                "- Remember that even-numbered rows are worked from left to right (wrong side)",
                f"\n## Pattern Visualization",
                f"![Pattern Visualization]({os.path.basename(save_path)})"
                if save_path
                else "![Pattern Visualization](pattern_visualization.png)",
            ]
        )

        # Join all content with newlines
        full_content = "\n".join(md_content)

        # Save to file if path provided
        if save_path:
            md_path = save_path.replace(".png", ".md")
            os.makedirs(os.path.dirname(md_path), exist_ok=True)
            with open(md_path, "w") as f:
                f.write(full_content)
            print(f"Saved markdown instructions to {md_path}")

        return full_content

    def visualize(self, save_path: Optional[str] = None) -> None:
        """Visualize the generated pattern and create markdown instructions."""
        knit_pattern, prob_dist = self.generate_pattern()

        # Create a version of the pattern that shows how it will actually appear when knitted
        knitted_appearance = knit_pattern.copy()
        # Reverse every even-numbered row to show how it will look when knitted
        for i in range(len(knitted_appearance)):
            if (i + 1) % 2 == 0:  # even-numbered rows
                knitted_appearance[i] = knitted_appearance[i, ::-1]

        # Increase figure size for larger patterns
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

        # Plot probability distribution with a custom colormap based on model's color scheme
        base_color = self.config.color_scheme[-1]  # Use darkest color
        prob_cmap = sns.light_palette(base_color, as_cmap=True)
        sns.heatmap(prob_dist, ax=ax1, cmap=prob_cmap)
        ax1.set_title("Token Probability Distribution")

        # Plot original pattern (as generated)
        knit_cmap = plt.cm.colors.ListedColormap(self.config.color_scheme)
        sns.heatmap(
            knit_pattern,
            ax=ax2,
            cmap=knit_cmap,
            cbar=True,
            linewidths=0.5,
            linecolor="black",
        )  # Add grid
        ax2.set_title("Pattern as Generated\n(All rows right to left)")
        # Add row numbers on the left
        ax2.set_yticks(np.arange(len(knit_pattern)) + 0.5)
        ax2.set_yticklabels(range(1, len(knit_pattern) + 1))
        # Add stitch numbers on top
        ax2.set_xticks(np.arange(len(knit_pattern[0])) + 0.5)
        ax2.set_xticklabels(range(1, len(knit_pattern[0]) + 1))

        # Plot knitted appearance (with alternating row directions)
        sns.heatmap(
            knitted_appearance,
            ax=ax3,
            cmap=knit_cmap,
            cbar=True,
            linewidths=0.5,
            linecolor="black",
        )  # Add grid
        ax3.set_title("Pattern as Knitted\n(Alternating row directions)")
        # Add row numbers on the left
        ax3.set_yticks(np.arange(len(knitted_appearance)) + 0.5)
        ax3.set_yticklabels(range(1, len(knitted_appearance) + 1))
        # Add stitch numbers on top
        ax3.set_xticks(np.arange(len(knitted_appearance[0])) + 0.5)
        ax3.set_xticklabels(range(1, len(knitted_appearance[0]) + 1))

        plt.suptitle(
            f'Pattern Generation for {self.model_name}\nPrompt: "{self.prompt}"'
        )

        if save_path:
            clean_filename = "".join(
                c if c.isalnum() or c in ("-", "_", ".") else "_" for c in save_path
            )
            os.makedirs("outputs", exist_ok=True)
            save_path = os.path.join("outputs", clean_filename)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Saved visualization to {save_path}")

            # Generate and save markdown instructions
            self.generate_markdown_instructions(knit_pattern, save_path)
        else:
            plt.show()

        plt.close()

    def print_knitting_instructions(self, pattern: np.ndarray) -> None:
        """Print human-readable knitting instructions with color information."""
        print("\nKnitting Instructions:")
        print("\nColor Scheme:")
        for i, color in enumerate(self.config.color_scheme):
            print(f"Color {i + 1}: {color}")

        print("\nPattern (20x20):")
        for i in range(len(pattern)):
            print(f"\nRow {i + 1}:")
            current_color = None
            count = 0
            instructions = []

            for value in pattern[i]:
                color_num = int(value) + 1
                if color_num != current_color:
                    if current_color is not None:
                        instructions.append(
                            f"Color {current_color} for {count} stitches"
                        )
                    current_color = color_num
                    count = 1
                else:
                    count += 1

            # Add the last group
            if current_color is not None:
                instructions.append(f"Color {current_color} for {count} stitches")

            print(" | ".join(instructions))


def main():
    models = [
        "bert-base-uncased",
        "gpt2-xl",
        "google-t5/t5-11b",
        "EleutherAI/gpt-j-6B",
        "bigscience/bloom-7b1",
        "meta-llama/Llama-2-13b-hf",
        "tiiuae/falcon-7b",
    ]

    prompts = [
        "Nothing will come of nothing",
        "As flies to wanton boys are we to th' gods: / They kill us for their sport.",
        "Men must endure / Their going hence, even as their coming hither.",
    ]

    for model_name in models:
        print(f"\nProcessing {model_name}...")
        for prompt in prompts:
            try:
                generator = LMPatternGenerator(model_name, prompt)
                clean_model_name = model_name.split("/")[-1].replace("-", "_")
                generator.visualize(f"pattern_{clean_model_name}_{prompt[:20]}.png")
            except Exception as e:
                print(f"Error processing {model_name} with prompt '{prompt}': {str(e)}")
                continue


if __name__ == "__main__":
    main()
