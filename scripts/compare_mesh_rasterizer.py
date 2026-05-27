import torch
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

def save_difference_image(file_path_1, file_path_2, output_path):
    # 1. Load the tensors
    # map_location='cpu' ensures it loads even if saved on a GPU you don't have active
    tensor1 = torch.load(file_path_1, map_location='cpu')
    # print("tensor1 shape:", tensor1.shape)
    tensor2 = torch.load(file_path_2, map_location='cpu')
    # print("tensor2 shape:", tensor2.shape)
    
    # 2. Ensure they are the same shape
    if tensor1.shape != tensor2.shape:
        print(f"Error: Shapes do not match. {tensor1.shape} vs {tensor2.shape}")
        return

    # 3. Calculate absolute difference
    # .detach() removes gradient info, .cpu() moves to CPU, .numpy() converts to array
    diff = torch.abs(tensor1 - tensor2)
    
    # Handle dimensions: if (1, H, W) or (3, H, W), we might need to squeeze or pick a channel
    # This block flattens 3D tensors to 2D by taking the mean across channels, 
    # or squeezes if it's just a singleton dimension.
    if diff.ndim == 3:
        if diff.shape[0] in [1, 3, 4]: # Assuming (C, H, W)
            diff = diff.mean(dim=0) 
        elif diff.shape[2] in [1, 3, 4]: # Assuming (H, W, C)
            diff = diff.mean(dim=2)
            
    diff_np = diff.detach().cpu().numpy()

    # 4. Plot and Save
    plt.figure(figsize=(10, 8))
    
    # 'viridis' or 'inferno' are good for heatmaps. 'gray' for grayscale.
    plt.imshow(diff_np, cmap='inferno') 
    plt.colorbar(label='Difference Magnitude')
    plt.title(f'Difference between {file_path_1} and {file_path_2}')
    
    # Remove axis ticks if you want a clean image
    plt.axis('off') 
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Close memory
    
    print(f"Saved difference image to: {output_path}")

# === Usage ===
# Replace with your actual filenames
if __name__ == "__main__":
    parser = ArgumentParser(description="Compare two tensor files and save the difference image.")
    parser.add_argument("--ref_input_dir", type=str, help="")
    parser.add_argument("--dist_input_dir", type=str, help="")
    parser.add_argument("--output_dir", type=str, help="Path to save the difference image.")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir) / "diff_result.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for ref_file_path in Path(args.ref_input_dir).glob("r_*.pt"):
        dist_file_path = Path(args.dist_input_dir) / ref_file_path.name
        output_path = Path(args.output_dir) / f"{ref_file_path.stem}.png"
        if not dist_file_path.exists():
            print(f"Warning: Corresponding file {dist_file_path} does not exist. Skipping.")
            continue
        save_difference_image(ref_file_path, dist_file_path, output_path)

# Example command to run the script:    
"""
python compare_mesh_rasterizer.py \
    --ref_input_dir /mnt/data1/syjintw/NEU/dataset/milo_meshes/hotdog/pytorch3d/mesh_depth \
    --dist_input_dir /mnt/data1/syjintw/NEU/dataset/milo_meshes/hotdog/nvdiffrast/mesh_depth \
    --output_dir /mnt/data1/syjintw/NEU/dataset/milo_meshes/hotdog/diff_output
"""