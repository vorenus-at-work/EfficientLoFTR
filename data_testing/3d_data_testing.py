#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
#%%
def load_coco_example():
    """
    Demonstrate loading and processing a COCO format image with annotations
    
    COCO format:
    - RGB images (typically JPG)
    - Annotations stored in JSON files
    - Contains instance segmentation masks, bounding boxes, keypoints
    - Multiple object categories
    """
    # Simulated COCO image (typically RGB)
    coco_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Simulated COCO annotation (JSON format)
    coco_annotation = {
        "images": [{
            "id": 1,
            "file_name": "000000123456.jpg",
            "width": 640,
            "height": 480
        }],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,  # person
            "bbox": [100, 150, 200, 300],  # [x, y, width, height]
            "segmentation": [[100, 150, 300, 150, 300, 450, 100, 450]],
            "area": 60000,
            "iscrowd": 0
        }],
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person"
        }]
    }
    
    # Display the image and bounding box
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(coco_img)
    plt.title("COCO RGB Image")
    
    # Add bounding box visualization
    bbox = coco_annotation["annotations"][0]["bbox"]
    x, y, w, h = bbox
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
    plt.text(x, y-10, "Person", fontsize=12, color='red')
    
    return coco_img, coco_annotation

def load_megadepth_example():
    """
    Demonstrate loading and processing a MegaDepth format image with depth information
    
    MegaDepth format:
    - RGB images (typically JPG)
    - Depth maps stored as separate files (.h5 or .npy)
    - Camera intrinsics/extrinsics included
    - Contains real-world scale information
    - Specialized for 3D structure, not object detection
    """
    # Simulated MegaDepth RGB image
    megadepth_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Simulated MegaDepth depth map (single channel float values in meters)
    # Values typically range from 0 (close) to larger values (far)
    megadepth_depth = np.random.uniform(0, 100, (480, 640)).astype(np.float32)
    
    # Camera parameters (typically stored separately)
    megadepth_camera = {
        "intrinsics": {
            "focal_length": [555.5, 555.5],  # fx, fy
            "principal_point": [320.0, 240.0]  # cx, cy
        },
        "extrinsics": {
            "rotation": np.eye(3).tolist(),
            "translation": [0, 0, 0]
        }
    }
    
    # Display the RGB image and depth map
    plt.subplot(1, 2, 2)
    plt.imshow(megadepth_img)
    plt.title("MegaDepth RGB Image")
    
    # Create a small subplot to show the depth visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(megadepth_depth, cmap='viridis')
    plt.colorbar(label='Depth (m)')
    plt.title("MegaDepth Depth Map")
    
    return megadepth_img, megadepth_depth, megadepth_camera

def compare_formats():
    """Compare the key differences between COCO and MegaDepth formats"""
    print("=== COCO Dataset Format ===")
    print("- Purpose: Object detection, segmentation, and keypoint detection")
    print("- Image format: RGB (typically JPEG)")
    print("- Annotations: JSON files with bounding boxes, masks, keypoints")
    print("- Classes: Multiple object categories (80 in COCO)")
    print("- Structure: Flat directory of images with separate annotation files")
    print("- Metadata: Object IDs, categories, crowds, areas\n")
    
    print("=== MegaDepth Dataset Format ===")
    print("- Purpose: Depth estimation, 3D reconstruction, structure-from-motion")
    print("- Image format: RGB (typically JPEG) + separate depth maps (.h5/.npy)")
    print("- Annotations: Depth values, camera parameters (intrinsics/extrinsics)")
    print("- Classes: No object classes (focused on scene geometry)")
    print("- Structure: Scene-based hierarchy with multi-view image sets")
    print("- Metadata: Camera calibration, real-world scale, dense depth\n")
    
    print("=== Key Differences ===")
    print("1. COCO focuses on 2D object understanding (what and where)")
    print("2. MegaDepth focuses on 3D scene understanding (depth and structure)")
    print("3. COCO has object-level annotations, MegaDepth has pixel-level depth")
    print("4. COCO uses JSON for annotations, MegaDepth uses HDF5/NPY for depth")
    print("5. MegaDepth includes camera calibration, COCO doesn't")
#%%
# Example usage
if __name__ == "__main__":
    # Load and display example data
    coco_data = load_coco_example()
    megadepth_data = load_megadepth_example()
    
    # Print format comparison
    compare_formats()
    
    # Example of how these might be used in practice
    print("\n=== Example Usage Scenarios ===")
    print("COCO: Training an object detector or instance segmentation model")
    print("MegaDepth: Training a monocular depth estimation model")
# %%
