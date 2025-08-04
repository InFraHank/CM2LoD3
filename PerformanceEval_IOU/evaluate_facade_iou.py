import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_and_resize_image(image_path, target_size=None, is_ground_truth=False):
    """Load image and resize if needed."""
    path = Path(image_path)
    
    if not path.exists():
        raise ValueError(f"Image file does not exist: {path}")
    
    # Try using PIL
    try:
        img = Image.open(path)
        img = np.array(img)
        
        # Convert to RGB if needed
        if len(img.shape) == 2:  # Grayscale
            img = np.stack([img, img, img], axis=2)
        elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        
        if target_size is not None:
            img = np.array(Image.fromarray(img).resize(target_size, Image.NEAREST))
        
        return img
    except Exception as e:
        print(f"Error with PIL: {e}, trying OpenCV...")
        
    # Fallback to OpenCV
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image (returned None): {path}")
    
    # Convert to RGB
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
    
    return img


def create_binary_masks(image, is_ground_truth=False):
    """Convert colored image to binary window and door masks."""
    window_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    door_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    if is_ground_truth:
        # Ground truth: windows are blue (0,0,255), doors are red (255,0,0)
        window_mask[
            (image[:, :, 2] > 200) & 
            (image[:, :, 0] < 100) & 
            (image[:, :, 1] < 100)
        ] = 1
        door_mask[
            (image[:, :, 0] > 200) & 
            (image[:, :, 1] < 100) & 
            (image[:, :, 2] < 100)
        ] = 1
    else:
        # Predictions: windows are yellow (255,255,0), doors are brown (139,69,19)
        window_mask[
            (image[:, :, 0] > 200) & 
            (image[:, :, 1] > 200) & 
            (image[:, :, 2] < 100)
        ] = 1
        door_mask[
            ((image[:, :, 0] > 100) & (image[:, :, 0] < 180)) & 
            ((image[:, :, 1] > 40) & (image[:, :, 1] < 100)) & 
            ((image[:, :, 2] > 0) & (image[:, :, 2] < 50))
        ] = 1
    
    return window_mask, door_mask


def calculate_metrics(pred_mask, gt_mask):
    """Calculate IoU, precision, recall, and F1 score."""
    # True positives: predicted positive and actually positive
    tp = np.logical_and(pred_mask, gt_mask).sum()
    # False positives: predicted positive but actually negative
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    # False negatives: predicted negative but actually positive
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
    # True negatives: predicted negative and actually negative
    tn = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum()
    
    # Calculate IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-10)  # Avoid division by zero
    
    # Calculate precision (how many selected items are relevant)
    precision = tp / (tp + fp + 1e-10)
    
    # Calculate recall (how many relevant items are selected)
    recall = tp / (tp + fn + 1e-10)
    
    # Calculate F1 score (harmonic mean of precision and recall)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }


def create_overlay_visualization(gt_image, gt_masks, pred_image, pred_masks, output_path, approach_name, metrics=None):
    """Create visualization showing ground truth with prediction overlay."""
    gt_window_mask, gt_door_mask = gt_masks
    pred_window_mask, pred_door_mask = pred_masks
    
    # Create a comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    
    # Ground truth image
    axes[0, 0].imshow(gt_image)
    axes[0, 0].set_title("Ground Truth Image")
    axes[0, 0].axis('off')
    
    # Prediction image
    axes[0, 1].imshow(pred_image)
    axes[0, 1].set_title(f"{approach_name} Prediction")
    axes[0, 1].axis('off')
    
    # Ground truth masks
    height, width = gt_window_mask.shape
    gt_vis = np.zeros((height, width, 3), dtype=np.uint8)
    gt_vis[gt_window_mask == 1] = [0, 0, 255]  # Blue for windows
    gt_vis[gt_door_mask == 1] = [255, 0, 0]    # Red for doors
    axes[0, 2].imshow(gt_vis)
    axes[0, 2].set_title("Ground Truth Masks")
    axes[0, 2].axis('off')
    
    # Prediction masks
    pred_vis = np.zeros((height, width, 3), dtype=np.uint8)
    pred_vis[pred_window_mask == 1] = [255, 255, 0]  # Yellow for windows
    pred_vis[pred_door_mask == 1] = [139, 69, 19]    # Brown for doors
    axes[1, 0].imshow(pred_vis)
    axes[1, 0].set_title(f"{approach_name} Masks")
    axes[1, 0].axis('off')
    
    # Create overlay with ground truth + prediction
    overlay = gt_image.copy()
    
    # Add semi-transparent overlays
    alpha = 0.5
    overlay_mask = np.zeros_like(overlay, dtype=np.float32)
    
    # Windows: ground truth in blue, prediction in yellow
    overlay_mask[gt_window_mask == 1] += np.array([0, 0, 255], dtype=np.float32) * 0.5
    overlay_mask[pred_window_mask == 1] += np.array([255, 255, 0], dtype=np.float32) * 0.5
    
    # Doors: ground truth in red, prediction in brown
    overlay_mask[gt_door_mask == 1] += np.array([255, 0, 0], dtype=np.float32) * 0.5
    overlay_mask[pred_door_mask == 1] += np.array([139, 69, 19], dtype=np.float32) * 0.5
    
    # Ensure values are in valid range
    overlay_mask = np.clip(overlay_mask, 0, 255).astype(np.uint8)
    
    # Blend overlay with original image
    overlay = cv2.addWeighted(overlay, 0.7, overlay_mask, 0.3, 0)
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Overlay (GT + Prediction)")
    axes[1, 1].axis('off')
    
    # Difference visualization (TP, FP, FN)
    diff_vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Window differences
    # True positives (green)
    diff_vis[(gt_window_mask == 1) & (pred_window_mask == 1)] = [0, 255, 0]
    # False positives (yellow)
    diff_vis[(gt_window_mask == 0) & (pred_window_mask == 1)] = [255, 255, 0]
    # False negatives (red)
    diff_vis[(gt_window_mask == 1) & (pred_window_mask == 0)] = [255, 0, 0]
    
    # Door differences
    # True positives (green)
    diff_vis[(gt_door_mask == 1) & (pred_door_mask == 1)] = [0, 255, 0]
    # False positives (yellow)
    diff_vis[(gt_door_mask == 0) & (pred_door_mask == 1)] = [255, 255, 0]
    # False negatives (red)
    diff_vis[(gt_door_mask == 1) & (pred_door_mask == 0)] = [255, 0, 0]
    
    axes[1, 2].imshow(diff_vis)
    axes[1, 2].set_title("Difference (Green=TP, Yellow=FP, Red=FN)")
    axes[1, 2].axis('off')
    
    # Add a legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=(0, 1, 0), label='True Positive'),
        plt.Rectangle((0, 0), 1, 1, color=(1, 1, 0), label='False Positive'),
        plt.Rectangle((0, 0), 1, 1, color=(1, 0, 0), label='False Negative')
    ]
    
    # Add metrics to the figure if provided
    if metrics:
        metrics_text = (
            f"Window Metrics:\n"
            f"IoU: {metrics.get('Window IoU', 0):.4f}\n"
            f"Precision: {metrics.get('Window Precision', 0):.4f}\n"
            f"Recall: {metrics.get('Window Recall', 0):.4f}\n"
            f"F1: {metrics.get('Window F1', 0):.4f}\n\n"
        )
        
        if 'Door IoU' in metrics:
            metrics_text += (
                f"Door Metrics:\n"
                f"IoU: {metrics.get('Door IoU', 0):.4f}\n"
                f"Precision: {metrics.get('Door Precision', 0):.4f}\n"
                f"Recall: {metrics.get('Door Recall', 0):.4f}\n"
                f"F1: {metrics.get('Door F1', 0):.4f}\n\n"
            )
            
        metrics_text += (
            f"Overall:\n"
            f"IoU: {metrics.get('Overall IoU', 0):.4f}\n"
            f"Precision: {metrics.get('Overall Precision', 0):.4f}\n"
            f"Recall: {metrics.get('Overall Recall', 0):.4f}\n"
            f"F1: {metrics.get('Overall F1', 0):.4f}"
        )
        
        # Add a text box with metrics
        plt.figtext(0.5, 0.01, metrics_text, ha="center", fontsize=10, 
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2 if metrics else 0.1)  # Make room for the legend and metrics
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved overlay visualization to: {output_path}")
    
    return overlay


def evaluate_facade(facade_dir, output_dir, facade_name):
    """Evaluate predictions for a single facade."""
    # Find ground truth file
    gt_files = glob.glob(os.path.join(facade_dir, "*ground_truth*.png")) + \
               glob.glob(os.path.join(facade_dir, "*gt*.png"))
    
    if not gt_files:
        print(f"No ground truth file found in {facade_dir}")
        return None
    
    ground_truth_path = gt_files[0]
    print(f"Found ground truth file: {ground_truth_path}")
    
    # Load ground truth
    gt_image = load_and_resize_image(ground_truth_path, is_ground_truth=True)
    target_size = (gt_image.shape[1], gt_image.shape[0])
    
    # Create binary masks for ground truth
    gt_window_mask, gt_door_mask = create_binary_masks(gt_image, is_ground_truth=True)
    print(f"Ground truth contains {gt_window_mask.sum()} window pixels and {gt_door_mask.sum()} door pixels")
    
    # Check if we have both classes or just windows
    has_doors = gt_door_mask.sum() > 0
    
    # Define prediction files to look for
    prediction_files = {
        "UNet": glob.glob(os.path.join(facade_dir, "*unet*.png")),
        "MaskRCNN": glob.glob(os.path.join(facade_dir, "*maskrcnn*.png")),
        "Fusion (before boxes)": glob.glob(os.path.join(facade_dir, "*fusion*before*.png")),
        "Fusion (after boxes)": glob.glob(os.path.join(facade_dir, "*fusion*after*.png"))
    }
    
    # Use filenames if glob doesn't find files
    if not any(files for files in prediction_files.values()):
        prediction_files = {
            "UNet": [os.path.join(facade_dir, "01_unet_prediction.png")],
            "MaskRCNN": [os.path.join(facade_dir, "02_maskrcnn_prediction.png")],
            "Fusion (before boxes)": [os.path.join(facade_dir, "03_fusion_prediction_before_contours.png")],
            "Fusion (after boxes)": [os.path.join(facade_dir, "04_fusion_prediction_after_rectangles.png")]
        }
    
    results = []
    
    # Process each prediction file
    for approach_name, files in prediction_files.items():
        if not files or not os.path.exists(files[0]):
            print(f"Warning: No {approach_name} prediction found for {facade_name}, skipping...")
            continue
        
        pred_path = files[0]
        
        try:
            # Load and resize prediction
            pred_image = load_and_resize_image(pred_path, target_size=target_size)
            
            # Create binary masks for prediction
            pred_window_mask, pred_door_mask = create_binary_masks(pred_image)
            
            # Calculate metrics for windows
            window_metrics = calculate_metrics(pred_window_mask, gt_window_mask)
            
            result_data = {
                "Facade": facade_name,
                "Approach": approach_name,
                "Window IoU": window_metrics["IoU"],
                "Window Precision": window_metrics["Precision"],
                "Window Recall": window_metrics["Recall"],
                "Window F1": window_metrics["F1"]
            }
            
            # Add door metrics only if doors exist in ground truth
            if has_doors:
                door_metrics = calculate_metrics(pred_door_mask, gt_door_mask)
                result_data.update({
                    "Door IoU": door_metrics["IoU"],
                    "Door Precision": door_metrics["Precision"],
                    "Door Recall": door_metrics["Recall"],
                    "Door F1": door_metrics["F1"]
                })
            
            # Calculate overall metrics (considering both classes together)
            pred_combined = np.logical_or(pred_window_mask, pred_door_mask)
            gt_combined = np.logical_or(gt_window_mask, gt_door_mask)
            overall_metrics = calculate_metrics(pred_combined, gt_combined)
            
            result_data.update({
                "Overall IoU": overall_metrics["IoU"],
                "Overall Precision": overall_metrics["Precision"],
                "Overall Recall": overall_metrics["Recall"],
                "Overall F1": overall_metrics["F1"]
            })
            
            # Create and save overlay visualization if output_dir is provided
            if output_dir:
                vis_path = os.path.join(output_dir, f"{facade_name}_{approach_name.replace(' ', '_').lower()}.png")
                create_overlay_visualization(
                    gt_image, 
                    (gt_window_mask, gt_door_mask),
                    pred_image,
                    (pred_window_mask, pred_door_mask),
                    vis_path,
                    approach_name,
                    metrics=result_data
                )
            
            results.append(result_data)
        except Exception as e:
            print(f"Error processing {approach_name} for {facade_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print(f"No predictions could be evaluated for {facade_name}")
        return None
    
    # Create results DataFrame for this facade
    df = pd.DataFrame(results)
    
    return df


def evaluate_all_facades(facades_dir, output_dir):
    """Evaluate all facades and compile results."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    visualization_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    per_facade_dir = os.path.join(output_dir, "per_facade_results")
    os.makedirs(per_facade_dir, exist_ok=True)
    
    # Find all facade directories
    facade_dirs = []
    
    # Try to find subdirectories that contain both ground truth and predictions
    for item in os.listdir(facades_dir):
        item_path = os.path.join(facades_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory contains ground truth and predictions
            has_gt = any("ground_truth" in f.lower() or "gt" in f.lower() for f in os.listdir(item_path) if f.endswith((".png", ".jpg")))
            has_pred = any("prediction" in f.lower() or "unet" in f.lower() or "mask" in f.lower() for f in os.listdir(item_path) if f.endswith((".png", ".jpg")))
            
            if has_gt and has_pred:
                facade_dirs.append((item, item_path))
    
    # If no structured subdirectories found, look for facade-specific files in the main directory
    if not facade_dirs:
        # Group files by facade identifier
        facade_files = defaultdict(list)
        for file in os.listdir(facades_dir):
            if file.endswith((".png", ".jpg")):
                # Try to extract facade identifier from filename
                parts = file.split("_")
                if len(parts) > 1:
                    facade_id = parts[0]  # Assuming first part is facade ID
                    facade_files[facade_id].append(file)
        
        # Check each group has ground truth and predictions
        for facade_id, files in facade_files.items():
            if any("ground_truth" in f.lower() or "gt" in f.lower() for f in files):
                facade_dirs.append((facade_id, facades_dir))
    
    print(f"Found {len(facade_dirs)} facades to evaluate")
    
    if not facade_dirs:
        print("No facades found for evaluation!")
        return None
    
    # Collect results from all facades
    all_results = []
    
    for facade_name, facade_dir in facade_dirs:
        print(f"\nEvaluating facade: {facade_name}")
        facade_results = evaluate_facade(
            facade_dir, 
            visualization_dir, 
            facade_name
        )
        
        if facade_results is not None:
            # Save per-facade results
            facade_csv = os.path.join(per_facade_dir, f"{facade_name}_results.csv")
            facade_results.to_csv(facade_csv, index=False)
            print(f"Saved individual results to: {facade_csv}")
            
            # Add to all results
            all_results.append(facade_results)
    
    if not all_results:
        print("No valid results generated for any facade!")
        return None
    
    # Concatenate all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Get list of approaches
    approaches = combined_results["Approach"].unique()
    
    # Calculate average per approach
    avg_results = []
    
    # List all metrics we want to average
    metrics = [
        "Overall IoU", "Window IoU", "Door IoU",
        "Overall Precision", "Window Precision", "Door Precision",
        "Overall Recall", "Window Recall", "Door Recall",
        "Overall F1", "Window F1", "Door F1"
    ]
    
    for approach in approaches:
        approach_results = combined_results[combined_results["Approach"] == approach]
        
        avg_data = {"Approach": approach}
        
        # Calculate mean and std for each metric if it exists
        for metric in metrics:
            if metric in approach_results.columns:
                avg_data[metric] = approach_results[metric].mean()
                avg_data[f"{metric} (std)"] = approach_results[metric].std()
        
        # Add count of facades
        avg_data["Facades count"] = len(approach_results)
        
        avg_results.append(avg_data)
    
    # Create average results DataFrame
    avg_df = pd.DataFrame(avg_results).sort_values("Overall F1", ascending=False)
    
    return avg_df, combined_results


def save_summary_results(avg_df, combined_results, output_dir):
    """Save and visualize summary results."""
    # Save average results
    summary_csv = os.path.join(output_dir, "summary_results.csv")
    avg_df.to_csv(summary_csv, index=False)
    print(f"Saved summary results to: {summary_csv}")
    
    # Save all results
    all_results_csv = os.path.join(output_dir, "all_results.csv")
    combined_results.to_csv(all_results_csv, index=False)
    
    # Print formatted table (focus on key metrics for display)
    key_metrics = ["Overall IoU", "Overall Precision", "Overall Recall", "Overall F1"]
    display_df = avg_df[["Approach"] + key_metrics].copy()
    
    print("\nSummary Results (averaged across all facades):")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(display_df.to_string(float_format='{:.4f}'.format, index=False))
    print("=" * 80)
    
    # Create LaTeX table for key metrics
    latex_table = display_df.to_latex(float_format='{:.4f}'.format, index=False)
    with open(os.path.join(output_dir, "summary_latex.txt"), 'w') as f:
        f.write(latex_table)
    
    # Create markdown table for key metrics
    markdown_table = display_df.to_markdown(floatfmt='.4f', index=False)
    with open(os.path.join(output_dir, "summary_markdown.md"), 'w') as f:
        f.write(markdown_table)
    
    # Create visual summary
    create_summary_visualizations(avg_df, combined_results, output_dir)


def create_summary_visualizations(avg_df, combined_results, output_dir):
    """Create visualization summarizing results across all facades."""
    
    # 1. Bar chart comparison of IoU, Precision, Recall, and F1 scores
    plt.figure(figsize=(14, 8))
    
    # Key metrics for overall performance
    metrics = ["Overall IoU", "Overall Precision", "Overall Recall", "Overall F1"]
    
    # Set positions for bars
    x = np.arange(len(avg_df))
    width = 0.2
    offsets = np.linspace(-(len(metrics)-1)/2*width, (len(metrics)-1)/2*width, len(metrics))
    
    # Plot each metric as a grouped bar
    for i, metric in enumerate(metrics):
        if metric in avg_df.columns:
            std_col = f"{metric} (std)"
            if std_col in avg_df.columns:
                plt.bar(x + offsets[i], avg_df[metric], width, label=metric.replace("Overall ", ""), 
                       yerr=avg_df[std_col], capsize=5)
            else:
                plt.bar(x + offsets[i], avg_df[metric], width, label=metric.replace("Overall ", ""))
    
    plt.xlabel('Approach')
    plt.ylabel('Score')
    plt.title('Average Performance Metrics Across All Facades')
    plt.xticks(x, avg_df['Approach'], rotation=30, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "overall_metrics_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Class-specific F1 scores
    class_metrics = []
    if "Window F1" in avg_df.columns:
        class_metrics.append("Window F1")
    if "Door F1" in avg_df.columns:
        class_metrics.append("Door F1")
    
    if class_metrics:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(avg_df))
        width = 0.35
        
        offsets = np.linspace(-(len(class_metrics)-1)/2*width, (len(class_metrics)-1)/2*width, len(class_metrics))
        
        for i, metric in enumerate(class_metrics):
            std_col = f"{metric} (std)"
            if std_col in avg_df.columns:
                plt.bar(x + offsets[i], avg_df[metric], width, label=metric.replace(" F1", ""), 
                       yerr=avg_df[std_col], capsize=5)
            else:
                plt.bar(x + offsets[i], avg_df[metric], width, label=metric.replace(" F1", ""))
        
        plt.xlabel('Approach')
        plt.ylabel('F1 Score')
        plt.title('Class-specific F1 Scores Across All Facades')
        plt.xticks(x, avg_df['Approach'], rotation=30, ha='right')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(output_dir, "class_f1_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Precision-Recall plot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot for each approach
    approaches = avg_df['Approach'].unique()
    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', '*']
    
    for i, approach in enumerate(approaches):
        approach_data = combined_results[combined_results['Approach'] == approach]
        
        # Plot overall precision vs recall for each facade
        plt.scatter(
            approach_data['Overall Recall'], 
            approach_data['Overall Precision'],
            label=approach,
            marker=markers[i % len(markers)],
            s=100,
            alpha=0.7
        )
        
        # Also plot the average point (larger)
        avg_recall = approach_data['Overall Recall'].mean()
        avg_precision = approach_data['Overall Precision'].mean()
        plt.scatter(
            avg_recall, 
            avg_precision,
            marker=markers[i % len(markers)],
            s=200,
            facecolors='none',
            edgecolors='black',
            linewidth=2
        )
    
    # Add diagonal lines for F1 score references
    f1_values = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for f1 in f1_values:
        x = np.linspace(0.01, 1, 100)
        y = (f1 * x) / (2 * x - f1)
        valid_idx = (y >= 0) & (y <= 1)
        plt.plot(x[valid_idx], y[valid_idx], 'k--', alpha=0.3)
        # Add F1 label at appropriate position
        idx = len(x[valid_idx]) // 2
        if idx > 0:
            plt.annotate(f'F1={f1}', 
                        xy=(x[valid_idx][idx], y[valid_idx][idx]),
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Plot for All Facades')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Radar chart for comparing approaches across metrics
    key_metrics = ["Overall IoU", "Overall Precision", "Overall Recall", "Overall F1"]
    
    # Create a figure for the radar chart
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    
    # Number of metrics
    N = len(key_metrics)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Plot each approach
    for i, approach in enumerate(avg_df['Approach']):
        # Get metrics for this approach
        values = [avg_df.loc[avg_df['Approach'] == approach, metric].values[0] for metric in key_metrics]
        values += values[:1]  # Close the loop
        
        # Plot
        ax.plot(angles, values, linewidth=2, label=approach)
        ax.fill(angles, values, alpha=0.1)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.replace("Overall ", "") for metric in key_metrics])
    
    # Draw y-axis lines
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    
    plt.title('Comparison of Approaches Across Metrics')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary visualizations to: {output_dir}")


def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    facades_dir = os.path.join(script_dir, "Evaluations/Facades")
    output_dir = os.path.join(script_dir, "Evaluations/Results")
    
    print(f"Starting multi-facade evaluation...")
    print(f"Facades directory: {facades_dir}")
    print(f"Output directory: {output_dir}")
    
    # Evaluate all facades
    avg_df, combined_results = evaluate_all_facades(facades_dir, output_dir)
    
    if avg_df is not None:
        # Save and visualize summary results
        save_summary_results(avg_df, combined_results, output_dir)
        print("Evaluation complete!")
    else:
        print("Evaluation failed - no valid results.")


if __name__ == "__main__":
    main()