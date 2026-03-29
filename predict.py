"""
Pill grid detection (YOLO) + grid geometry.
Model loads lazily on first inference so importing this module is safe for API servers.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        import ultralytics

        ultralytics.checks()
        from ultralytics import YOLO

        weights = os.environ.get(
            "PILL_GRID_WEIGHTS",
            "runs/detect/train/weights/best.pt",
        )
        _yolo_model = YOLO(weights)
    return _yolo_model


def cluster_points(points, threshold):
    """Cluster points that are close together using simple distance-based clustering"""
    if len(points) == 0:
        return []

    points = np.array(points)
    sorted_indices = np.argsort(points)
    sorted_points = points[sorted_indices]

    clusters = np.zeros(len(points), dtype=int)
    cluster_id = 0
    current_cluster_points = [sorted_points[0]]
    clusters[sorted_indices[0]] = cluster_id

    for i in range(1, len(sorted_points)):
        if sorted_points[i] - current_cluster_points[-1] <= threshold:
            clusters[sorted_indices[i]] = cluster_id
            current_cluster_points.append(sorted_points[i])
        else:
            cluster_id += 1
            clusters[sorted_indices[i]] = cluster_id
            current_cluster_points = [sorted_points[i]]

    return clusters.tolist()


def create_grid(centers, img_shape, boxes_xy):
    """Create grid based on pill centers"""
    if len(centers) == 0:
        return [], [], [], None

    centers_np = np.array(centers)
    x_coords = centers_np[:, 0]
    y_coords = centers_np[:, 1]

    x_sorted = np.sort(x_coords)
    y_sorted = np.sort(y_coords)

    if len(x_sorted) > 1:
        x_threshold = np.mean(np.diff(x_sorted)) * 0.3
        y_threshold = np.mean(np.diff(y_sorted)) * 0.3
    else:
        x_threshold = 50
        y_threshold = 50

    x_clusters = cluster_points(x_coords, x_threshold)
    y_clusters = cluster_points(y_coords, y_threshold)

    pill_grid = defaultdict(dict)

    for i, (x, y) in enumerate(centers):
        row_id = y_clusters[i]
        col_id = x_clusters[i]
        pill_grid[row_id][col_id] = (x, y)

    row_ids = sorted(pill_grid.keys())
    col_ids = sorted(set(col_id for row in pill_grid.values() for col_id in row.keys()))

    grid = []
    for row_id in row_ids:
        row = []
        for col_id in col_ids:
            if col_id in pill_grid[row_id]:
                row.append(pill_grid[row_id][col_id])
            else:
                row.append(None)
        grid.append(row)

    bottom_right_cell = None
    if len(grid) > 0:
        for r in range(len(grid) - 1, -1, -1):
            row_has_pills = False
            for c in range(len(grid[r]) - 1, -1, -1):
                if grid[r][c] is not None:
                    bottom_right_cell = (r, c)
                    row_has_pills = True
                    break
            if row_has_pills:
                break

    col_lines = []

    if len(col_ids) > 0:
        first_col_boxes = []
        for r in range(len(grid)):
            if grid[r][0] is not None:
                x, y = grid[r][0]
                idx = centers.index((x, y))
                first_col_boxes.append(boxes_xy[idx][0])

        if first_col_boxes:
            left_edge = min(first_col_boxes)
            col_lines.append(left_edge)

        if len(col_ids) > 1:
            for col_idx in range(len(col_ids) - 1):
                center_points = []
                for row_idx in range(len(grid)):
                    if grid[row_idx][col_idx] is not None and grid[row_idx][col_idx + 1] is not None:
                        center_x = (
                            grid[row_idx][col_idx][0] + grid[row_idx][col_idx + 1][0]
                        ) / 2
                        center_points.append(center_x)

                if center_points:
                    grid_line_x = np.mean(center_points)
                    col_lines.append(grid_line_x)

        last_col = len(grid[0]) - 1
        last_col_boxes = []

        for r in range(len(grid)):
            if grid[r][last_col] is not None:
                x, y = grid[r][last_col]
                idx = centers.index((x, y))
                last_col_boxes.append(boxes_xy[idx][2])

        if last_col_boxes:
            right_edge = max(last_col_boxes)
            col_lines.append(right_edge)

    row_lines = []
    if len(row_ids) > 0:
        avg_row_spacing = None
        if len(row_ids) > 1:
            row_spacings = []
            for row_idx in range(len(grid) - 1):
                spacings = []
                for col_idx in range(len(grid[0])):
                    if grid[row_idx][col_idx] is not None and grid[row_idx + 1][col_idx] is not None:
                        spacing = grid[row_idx + 1][col_idx][1] - grid[row_idx][col_idx][1]
                        spacings.append(spacing)
                if spacings:
                    row_spacings.append(np.mean(spacings))

            if row_spacings:
                avg_row_spacing = np.mean(row_spacings)

        first_row_centers = [grid[0][c][1] for c in range(len(grid[0])) if grid[0][c] is not None]
        if first_row_centers:
            if avg_row_spacing:
                top_edge = min(first_row_centers) - avg_row_spacing / 2
            else:
                top_edge = min(first_row_centers) - 20
            row_lines.append(max(0, top_edge))

        if len(row_ids) > 1:
            for row_idx in range(len(grid) - 1):
                center_points = []
                for col_idx in range(len(grid[0])):
                    if grid[row_idx][col_idx] is not None and grid[row_idx + 1][col_idx] is not None:
                        center_y = (
                            grid[row_idx][col_idx][1] + grid[row_idx + 1][col_idx][1]
                        ) / 2
                        center_points.append(center_y)

                if center_points:
                    grid_line_y = np.mean(center_points)
                    row_lines.append(grid_line_y)

        last_row_centers = [grid[-1][c][1] for c in range(len(grid[-1])) if grid[-1][c] is not None]
        if last_row_centers:
            if avg_row_spacing:
                bottom_edge = max(last_row_centers) + avg_row_spacing / 2
            else:
                bottom_edge = max(last_row_centers) + 20
            row_lines.append(min(img_shape[0], bottom_edge))

    return col_lines, row_lines, grid, bottom_right_cell


def run_pill_grid_on_bgr(
    img_bgr: np.ndarray,
    conf: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run YOLO + grid extraction on a BGR image (OpenCV).
    Returns a JSON-serializable dict (no numpy types in nested structures).
    """
    if conf is None:
        conf = float(os.environ.get("PILL_GRID_CONF", "0.4"))

    model = _get_yolo()
    prediction_results = model.predict(
        img_bgr,
        save=False,
        show_labels=False,
        show_boxes=False,
        conf=conf,
    )

    img_shape = img_bgr.shape[:2]
    out: Dict[str, Any] = {
        "pill_count": 0,
        "grid_columns": 0,
        "grid_rows": 0,
        "tray_coordinates": None,
        "cut_coordinates": None,
        "bottom_right_cell": None,
        "pill_center_xy": None,
        "servo_target_deg": None,
    }

    for result in prediction_results:
        centers: List[Tuple[int, int]] = []
        boxes_xy: List[Tuple[float, float, float, float]] = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                centers.append((center_x, center_y))
                boxes_xy.append((float(x1), float(y1), float(x2), float(y2)))

        out["pill_count"] = len(centers)
        if len(centers) == 0:
            return out

        col_lines, row_lines, grid, bottom_right_cell = create_grid(
            centers, img_shape, boxes_xy
        )
        out["grid_columns"] = max(0, len(col_lines) - 1)
        out["grid_rows"] = max(0, len(row_lines) - 1)
        out["bottom_right_cell"] = (
            [int(bottom_right_cell[0]), int(bottom_right_cell[1])]
            if bottom_right_cell
            else None
        )

        if len(col_lines) >= 2 and len(row_lines) >= 2:
            out["tray_coordinates"] = [
                int(col_lines[0]),
                int(row_lines[0]),
                int(col_lines[-1]),
                int(row_lines[-1]),
            ]

        if bottom_right_cell is not None:
            row_idx, col_idx = bottom_right_cell
            if col_idx < len(col_lines) - 1 and row_idx < len(row_lines) - 1:
                x1 = int(col_lines[col_idx])
                y1 = int(row_lines[row_idx])
                x2 = int(col_lines[col_idx + 1])
                y2 = int(row_lines[row_idx + 1])
                out["cut_coordinates"] = [x1, y1, x2, y2]
                pill_center = grid[row_idx][col_idx]
                if pill_center:
                    out["pill_center_xy"] = [int(pill_center[0]), int(pill_center[1])]
                target_x = (x1 + x2) / 2
                target_y = (y1 + y2) / 2
                img_h, img_w = img_shape[0], img_shape[1]
                servo_x = float(np.interp(target_x, [0, img_w], [0, 180]))
                servo_y = float(np.interp(target_y, [0, img_h], [0, 180]))
                out["servo_target_deg"] = {"x": round(servo_x, 2), "y": round(servo_y, 2)}

    return out


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "pill2.jpeg"
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"Cannot load {path}")
    payload = run_pill_grid_on_bgr(image)
    print(payload)
