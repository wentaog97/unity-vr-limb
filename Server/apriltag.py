#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import time
import argparse
import os
import json

try:
    import yaml  # optional, for loading .yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

from pupil_apriltags import Detector  # pip install pupil-apriltags


def load_intrinsics(path):
    """
    Load camera intrinsics (fx, fy, cx, cy) from JSON or YAML.
    Accepts:
      JSON: {"fx":..., "fy":..., "cx":..., "cy":..., "width":..., "height":...}
      YAML (OpenCV style): contains camera_matrix: [[fx,0,cx],[0,fy,cy],[0,0,1]] and image_width/height (optional)
    Returns (fx, fy, cx, cy, width, height) where width/height may be None.
    """
    if not path or not os.path.exists(path):
        return None

    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "r") as f:
            if ext in [".json"]:
                d = json.load(f)
                fx, fy, cx, cy = float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"])
                w = int(d.get("width")) if "width" in d else None
                h = int(d.get("height")) if "height" in d else None
                return fx, fy, cx, cy, w, h
            elif ext in [".yml", ".yaml"] and HAVE_YAML:
                data = yaml.safe_load(f)
                if "camera_matrix" in data:
                    K = np.array(data["camera_matrix"], dtype=np.float64)
                elif "K" in data:
                    K = np.array(data["K"], dtype=np.float64)
                else:
                    raise ValueError("YAML missing camera_matrix/K")
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                w = int(data.get("image_width")) if "image_width" in data else None
                h = int(data.get("image_height")) if "image_height" in data else None
                return fx, fy, cx, cy, w, h
            else:
                raise ValueError("Unsupported calibration file format. Use .json or .yaml/.yml")
    except Exception as e:
        print(f"[WARN] Failed to load intrinsics from {path}: {e}")
        return None


def approx_intrinsics(frame_width, frame_height, hfov_deg=60.0):
    """
    Make a reasonable guess at intrinsics if no calibration is available.
    Assumes square pixels and principal point at the image center.
    """
    hfov = np.deg2rad(hfov_deg)
    fx = (frame_width / 2.0) / np.tan(hfov / 2.0)
    fy = fx
    cx = frame_width / 2.0
    cy = frame_height / 2.0
    return fx, fy, cx, cy


def draw_tag_outline(img, corners, color=(0, 255, 255), thickness=2):
    c = corners.astype(int)
    for i in range(4):
        p1 = tuple(c[i])
        p2 = tuple(c[(i + 1) % 4])
        cv.line(img, p1, p2, color, thickness, cv.LINE_AA)


def rotation_to_euler_xyz(R):
    """
    Convert a 3x3 rotation matrix to XYZ Euler angles (radians).
    Returns roll (x), pitch (y), yaw (z).
    """
    sy = -R[2, 0]
    cy = np.sqrt(1 - sy**2)
    singular = cy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: cy ~ 0
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arcsin(sy)
        yaw = 0.0
    return roll, pitch, yaw


def draw_axes(img, K, dist, R, t, axis_len):
    """
    Draw XYZ axes (right=X, down=Y, out=Z in image convention; in tag frame we’ll use X-red, Y-green, Z-blue).
    Uses cv.projectPoints on 4 points: origin and the three axis endpoints in TAG frame.
    """
    # Axis endpoints in tag coordinate system (meters)
    pts_3d = np.float32([
        [0, 0, 0],                     # origin
        [axis_len, 0, 0],              # X
        [0, axis_len, 0],              # Y
        [0, 0, axis_len],              # Z
    ])
    rvec, _ = cv.Rodrigues(R)
    tvec = t.reshape(3, 1)
    pts_2d, _ = cv.projectPoints(pts_3d, rvec, tvec, K, dist)

    o = tuple(pts_2d[0].ravel().astype(int))
    x = tuple(pts_2d[1].ravel().astype(int))
    y = tuple(pts_2d[2].ravel().astype(int))
    z = tuple(pts_2d[3].ravel().astype(int))

    cv.line(img, o, x, (0, 0, 255), 3, cv.LINE_AA)   # X - red
    cv.line(img, o, y, (0, 255, 0), 3, cv.LINE_AA)   # Y - green
    cv.line(img, o, z, (255, 0, 0), 3, cv.LINE_AA)   # Z - blue


def draw_box_on_tag(img, K, dist, R, t, tag_size, height,
                    color_bottom=(0, 255, 255), color_top=(255, 0, 255),
                    color_vertical=(0, 255, 0), thickness=2):
    """
    Project and draw a 3D box whose bottom face lies on the tag plane (z=0 in tag frame)
    and whose top face is at z=height (in meters).
    The bottom face is a square of size tag_size centered at the tag origin.
    """
    half = float(tag_size) * 0.5

    # 8 corners in TAG coordinate frame
    pts_3d = np.float32([
        [-half, -half, 0.0],              # 0 bottom on tag plane
        [ +half, -half, 0.0],             # 1
        [ +half,  +half, 0.0],            # 2
        [ -half,  +half, 0.0],            # 3
        [-half, -half, -float(height)],   # 4 top (extrude away from tag +Z)
        [ +half, -half, -float(height)],  # 5
        [ +half,  +half, -float(height)], # 6
        [ -half,  +half, -float(height)], # 7
    ])

    rvec, _ = cv.Rodrigues(R)
    tvec = t.reshape(3, 1)
    pts_2d, _ = cv.projectPoints(pts_3d, rvec, tvec, K, dist)
    pts = pts_2d.reshape(-1, 2).astype(int)

    # Bottom face edges: (0-1-2-3-0)
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv.line(img, p1, p2, color_bottom, thickness, cv.LINE_AA)

    # Top face edges: (4-5-6-7-4)
    for i in range(4, 8):
        p1 = tuple(pts[i])
        p2 = tuple(pts[4 + ((i - 3) % 4)])
        cv.line(img, p1, p2, color_top, thickness, cv.LINE_AA)

    # Vertical edges: (0-4, 1-5, 2-6, 3-7)
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[i + 4])
        cv.line(img, p1, p2, color_vertical, thickness, cv.LINE_AA)


def put_multiline_text(img, lines, org=(10, 30), line_gap=22, color=(255, 255, 255)):
    x, y = org
    for line in lines:
        cv.putText(img, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(img, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv.LINE_AA)
        y += line_gap


def main():
    ap = argparse.ArgumentParser(description="AprilTag detection with sticky XYZ axes, pose overlay, and tag-aligned 3D box.")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--width", type=int, default=1280, help="Capture width")
    ap.add_argument("--height", type=int, default=720, help="Capture height")
    ap.add_argument("--family", type=str, default="tag36h11",
                    help="AprilTag family (e.g., tag36h11, tag25h9, tag16h5)")
    ap.add_argument("--tag-size", type=float, default=0.050,
                    help="Tag size in meters (outer black square width). Default: 0.050 m (5 cm)")
    ap.add_argument("--calib", type=str, default="",
                    help="Path to camera intrinsics file (.json/.yaml).")
    ap.add_argument("--fx", type=float, default=None, help="Focal length x (overrides calib)")
    ap.add_argument("--fy", type=float, default=None, help="Focal length y (overrides calib)")
    ap.add_argument("--cx", type=float, default=None, help="Principal point x (overrides calib)")
    ap.add_argument("--cy", type=float, default=None, help="Principal point y (overrides calib)")
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="Only used for approximate intrinsics if none provided.")
    ap.add_argument("--decimate", type=float, default=1.0, help="Detector decimate factor (speed/accuracy tradeoff)")
    ap.add_argument("--sigma", type=float, default=0.0, help="Quad sigma (blur)")
    ap.add_argument("--refine-edges", action="store_true", help="Refine edges in detector")
    ap.add_argument("--no-box", action="store_true", help="Disable drawing the 3D box on the tag")
    ap.add_argument("--box-height", type=float, default=0.03, help="3D box height in meters above tag plane")
    # Filtering to reduce false positives
    ap.add_argument("--min-margin", type=float, default=0.0, help="Minimum decision margin to accept a detection")
    ap.add_argument("--max-hamming", type=int, default=0, help="Maximum Hamming distance to accept a detection")
    ap.add_argument("--allow-ids", type=str, default="", help="Comma-separated list of allowed tag IDs (empty = allow all)")
    ap.add_argument("--min-area", type=float, default=0.0, help="Minimum detected tag area (px^2) to accept a detection")
    ap.add_argument("--output-data", action="store_true", help="Output pose data in JSON format for external consumption")
    ap.add_argument("--output-file", type=str, default="", help="Output pose data to file instead of stdout")
    args = ap.parse_args()

    cap = cv.VideoCapture(args.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("[ERROR] Could not open video capture.")
        return

    # Prepare intrinsics
    K = None
    dist = np.zeros((5, 1), dtype=np.float64)
    frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) or args.width
    frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or args.height

    loaded = load_intrinsics(args.calib) if args.calib else None
    if loaded:
        fx, fy, cx, cy, w_cal, h_cal = loaded
        if args.fx is not None: fx = args.fx
        if args.fy is not None: fy = args.fy
        if args.cx is not None: cx = args.cx
        if args.cy is not None: cy = args.cy
        # Optionally resize intrinsics if capture size differs from calibration
        if w_cal and h_cal and (w_cal != frame_w or h_cal != frame_h):
            sx = frame_w / float(w_cal)
            sy = frame_h / float(h_cal)
            fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
    else:
        # Manual overrides or approximate
        if all(v is not None for v in [args.fx, args.fy, args.cx, args.cy]):
            fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
        else:
            fx, fy, cx, cy = approx_intrinsics(frame_w, frame_h, hfov_deg=args.hfov)
            print(f"[WARN] Using approximate intrinsics (hfov={args.hfov}°). For accuracy, provide --calib or --fx/--fy/--cx/--cy.")

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    # Set up detector
    at_detector = Detector(
        families=args.family,
        nthreads=1,
        quad_decimate=args.decimate,
        quad_sigma=args.sigma,
        refine_edges=args.refine_edges,
    )

    print("[INFO] Press ESC to quit.")
    last_time = time.time()
    fps = 0.0
    
    # Setup data output
    output_file = None
    if args.output_data and args.output_file:
        try:
            output_file = open(args.output_file, 'w')
        except Exception as e:
            print(f"[ERROR] Could not open output file {args.output_file}: {e}")
            return

    # Parse allowed IDs once
    allowed_ids = None
    if args.allow_ids and args.allow_ids.strip():
        try:
            allowed_ids = set(int(s) for s in args.allow_ids.replace(" ", "").split(",") if s != "")
        except Exception:
            print("[WARN] Failed to parse --allow-ids. Ignoring.")
            allowed_ids = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame grab failed.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect tags; provide intrinsics & tag_size for pose solve
        detections = at_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=args.tag_size
        )

        now = time.time()
        dt = now - last_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        last_time = now

        # Filter detections to reduce noise
        filtered = []
        for d in detections:
            # Hamming filter
            if args.max_hamming >= 0 and d.hamming > args.max_hamming:
                continue
            # Decision margin filter
            dm = getattr(d, "decision_margin", None)
            if dm is not None and args.min_margin > 0.0 and dm < args.min_margin:
                continue
            # Allowed ID filter
            if allowed_ids is not None and d.tag_id not in allowed_ids:
                continue
            # Area filter (in pixels^2)
            if args.min_area > 0.0:
                a = float(cv.contourArea(d.corners.astype(np.float32)))
                if a < args.min_area:
                    continue
            filtered.append(d)

        # Draw results
        for d in filtered:
            # Outline
            draw_tag_outline(frame, d.corners)

            # ID at center
            center = tuple(d.center.astype(int))
            cv.circle(frame, center, 4, (0, 255, 255), -1, cv.LINE_AA)
            cv.putText(frame, f"id {d.tag_id}", (center[0] + 6, center[1] - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(frame, f"id {d.tag_id}", (center[0] + 6, center[1] - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv.LINE_AA)

            # Pose (R, t) in meters, tag frame: +Z is out of tag plane
            R = d.pose_R  # 3x3
            t = d.pose_t  # 3x1

            # Draw axes
            axis_len = args.tag_size * 0.5  # axes half the tag width
            draw_axes(frame, K, dist, R, t, axis_len)

            # Draw 3D box with bottom on the tag plane
            if not args.no_box:
                draw_box_on_tag(frame, K, dist, R, t, args.tag_size, args.box_height)

            # Telemetry text
            roll, pitch, yaw = rotation_to_euler_xyz(R)
            pos = t.reshape(-1)  # x, y, z in meters
            lines = [
                f"pos (m): x={pos[0]:+.3f}, y={pos[1]:+.3f}, z={pos[2]:+.3f}",
                f"rpy (deg): roll={np.degrees(roll):+.1f}, pitch={np.degrees(pitch):+.1f}, yaw={np.degrees(yaw):+.1f}",
            ]
            # Pose error if available
            if hasattr(d, "pose_err") and d.pose_err is not None:
                lines.append(f"pose err: {d.pose_err:.3f}")
            put_multiline_text(frame, lines, org=(10, 30 + 70 * (d.tag_id % 4)))
            
            # Output data for external consumption
            if args.output_data:
                data = {
                    "timestamp": time.time(),
                    "tag_id": int(d.tag_id),
                    "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                    "rotation": {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
                    "rotation_degrees": {"roll": float(np.degrees(roll)), "pitch": float(np.degrees(pitch)), "yaw": float(np.degrees(yaw))},
                    "quaternion": {
                        "w": float(np.cos(yaw/2) * np.cos(pitch/2) * np.cos(roll/2) + np.sin(yaw/2) * np.sin(pitch/2) * np.sin(roll/2)),
                        "x": float(np.cos(yaw/2) * np.cos(pitch/2) * np.sin(roll/2) - np.sin(yaw/2) * np.sin(pitch/2) * np.cos(roll/2)),
                        "y": float(np.cos(yaw/2) * np.sin(pitch/2) * np.cos(roll/2) + np.sin(yaw/2) * np.cos(pitch/2) * np.sin(roll/2)),
                        "z": float(np.sin(yaw/2) * np.cos(pitch/2) * np.cos(roll/2) - np.cos(yaw/2) * np.sin(pitch/2) * np.sin(roll/2))
                    }
                }
                if hasattr(d, "pose_err") and d.pose_err is not None:
                    data["pose_error"] = float(d.pose_err)
                
                output_line = json.dumps(data)
                if output_file:
                    output_file.write(output_line + "\n")
                    output_file.flush()
                else:
                    print(output_line, flush=True)

        put_multiline_text(frame, [f"FPS: {fps:.1f}", f"Detections: {len(detections)} ({len(filtered)})"], org=(10, frame_h - 40))

        cv.imshow("AprilTag Pose", frame)
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()
    
    # Close output file if opened
    if output_file:
        output_file.close()


if __name__ == "__main__":
    main()
