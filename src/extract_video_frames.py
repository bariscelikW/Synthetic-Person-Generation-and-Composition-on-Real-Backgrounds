# extract_video_frames.py
import cv2
import os
from pathlib import Path
import argparse

"""
Example Usage:

python extract_video_frames.py input.mp4 -o output_frames -f 0.2    

"""


def extract_frames(
    video_path: str, output_folder: str, target_fps: float = 1.0, max_frames: int = None
):
    """
    Extract frames from video at specified FPS rate

    Args:
        video_path: Path to the input video file
        output_folder: Where to save the extracted frames
        target_fps: How many frames per second you want to keep (e.g. 1 = 1 fps, 0.5 = every 2 seconds)
        max_frames: Optional - stop after extracting this many frames
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / original_fps if original_fps > 0 else 0

    print("Video Info:")
    print(f"  • Original FPS:     {original_fps:.2f}")
    print(f"  • Total frames:     {frame_count:,d}")
    print(
        f"  • Duration:         {duration_sec:.1f} seconds ≈ {duration_sec / 60:.1f} minutes"
    )
    print(f"  • Target extraction: {target_fps} fps")

    if target_fps <= 0:
        print("Target FPS must be > 0")
        cap.release()
        return

    # Calculate how often we should save a frame
    frame_interval = max(1, round(original_fps / target_fps))
    print(
        f"  → Saving every {frame_interval} frame(s) (≈ {original_fps / frame_interval:.2f} fps)"
    )

    saved_count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            output_path = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            saved_count += 1

            if saved_count % 100 == 0:
                print(f"Saved {saved_count:,} frames...")

            if max_frames is not None and saved_count >= max_frames:
                print(f"Reached maximum requested frames ({max_frames})")
                break

        frame_idx += 1

    cap.release()

    print("\nExtraction finished:")
    print(f"  • Processed frames: {frame_idx:,d}")
    print(f"  • Saved frames:     {saved_count:,d}")
    print(f"  • Output folder:    {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from CCTV/security video at chosen FPS"
    )
    parser.add_argument(
        "video", type=str, help="Path to video file (mp4, avi, mov, etc.)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="frames",
        help="Output folder for frames (default: ./frames)",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=float,
        default=1.0,
        help="Target FPS to extract (default: 1.0 = 1 frame per second)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of frames to extract (optional)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    extract_frames(
        video_path=args.video,
        output_folder=args.output,
        target_fps=args.fps,
        max_frames=args.max,
    )


if __name__ == "__main__":
    main()
