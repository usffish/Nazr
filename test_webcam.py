#!/usr/bin/env python3
"""
test_webcam.py — Test AuraGuard Vision Engine with your laptop webcam.

This is a standalone test script that runs the face recognition engine
using your built-in webcam instead of the RTMP stream from the glasses.

Usage:
    python test_webcam.py

Press 'q' to quit.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.vision.face_recognition_engine import run

if __name__ == "__main__":
    print("=" * 60)
    print("  AuraGuard Vision Engine — Webcam Test Mode")
    print("=" * 60)
    print()
    print("Using laptop webcam (camera index 0)")
    print("Press 'q' in the video window to quit")
    print()
    
    # Use camera index 0 for default laptop webcam
    # Pass 0 as integer (not string) for cv2.VideoCapture
    run(video_source=0)
