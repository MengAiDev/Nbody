#!/usr/bin/env python3
"""
N-Body Simulation Video Generator with Full Trajectory Visualization

This script simulates gravitational interactions between multiple bodies and
generates a video of their motion. Includes option to display full trajectories.

Usage:
  python nbody.py --n_bodies 10 --t_end 20 --dt 0.01 --output nbody.mp4 --show_full_trajectory

Required dependencies:
  - numpy
  - matplotlib
  - ffmpeg (for video encoding)

Install dependencies with:
  pip install numpy matplotlib
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import argparse
import sys
import os

def compute_acceleration(positions, masses, G=1.0, softening=0.1):
    """
    Calculate acceleration for all bodies using Newtonian gravity
    
    Args:
        positions: Array of shape (n, 2) with current positions
        masses: Array of shape (n,) with masses
        G: Gravitational constant (default=1.0)
        softening: Softening parameter to prevent singularities
    
    Returns:
        accelerations: Array of shape (n, 2)
    """
    n = len(positions)
    accelerations = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            # Calculate distance vector with softening
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            r = np.sqrt(dx**2 + dy**2 + softening**2)
            
            # Calculate acceleration contribution
            acc = G * masses[j] / (r**3)
            accelerations[i, 0] += dx * acc
            accelerations[i, 1] += dy * acc
            
    return accelerations

def generate_colors(n_bodies, masses, color_scheme=None):
    """
    Generate color array for bodies based on specified color scheme
    
    Args:
        n_bodies: Number of bodies
        masses: Array of masses (for 'mass' color scheme)
        color_scheme: Color specification string (default=None)
        
    Returns:
        colors_array: List of colors for each body
    """
    if color_scheme is None:
        # Use default color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        return [color_cycle[i % len(color_cycle)] for i in range(n_bodies)]
    
    if color_scheme == 'mass':
        # Color by mass using viridis colormap
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=min(masses), vmax=max(masses))
        return [cmap(norm(m)) for m in masses]
    
    if color_scheme == 'random':
        # Generate random RGB colors
        return [np.random.rand(3,) for _ in range(n_bodies)]
    
    if ',' in color_scheme:
        # Comma-separated list of colors
        color_list = [c.strip() for c in color_scheme.split(',')]
        return [color_list[i % len(color_list)] for i in range(n_bodies)]
    
    if color_scheme in plt.colormaps():
        # Use specified colormap by index
        cmap = plt.get_cmap(color_scheme)
        return [cmap(i / (n_bodies - 1)) for i in range(n_bodies)]
    
    # Single color for all bodies
    return [color_scheme] * n_bodies

def main():
    parser = argparse.ArgumentParser(description='N-Body Simulation Video Generator')
    parser.add_argument('--n_bodies', type=int, default=3, 
                        help='Number of celestial bodies (default: 3)')
    parser.add_argument('--t_end', type=float, default=15.0,
                        help='Total simulation time (default: 15.0)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step size (default: 0.01)')
    parser.add_argument('--output', type=str, default='nbody.mp4',
                        help='Output video filename (default: nbody.mp4)')
    parser.add_argument('--size', type=float, default=10.0,
                        help='Simulation area size (default: 10.0)')
    parser.add_argument('--mass_min', type=float, default=0.1,
                        help='Minimum body mass (default: 0.1)')
    parser.add_argument('--mass_max', type=float, default=1.0,
                        help='Maximum body mass (default: 1.0)')
    parser.add_argument('--softening', type=float, default=0.2,
                        help='Gravity softening parameter (default: 0.2)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for output video (default: 30)')
    parser.add_argument('--trail', type=int, default=30,
                        help='Length of motion trails (0 to disable, default: 30)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='Dots per inch for output video (default: 100)')
    parser.add_argument('--colors', type=str, default=None,
                        help="Color scheme for bodies. Options:\n"
                             "- Single color name (e.g., 'red')\n"
                             "- Comma-separated list (e.g., 'red,green,blue')\n"
                             "- Colormap name (e.g., 'viridis', 'plasma')\n"
                             "- 'mass' to color by mass\n"
                             "- 'random' for random colors\n"
                             "Default: matplotlib color cycle")
    parser.add_argument('--show_full_trajectory', action='store_true',
                        help='Show full trajectory for each body from start to current frame')
    parser.add_argument('--trail_alpha', type=float, default=0.7,
                        help='Alpha (transparency) for motion trails (default: 0.7)')
    parser.add_argument('--trail_fade', action='store_true',
                        help='Fade out older parts of the trajectory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.n_bodies <= 0:
        parser.error("n_bodies must be positive")
    if args.t_end <= 0:
        parser.error("t_end must be positive")
    if args.dt <= 0:
        parser.error("dt must be positive")
    if args.mass_min <= 0 or args.mass_max <= 0:
        parser.error("Mass values must be positive")
    if args.mass_min > args.mass_max:
        parser.error("mass_min cannot be greater than mass_max")
    if args.trail_alpha <= 0 or args.trail_alpha > 1:
        parser.error("trail_alpha must be between 0 and 1")
    
    print(f"Starting N-body simulation with {args.n_bodies} bodies...")
    print(f"Parameters: t_end={args.t_end}, dt={args.dt}, size={args.size}")
    print(f"Output: {args.output} ({args.fps} FPS, {args.dpi} DPI)")
    
    # Calculate number of steps
    n_steps = int(args.t_end / args.dt)
    if n_steps <= 0:
        parser.error("Insufficient steps (t_end/dt too small)")
    
    # Initialize bodies
    np.random.seed(42)  # For reproducibility
    masses = np.random.uniform(args.mass_min, args.mass_max, args.n_bodies)
    
    # Random positions within simulation area
    positions = np.random.uniform(-args.size, args.size, (args.n_bodies, 2))
    
    # Random velocities (scaled by size for stability)
    velocities = np.random.uniform(-0.5, 0.5, (args.n_bodies, 2)) * np.sqrt(args.size)
    
    # Generate body colors
    body_colors = generate_colors(args.n_bodies, masses, args.colors)
    print(f"Using color scheme: {args.colors or 'default cycle'}")
    
    # Initialize storage for trajectories
    trajectories = np.zeros((n_steps, args.n_bodies, 2))
    
    # Compute initial acceleration
    accelerations = compute_acceleration(positions, masses, softening=args.softening)
    
    # Velocity Verlet integration
    for step in range(n_steps):
        # Store current positions
        trajectories[step] = positions.copy()
        
        # Update positions
        positions += velocities * args.dt + 0.5 * accelerations * args.dt**2
        
        # Calculate new acceleration
        new_accelerations = compute_acceleration(positions, masses, softening=args.softening)
        
        # Update velocities
        velocities += 0.5 * (accelerations + new_accelerations) * args.dt
        
        # Update acceleration for next step
        accelerations = new_accelerations
        
        # Periodic boundary conditions (optional)
        positions = np.where(np.abs(positions) > args.size, 
                             -np.sign(positions) * args.size, 
                             positions)
        
        # Progress report
        if step % max(1, n_steps//10) == 0:
            print(f"Simulation progress: {100 * step / n_steps:.1f}%")
    
    print("Simulation completed. Generating animation...")
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=args.dpi)
    ax.set_xlim(-args.size, args.size)
    ax.set_ylim(-args.size, args.size)
    ax.set_aspect('equal')
    ax.set_title(f'N-Body Simulation (N={args.n_bodies})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create scatter plot for bodies
    scatter = ax.scatter(
        trajectories[0, :, 0],  # 使用第一帧的X坐标
        trajectories[0, :, 1],  # 使用第一帧的Y坐标
        s=30, 
        c=body_colors, 
        zorder=3
    )
    
    # Create trails with body colors
    trails = []
    for i in range(args.n_bodies):
        # Use body color with reduced alpha for trails
        rgba = mcolors.to_rgba(body_colors[i], alpha=args.trail_alpha)
        trail, = ax.plot([], [], '-', lw=1, color=rgba)
        trails.append(trail)
    
    # Initialize full trajectory visualization
    full_trajectories = []
    if args.show_full_trajectory:
        # Create colormap for time-based coloring
        time_cmap = plt.get_cmap('viridis')
        
        for i in range(args.n_bodies):
            # Create empty line collection for each body
            segments = np.zeros((0, 2, 2))  # Empty segments array
            lc = LineCollection(segments, cmap=time_cmap, alpha=args.trail_alpha)
            lc.set_array(np.array([]))  # Empty time array
            full_trajectories.append(ax.add_collection(lc))
    
    # Animation update function
    def update(frame):
        # Update main positions
        scatter.set_offsets(trajectories[frame])
        
        # Update trails
        if args.trail > 0:
            start = max(0, frame - args.trail)
            for i, trail in enumerate(trails):
                trail.set_data(
                    trajectories[start:frame+1, i, 0],
                    trajectories[start:frame+1, i, 1]
                )
        
        # Update full trajectories
        if args.show_full_trajectory and frame > 0:
            for i in range(args.n_bodies):
                # Create segments for each step
                points = trajectories[:frame+1, i, :]
                segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
                
                # Create time array for coloring (normalized from 0 to 1)
                times = np.linspace(0, 1, frame)
                
                # Update line collection
                full_trajectories[i].set_segments(segments)
                full_trajectories[i].set_array(times)
                
                # Apply fade effect if requested
                if args.trail_fade:
                    alpha_values = np.linspace(0.1, args.trail_alpha, frame)
                    full_trajectories[i].set_alpha(alpha_values)
        
        # Combine all artists to return
        artists = [scatter] + trails
        if args.show_full_trajectory:
            artists += full_trajectories
        
        return artists
    
    # Create animation
    ani = FuncAnimation(
        fig, 
        update, 
        frames=min(n_steps, 1000) if args.show_full_trajectory else n_steps,  # Limit frames for performance
        interval=1000/args.fps,
        blit=True
    )
    
    # Save animation
    try:
        writer = matplotlib.animation.FFMpegWriter(
            fps=args.fps,
            metadata=dict(artist='N-Body Simulation'),
            bitrate=1800
        )
        ani.save(args.output, writer=writer)
        print(f"\nVideo saved to: {os.path.abspath(args.output)}")
        print(f"Total frames: {n_steps}, Duration: {n_steps/args.fps:.1f} seconds")
    except RuntimeError as e:
        if "ffmpeg" in str(e).lower():
            print("\nERROR: FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
            print("  On Ubuntu: sudo apt install ffmpeg")
            print("  On macOS: brew install ffmpeg")
            print("  On Windows: https://ffmpeg.org/download.html")
        else:
            print(f"\nERROR saving video: {e}")
        sys.exit(1)
    
    plt.close(fig)

if __name__ == "__main__":
    main()