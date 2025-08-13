# N-Body Simulation Video Generator

This repository contains a Python script that simulates gravitational interactions between multiple celestial bodies and generates a video visualization of their motion. The implementation uses velocity Verlet integration for numerical stability.

![Sample Simulation](nbody3.mp4)

## Features

- Simulates N-body gravitational systems with customizable parameters
- Generates MP4 video output of the simulation
- Uses velocity Verlet integration for improved numerical stability
- Includes gravity softening to prevent singularities
- Configurable parameters including number of bodies, simulation time, time step size, and more
- Optional motion trails for visualizing trajectories
- Periodic boundary conditions

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- FFmpeg (for video encoding)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MengAiDev/Nbody.git
   cd nbody-simulation
   ```

2. Install the required Python packages:
   ```bash
   pip install numpy matplotlib
   ```

3. Install FFmpeg for video encoding:
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Usage

Basic usage:
```bash
python main.py --n_bodies 10 --t_end 20 --dt 0.01 --output nbody_simulation.mp4
```

### Command Line Arguments

| Argument | Description | Default |
|---------|-------------|---------|
| `--n_bodies` | Number of celestial bodies | 3 |
| `--t_end` | Total simulation time | 15.0 |
| `--dt` | Time step size | 0.01 |
| `--output` | Output video filename | nbody.mp4 |
| `--size` | Simulation area size | 10.0 |
| `--mass_min` | Minimum body mass | 0.1 |
| `--mass_max` | Maximum body mass | 1.0 |
| `--softening` | Gravity softening parameter | 0.2 |
| `--fps` | Frames per second for output video | 30 |
| `--trail` | Length of motion trails (0 to disable) | 30 |
| `--dpi` | Dots per inch for output video | 100 |

### Examples

Generate a simple 3-body simulation:
```bash
python main.py
```

Generate a 10-body simulation with longer duration:
```bash
python main.py --n_bodies 10 --t_end 30 --output nbody10.mp4
```

Create a high-resolution video with motion trails:
```bash
python main.py --n_bodies 15 --t_end 25 --dpi 200 --trail 50 --output high_res_nbody.mp4
```

## How It Works

The simulation uses the [velocity Verlet integration method](https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet), which is symplectic and provides better long-term stability for orbital mechanics than simple Euler integration.

The gravitational force between bodies is calculated using Newton's law of universal gravitation with a softening parameter to prevent numerical instabilities when bodies get too close to each other.

## Sample Outputs

- [3-body simulation (nbody3.mp4)](nbody3.mp4) - A basic simulation with 3 celestial bodies
- [10-body simulation with 1000 steps (nbody10_1000.mp4)](nbody10_1000.mp4) - A more complex simulation with 10 bodies running for 1000 time steps

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uses [Matplotlib](https://matplotlib.org/) for visualization
- Inspired by classic N-body problem simulations in astrophysics