# LIDAR_SLAM_Project
Public repository for showcasing a coursework project (from my private repo) with all course identifying information scrubbed. I've uploaded the code here to serve as a code sample, and will expand this page in the future with a write up and figures of the SLAM results.

## Code Overview/Instructions

Code Files (6) :
- main.py (runs project parts 1 through 4)
- icp.py (library functions for ICP algorithm)
- constants.py (declares constants based on robot attributes)
- load_data.py (handles loading/processing sensor data)
- custom_utils.py (all helpers for part 1 through 4)
- given_utils.py (helpers not written by me, provided by course as part of assignment)

Setup Instructions:
The code was run with Python 3.9 in a venv with the packages listed in the requirements.txt. The hierarchy should be as follows. 

LIDAR_SLAM_Project/code/
LIDAR_SLAM_Project/data/SensorData
LIDAR_SLAM_Project/data/SavedEstimates


Running Instructions (main.py):
The dataset to run should be selected at the start of the main function, by modifying the config
variable dataset to the desired dataset #, 20 or 21. The ICP scan matching mode has not been fully optimized and runs slowly, but the computation will be skipped if the SavedEstimates files are included. The script will plot the trajectories for part 1 and 2 together (blue: odometry only, orange: icp-optimized), then the occupancy grid for part 3 with the part1/2 trajectories plotted over, and finally
the pose graph SLAM part 4 results with the part1/2 trajectories also plotted.

Summary & Steps:
1. Ensure you are using a Python 3.9 interpreter
2. Install appropriate packages using requirements.txt
3. Place the two data folders in the respective locations.
4. Run main.py (from inside the code directory so the script successfully finds the data directory) to see part 1/2/3/4. Select the dataset and leave all program flags as True to run all part.
