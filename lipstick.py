import cv2
import numpy as np
import math
import random
from syphon import SyphonMetalServer
import Metal  # Add Metal import
from collections import deque
import gc  # Add garbage collector
import subprocess  # Add for camera listing


# List available cameras
def list_cameras():
    try:
        result = subprocess.run(
            ['system_profiler', 'SPCameraDataType'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        devices = [line.strip()
                   for line in output.split('\n') if 'Model ID' in line]
        return devices
    except Exception as e:
        print(f"Error listing cameras: {e}")
        return []


camera_devices = list_cameras()
print("Choose a camera: ")
for i, device in enumerate(camera_devices):
    print(f"{i}: {device}")
camera_choice = int(input("Enter the camera number: "))
camera_device = list(camera_devices)[camera_choice]
print(f"Using camera: {camera_device}")

# Add resolution selection with error handling
print("\nChoose resolution:")
resolutions = [
    {"name": "Low (640x480) - faster", "width": 640, "height": 480},
    {"name": "HD (720x405) - 16:9", "width": 720, "height": 405},
    {"name": "HD (720x450) - 16:10", "width": 720, "height": 450},
    {"name": "HD (1280x720) - 16:9", "width": 1280, "height": 720},
    {"name": "HD (1280x800) - 16:10", "width": 1280, "height": 800},
    {"name": "Full HD (1920x1080) - 16:9", "width": 1920, "height": 1080}
]

for i, res in enumerate(resolutions):
    print(f"{i}: {res['name']}")

# Add error handling for resolution selection
try:
    user_input = input(
        "Enter resolution number (or press Enter for default): ")
    resolution_choice = int(user_input) if user_input.strip() else 0

    # Validate resolution choice is within range
    if resolution_choice < 0 or resolution_choice >= len(resolutions):
        print(f"Invalid resolution. Using default: {resolutions[0]['name']}")
        resolution_choice = 0
    else:
        print(f"Using resolution: {resolutions[resolution_choice]['name']}")
except ValueError:
    print("Invalid input. Using default resolution (640x480)")
    resolution_choice = 0

frame_width = resolutions[resolution_choice]["width"]
frame_height = resolutions[resolution_choice]["height"]

# Start video capture with selected camera
cap = cv2.VideoCapture(camera_choice)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Verify actual capture resolution (camera might not support requested resolution)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual capture resolution: {actual_width}x{actual_height}")

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Try to read first frame with retry
max_attempts = 5
attempt = 0
while attempt < max_attempts:
    ret, frame = cap.read()
    if ret and frame is not None:
        break
    print(f"Failed to read frame, attempt {attempt + 1} of {max_attempts}")
    attempt += 1
    cv2.waitKey(1000)  # Wait 1 second before retry

if frame is None:
    print("Error: Could not read frame from camera")
    cap.release()
    exit()

height, width = frame.shape[:2]

# Create transparent overlay for lines
overlay = np.zeros((height, width, 3), dtype=np.uint8)

# Define red color range in HSV (for lipstick)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

draw_line = False  # Flag to control line drawing
show_only_lines = False
fade_alpha = 1.0  # For fade effect
line_points = []  # Store endpoints of lines
mesh_mode = False
point_velocities = []  # Store velocities for each point

# Add after other global variables
line_velocities = []  # Store velocities for line endpoints
current_contour = None  # Store the current contour for boundary checking

# Add these variables after other global variables
pause_video = False
paused_frame = None

# Add after other global variables
transition_alpha = 1.0  # For transition between line and mesh modes
TRANSITION_SPEED = 0.3  # Speed of transition

# Add after frame dimensions
MASK_WINDOW_NAME = 'Mask Monitor'
cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(MASK_WINDOW_NAME, 320, 240)  # Smaller size for monitoring

# Add after other global variables
manual_move = False  # Flag for manual movement mode
MANUAL_MOVE_SPEED = 3  # Speed for individual node movement
current_direction = None  # Store current movement direction
DIRECTION_CHANGE_PROB = 0.45  # Higher probability for individual direction changes
MIN_POINTS_IN_CONTOUR = 0.3  # Allow movement if 30% of points stay in contour
NODE_MOVE_RANGE = 50  # Increased from 30
BOUNDARY_TOLERANCE = 300  # Distance allowed outside contour
MAX_SPEED = 3.0  # Increased maximum speed for points

# Add after other global variables
LINE_THICKNESS = 2  # Increased line thickness

# Add after other global variables
mouse_pos = (0, 0)  # Store current mouse position
MOUSE_FOLLOW_SPEED = 0.1  # Speed at which points follow mouse
mouse_follow_mode = False  # Toggle for mouse following

# Add after other global variables
RANDOM_MOVE_SPEED = 5.0  # Speed for random movement
TARGET_CHANGE_INTERVAL = 60  # Frames before changing random target
target_positions = []  # Store target positions for each point
frame_counter = 0  # Counter for changing targets

# Add after video capture initialization
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 480

cv2.namedWindow('Lipstick Lines', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Lipstick Lines', WINDOW_WIDTH,
                 WINDOW_HEIGHT)  # Set main window size

# Add these constants after other global variables
EXPLOSION_SPEED = 8.0  # Reduced initial speed for explosion
EXPLOSION_DECAY = 0.95  # Faster speed decay factor
EDGE_BOUNCE_FACTOR = 2.5  # Reduced energy retention on bounce
MIN_EXPLOSION_SPEED = 1.0  # Minimum speed after decay

# Add these constants after other global variables
EXPLOSION_MESH_DIST = 300  # Maximum distance for mesh connections during explosion
MIN_MESH_DIST = 30  # Minimum distance for mesh connections
MESH_PROBABILITY = 0.6  # Probability of creating mesh connections

# Add these constants after other global variables
INITIAL_MESH_SPEED = 0.05  # Reduced initial speed
MESH_ACCELERATION = 0.0005  # Reduced acceleration
MAX_MESH_SPEED = 0.3  # Reduced max speed
MESH_MIN_DIST = 20  # Minimum distance for mesh connections
MESH_MAX_DIST = 150  # Maximum distance for initial mesh connections
MESH_CONNECT_PROB = 0.8  # Probability of creating mesh connections
current_mesh_speed = INITIAL_MESH_SPEED  # Current speed for mesh transitions

# Add constants for performance tuning
FRAME_SKIP = 2  # Process every nth frame
POINT_BATCH_SIZE = 50  # Process points in batches
CACHE_SIZE = 5  # Number of frames to cache for motion detection
MAX_POINTS = 200  # Limit total number of points for performance

# Add after other global variables
frame_buffer = deque(maxlen=CACHE_SIZE)
frame_count = 0
cached_contour = None
cached_mask = None

# Add these constants for memory management
MAX_STORED_FRAMES = 5
CLEANUP_INTERVAL = 100
frame_counter = 0
last_cleanup = 0

# Add memory management class


class MemoryManager:
    def __init__(self):
        self.frame_buffer = deque(maxlen=MAX_STORED_FRAMES)

    def add_frame(self, frame):
        if len(self.frame_buffer) == MAX_STORED_FRAMES:
            self.frame_buffer.popleft()  # Remove oldest frame
        self.frame_buffer.append(frame)

    def clear_buffer(self):
        self.frame_buffer.clear()
        gc.collect()  # Force garbage collection


memory_manager = MemoryManager()


def create_mesh(points, contour, max_dist=50):
    """Optimized mesh creation with numpy operations"""
    if len(points) < 2:
        return []

    # Convert points to numpy array for vectorized operations
    points = np.array(points, dtype=np.int32)

    # Calculate all pairwise distances at once
    diff = points[:, np.newaxis] - points
    distances = np.sqrt(np.sum(diff * diff, axis=2))

    # Create mask for valid distances
    valid_mask = (distances > 0) & (distances < max_dist)

    # Get indices of valid pairs
    valid_pairs = np.where(valid_mask)

    # Convert back to list of tuples for compatibility
    mesh_lines = [
        (tuple(points[i]), tuple(points[j]))
        for i, j in zip(*valid_pairs)
        if i < j  # Avoid duplicates
    ]

    return mesh_lines


def update_point_positions(points, velocities, contour):
    """Vectorized point position updates"""
    if not points:
        return

    points_array = np.array(points)
    velocities_array = np.array(velocities)

    # Update positions
    new_positions = points_array + velocities_array

    # Check boundary conditions vectorized
    in_bounds = (new_positions[:, 0] >= 0) & (new_positions[:, 0] < width) & \
                (new_positions[:, 1] >= 0) & (new_positions[:, 1] < height)

    # Update only valid positions
    points_array[in_bounds] = new_positions[in_bounds]

    # Update velocities with bounds
    velocities_array = np.clip(velocities_array + np.random.uniform(-0.1, 0.1, velocities_array.shape),
                               -MAX_SPEED, MAX_SPEED)

    return points_array.tolist(), velocities_array.tolist()


def cleanup_resources():
    """Clean up resources and force garbage collection"""
    global overlay, frame_buffer, cached_mask, cached_contour

    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    memory_manager.clear_buffer()
    cached_mask = None
    cached_contour = None
    gc.collect()


def find_shorter_line(contour, original_left, original_right, ratio=0.5):
    """Find shorter line within contour by moving endpoints inward"""
    center_x = (original_left[0] + original_right[0]) // 2
    center_y = (original_left[1] + original_right[1]) // 2

    # Calculate shorter line endpoints
    vec_x = (original_right[0] - original_left[0]) * ratio
    vec_y = (original_right[1] - original_left[1]) * ratio

    new_left = (int(center_x - vec_x/2), int(center_y - vec_y/2))
    new_right = (int(center_x + vec_x/2), int(center_y + vec_y/2))

    return new_left, new_right


def create_texture_descriptor(width, height):
    """Create a Metal texture descriptor with the correct attributes"""
    descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
        Metal.MTLPixelFormatRGBA8Unorm,
        width,
        height,
        False
    )
    descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
    return descriptor


# Modify point_in_contour to bypass contour checks when contour is None
def point_in_contour(point, contour):
    """Check if point is inside contour; if no contour, return True"""
    if contour is None:
        return True
    pt = (float(point[0]), float(point[1]))
    return cv2.pointPolygonTest(contour, pt, False) >= 0


def update_line_positions(lines, velocities, contour):
    """Update line positions while keeping them inside contour"""
    if not contour.any():
        return

    for i in range(0, len(lines), 2):
        if i+1 >= len(lines):
            break

        for j, point_idx in enumerate([i, i+1]):
            x, y = lines[point_idx]
            vx, vy = velocities[point_idx]

            # Update position
            new_x = x + vx
            new_y = y + vy
            new_point = (int(new_x), int(new_y))

            # Check if new position is inside contour
            if point_in_contour(new_point, contour):
                lines[point_idx] = new_point
            else:
                # Reverse velocity if hitting boundary
                velocities[point_idx] = (-vx * 0.8, -vy * 0.8)

            # Add small random acceleration
            velocities[point_idx] = (
                vx + random.uniform(-0.1, 0.1),
                vy + random.uniform(-0.1, 0.1)
            )

            # Limit velocity
            max_speed = 2.0
            vx, vy = velocities[point_idx]
            speed = math.sqrt(vx*vx + vy*vy)
            if speed > max_speed:
                velocities[point_idx] = (
                    vx * max_speed / speed,
                    vy * max_speed / speed
                )


# Increased max_dist from 120
def create_static_mesh(line_points, contour, max_dist=MESH_MAX_DIST, min_dist=MESH_MIN_DIST, explosion_mode=False, max_neighbors=3):
    """Create mesh by connecting line endpoints with limited connections per point for a natural appearance"""
    mesh_lines = []
    if len(line_points) < 4:
        return mesh_lines

    # Convert all points to integer tuples
    line_points = [(int(p[0]), int(p[1])) for p in line_points]

    # Always add original line pairs first and track connection counts
    connections = {i: 0 for i in range(len(line_points))}
    for i in range(0, len(line_points), 2):
        if i+1 < len(line_points):
            mesh_lines.append((line_points[i], line_points[i+1]))
            connections[i] += 1
            connections[i+1] += 1

    # Use explosion or normal parameters
    if explosion_mode:
        max_dist = EXPLOSION_MESH_DIST
        min_dist = MIN_MESH_DIST
        connect_prob = MESH_PROBABILITY
    else:
        connect_prob = MESH_CONNECT_PROB

    # Connect between all points while limiting maximum neighbors per point
    for i in range(len(line_points)):
        if connections[i] >= max_neighbors:
            continue
        for j in range(i+1, len(line_points)):
            if connections[j] >= max_neighbors:
                continue
            p1, p2 = line_points[i], line_points[j]
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            if min_dist <= dist <= max_dist and random.random() < connect_prob:
                mesh_lines.append((p1, p2))
                connections[i] += 1
                connections[j] += 1
                if connections[i] >= max_neighbors:
                    break

    return mesh_lines


# Modify update_mesh_points function to prevent overflow errors
def update_mesh_points(points, velocities, contour):
    """Update mesh points with gradual acceleration and overflow protection"""
    if not points:
        return

    global mouse_follow_mode, current_mesh_speed

    # Only accelerate if not in explosion mode
    if not mouse_follow_mode:
        current_mesh_speed = min(
            current_mesh_speed + MESH_ACCELERATION, MAX_MESH_SPEED)

    # Define max safe values to prevent overflow
    MAX_SAFE_VELOCITY = 1000.0  # Limit maximum velocity
    MAX_SAFE_COORDINATE = 10000.0  # Limit maximum coordinate value

    for i in range(len(points)):
        if i >= len(velocities):
            continue

        x, y = points[i]

        # Ensure x and y are within reasonable range
        x = max(-MAX_SAFE_COORDINATE, min(MAX_SAFE_COORDINATE, x))
        y = max(-MAX_SAFE_COORDINATE, min(MAX_SAFE_COORDINATE, y))

        # Convert to regular float to prevent overflow
        vx = float(velocities[i][0])
        vy = float(velocities[i][1])

        # Ensure velocities are within reasonable range
        vx = max(-MAX_SAFE_VELOCITY, min(MAX_SAFE_VELOCITY, vx))
        vy = max(-MAX_SAFE_VELOCITY, min(MAX_SAFE_VELOCITY, vy))

        # Update position with clamped values
        new_x = x + vx
        new_y = y + vy

        if mouse_follow_mode:
            # Handle explosion mode with overflow protection
            bounce_occurred = False

            if new_x <= 0 or new_x >= width:
                # Apply bounce with safety checks
                # Limit bounce factor
                bounce_factor = min(EDGE_BOUNCE_FACTOR, 3.0)
                vx = -vx * bounce_factor
                bounce_occurred = True

            if new_y <= 0 or new_y >= height:
                # Apply bounce with safety checks
                # Limit bounce factor
                bounce_factor = min(EDGE_BOUNCE_FACTOR, 3.0)
                vy = -vy * bounce_factor
                bounce_occurred = True

            # Add random spin on bounce with limits
            if bounce_occurred:
                vx += random.uniform(-0.5, 0.5)  # Reduced random factor
                vy += random.uniform(-0.5, 0.5)  # Reduced random factor

            # Apply speed decay with safety checks
            try:
                speed = math.sqrt(vx*vx + vy*vy)
                # Add upper limit to speed to prevent overflow
                speed = min(speed, MAX_SAFE_VELOCITY)

                if speed > MIN_EXPLOSION_SPEED:
                    vx *= EXPLOSION_DECAY
                    vy *= EXPLOSION_DECAY
            except (OverflowError, ValueError):
                # If calculation overflows, reset to safe values
                vx = random.uniform(-1, 1) * EXPLOSION_SPEED / 2
                vy = random.uniform(-1, 1) * EXPLOSION_SPEED / 2

            # Ensure final position is within valid screen coordinates
            nx = max(0, min(width-1, int(new_x)))
            ny = max(0, min(height-1, int(new_y)))

            points[i] = (nx, ny)
            velocities[i] = (vx, vy)
        else:
            # Normal mesh movement with overflow protection
            if contour is not None:
                # Add random acceleration with limits
                vx += random.uniform(-0.1, 0.1) * current_mesh_speed
                vy += random.uniform(-0.1, 0.1) * current_mesh_speed

                # Check for NaN or infinity
                if math.isnan(vx) or math.isinf(vx) or math.isnan(vy) or math.isinf(vy):
                    vx = random.uniform(-0.1, 0.1) * current_mesh_speed
                    vy = random.uniform(-0.1, 0.1) * current_mesh_speed

                # Apply speed limit with safety check
                try:
                    speed = math.sqrt(vx*vx + vy*vy)
                    if speed > current_mesh_speed and speed > 0:
                        scale = current_mesh_speed / speed
                        vx *= scale
                        vy *= scale
                except (OverflowError, ZeroDivisionError):
                    vx = random.uniform(-0.1, 0.1) * current_mesh_speed
                    vy = random.uniform(-0.1, 0.1) * current_mesh_speed

                # Ensure new point is valid
                nx = max(0, min(width-1, int(new_x)))
                ny = max(0, min(height-1, int(new_y)))
                new_point = (nx, ny)

                try:
                    # Check point within contour with error handling
                    if cv2.pointPolygonTest(contour, (float(nx), float(ny)), True) > -BOUNDARY_TOLERANCE:
                        points[i] = new_point
                        velocities[i] = (vx, vy)
                    else:
                        velocities[i] = (-vx * 0.8, -vy * 0.8)
                except:
                    # On any error, use safe fallback
                    points[i] = (nx, ny)
                    velocities[i] = (random.uniform(-0.1, 0.1),
                                     random.uniform(-0.1, 0.1))


# Also fix create_explosion_velocities function with better bounds
def create_explosion_velocities(points, center=None):
    """Create velocities for explosive movement with safety checks"""
    velocities = []
    if center is None:
        # Use average position of all points as center
        center = (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points)
        )

    # Use reduced explosion speed for safety
    safe_explosion_speed = min(EXPLOSION_SPEED, 8.0)

    for point in points:
        try:
            # Calculate direction from center
            dx = point[0] - center[0]
            dy = point[1] - center[1]

            # Normalize and add some randomness with safety
            dist = math.sqrt(dx*dx + dy*dy) or 1.0  # Avoid division by zero

            # Ensure direction is valid (not NaN)
            if math.isnan(dx/dist) or math.isnan(dy/dist):
                dx = random.uniform(-1, 1)
                dy = random.uniform(-1, 1)
            else:
                dx = dx/dist + random.uniform(-0.2, 0.2)  # Reduced randomness
                dy = dy/dist + random.uniform(-0.2, 0.2)  # Reduced randomness

            velocities.append((dx * safe_explosion_speed,
                              dy * safe_explosion_speed))
        except:
            # On any error, add a random safe velocity
            velocities.append((random.uniform(-2, 2), random.uniform(-2, 2)))

    return velocities


# Initialize Metal device and Syphon server after video capture
mtl_device = Metal.MTLCreateSystemDefaultDevice()
syphon_server = SyphonMetalServer("Lipstick Effect", device=mtl_device)


# Add after window creation
cv2.namedWindow('Lipstick Lines')

# Add this new function before the main loop


def update_manual_movement(points, velocities, direction, contour):
    """Update point positions for manual movement mode"""
    if not points or not direction:
        return

    dx, dy = direction
    points_in_contour = 0

    for i in range(len(points)):
        x, y = points[i]
        new_x = x + dx * MANUAL_MOVE_SPEED
        new_y = y + dy * MANUAL_MOVE_SPEED
        new_point = (int(new_x), int(new_y))

        # Check if new position is within bounds
        if contour is not None and cv2.pointPolygonTest(contour, (float(new_x), float(new_y)), True) > -BOUNDARY_TOLERANCE:
            points[i] = new_point
            points_in_contour += 1
            velocities[i] = (dx * MANUAL_MOVE_SPEED, dy * MANUAL_MOVE_SPEED)

    # Return True if enough points stayed within bounds
    return points_in_contour >= len(points) * MIN_POINTS_IN_CONTOUR


def optimize_contour_operations(mask):
    """Optimize contour detection with caching"""
    global cached_contour, cached_mask

    if cached_mask is not None and np.array_equal(mask, cached_mask):
        return cached_contour

    # Use more efficient contour finding mode
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cached_contour = max(contours, key=cv2.contourArea)
        cached_mask = mask.copy()
        return cached_contour
    return None


def vectorized_update_points(points, velocities, contour):
    """Vectorized version of point updates"""
    if not len(points):
        return

    # Convert to numpy arrays for vectorization
    points_array = np.array(points)
    velocities_array = np.array(velocities)

    # Update all points at once
    new_positions = points_array + velocities_array

    # Vectorized boundary check
    in_bounds = (new_positions[:, 0] >= 0) & (new_positions[:, 0] < width) & \
                (new_positions[:, 1] >= 0) & (new_positions[:, 1] < height)

    # Update only valid positions
    points_array[in_bounds] = new_positions[in_bounds]

    # Add random acceleration vectorized
    random_acc = np.random.uniform(-0.1, 0.1,
                                   velocities_array.shape) * current_mesh_speed
    velocities_array += random_acc

    # Speed limiting vectorized
    speeds = np.linalg.norm(velocities_array, axis=1)
    over_speed = speeds > current_mesh_speed
    if np.any(over_speed):
        scale_factors = np.where(over_speed, current_mesh_speed / speeds, 1)
        velocities_array *= scale_factors[:, np.newaxis]

    return points_array.tolist(), velocities_array.tolist()


def batch_process_mesh_lines(points, max_dist):
    """Process mesh lines in batches for better performance"""
    mesh_lines = []
    points = np.array(points)

    for i in range(0, len(points), POINT_BATCH_SIZE):
        batch = points[i:i + POINT_BATCH_SIZE]
        # Compute pairwise distances using vectorization
        distances = np.sqrt(((batch[:, None] - points) ** 2).sum(axis=2))
        valid_pairs = np.where((distances > 0) & (distances <= max_dist))

        for idx1, idx2 in zip(*valid_pairs):
            if idx1 + i < idx2:  # Avoid duplicates
                mesh_lines.append((tuple(batch[idx1]), tuple(points[idx2])))

    return mesh_lines


while True:
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:  # Skip frames for performance
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if not pause_video:
        ret, frame = cap.read()
        if not ret:
            break
        memory_manager.add_frame(frame.copy())
    else:
        frame = paused_frame.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for red color (both ranges)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Print debug info
        if draw_line:
            print(f"Contour area: {area}")

        # Lower the area threshold to 50
        if draw_line and area > 50:
            leftmost = tuple(
                largest_contour[largest_contour[:, :, 0].argmin()][0])
            rightmost = tuple(
                largest_contour[largest_contour[:, :, 0].argmax()][0])

            dx = rightmost[0] - leftmost[0]
            dy = rightmost[1] - leftmost[1]
            angle = abs(math.degrees(math.atan2(dy, dx)))

            print(f"Line angle: {angle}")

            # Modified angle handling
            if 15 <= angle <= 165:
                cv2.line(overlay, leftmost, rightmost,
                         (0, 0, 255), LINE_THICKNESS, cv2.LINE_AA)
                line_endpoints = [leftmost, rightmost]
            else:
                # Draw shorter line for out-of-range angles
                short_left, short_right = find_shorter_line(
                    largest_contour, leftmost, rightmost, 0.5)
                cv2.line(overlay, short_left, short_right,
                         (0, 0, 255), LINE_THICKNESS, cv2.LINE_AA)
                line_endpoints = [short_left, short_right]
                print("Drawing shorter line")

            # Store line endpoints
            line_points.extend(line_endpoints)
            # Add initial velocities for new line endpoints
            line_velocities.extend([
                (random.uniform(-1, 1), random.uniform(-1, 1))
                for _ in range(2)
            ])
            draw_line = False
    elif draw_line:
        print("No contours found!")

    # After contour processing, update line positions
    if line_points and not mesh_mode:
        update_line_positions(line_points, line_velocities, largest_contour)

        # Redraw all lines in overlay
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, len(line_points), 2):
            if i+1 < len(line_points):
                cv2.line(
                    overlay, line_points[i], line_points[i+1], (0, 0, 255), LINE_THICKNESS, cv2.LINE_AA)

    # Visual feedback when 'l' is pressed
    if draw_line:
        cv2.putText(frame, "Ready to draw!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw mesh or normal overlay
    if mesh_mode:
        mesh_frame = np.zeros((height, width, 3), dtype=np.uint8)
        line_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Instead of obtaining contours for mesh mode, pass None
        update_mesh_points(line_points, point_velocities, None)

        # Draw all connections
        mesh_lines = create_static_mesh(
            line_points,
            None,  # no contour reference
            explosion_mode=mouse_follow_mode
        )

        # Draw original lines first
        for i in range(0, len(line_points), 2):
            if i+1 < len(line_points):
                cv2.line(
                    line_frame,
                    line_points[i],
                    line_points[i+1],
                    (0, 0, 255),
                    LINE_THICKNESS,
                    cv2.LINE_AA
                )

        # Draw mesh connections
        for p1, p2 in mesh_lines:
            try:
                cv2.line(
                    mesh_frame,
                    (int(p1[0]), int(p1[1])),
                    (int(p2[0]), int(p2[1])),
                    (0, 0, 255),
                    # Slightly thinner lines for mesh
                    max(1, LINE_THICKNESS-1),
                    cv2.LINE_AA
                )
            except Exception as e:
                continue

        # Blend frames with transition
        blended = cv2.addWeighted(
            line_frame,
            1.0 - transition_alpha,
            mesh_frame,
            transition_alpha,
            0
        )
        result = blended

        # Update transition alpha more slowly
        if transition_alpha < 1.0:
            transition_alpha = min(1.0, transition_alpha + 0.02)
    else:
        # Use black background when showing only lines
        if show_only_lines:
            result = overlay
        else:
            result = cv2.addWeighted(frame, fade_alpha, overlay, 1, 0)
        transition_alpha = 0.0

    # Show results with modified window handling
    cv2.imshow('Lipstick Lines', result)
    cv2.imshow(MASK_WINDOW_NAME, mask)

    # Add Syphon server publishing
    if syphon_server.has_clients:
        try:
            # Convert result to RGBA and flip vertically
            if result.shape[2] == 3:
                rgba = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
            else:
                rgba = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)

            rgba = cv2.flip(rgba, 0)  # Flip vertically for Syphon
            h, w = rgba.shape[:2]

            # Create Metal texture
            descriptor = create_texture_descriptor(w, h)
            texture = mtl_device.newTextureWithDescriptor_(descriptor)

            # Copy frame data to texture
            rgba = np.ascontiguousarray(rgba)
            region = Metal.MTLRegionMake2D(0, 0, w, h)
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                region, 0, rgba.tobytes(), w * 4)

            # Publish texture
            syphon_server.publish_frame_texture(texture)

            # Ensure proper cleanup of Metal resources
            texture.setPurgeableState_(Metal.MTLPurgeableStateEmpty)
            del texture

        except Exception as e:
            print(f"Failed to publish frame: {e}")
            import traceback
            traceback.print_exc()

    # Modified key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cleanup_resources()
        break
    elif key == ord('l'):
        draw_line = True
    elif key == ord('k'):
        # Store current effects for fade out
        fade_frames = 20  # Number of fade steps
        current_frame = result.copy()  # Store current effect frame

        # Gradual fade out
        for i in range(fade_frames):
            alpha = 1.0 - (i / fade_frames)
            fade_frame = cv2.addWeighted(current_frame, alpha,
                                         np.zeros_like(current_frame), 1-alpha, 0)
            cv2.imshow('Lipstick Lines', fade_frame)
            cv2.waitKey(20)  # Small delay for smooth fade

        # Kill all effects after fade
        mesh_mode = False
        show_only_lines = False
        fade_alpha = 0.0
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        line_points = []
        line_velocities = []
        point_velocities = []

        # Show final black frame
        black_frame = np.zeros_like(frame)
        cv2.imshow('Lipstick Lines', black_frame)
        cv2.imshow(MASK_WINDOW_NAME, mask)

    elif key == ord('p'):
        pause_video = not pause_video  # Toggle pause state
        show_only_lines = not show_only_lines  # Toggle visibility

        if pause_video:
            paused_frame = frame.copy()  # Store current frame
            fade_alpha = 0.0  # Hide video
        else:
            # Restore video visibility when unpausing
            fade_alpha = 1.0
            show_only_lines = False  # Ensure video is visible when unpausing

        # Only do fade effect if showing video
        if not show_only_lines:
            temp_frame = frame.copy()
            fade_in_alpha = 0.0
            while fade_in_alpha < 1.0:
                fade_in_alpha += 0.1
                fade_result = cv2.addWeighted(temp_frame, fade_in_alpha,
                                              np.zeros_like(temp_frame), 1-fade_in_alpha, 0)
                # Add effects if they exist
                if len(line_points) > 0 or np.any(overlay):
                    fade_result = cv2.addWeighted(
                        fade_result, 1, overlay, 1, 0)

                cv2.imshow('Lipstick Lines', fade_result)
                cv2.imshow(MASK_WINDOW_NAME, mask)
                cv2.waitKey(30)

    elif key == ord('c'):
        mesh_mode = not mesh_mode
        show_only_lines = True
        fade_alpha = 0.0
        current_mesh_speed = INITIAL_MESH_SPEED
        transition_alpha = 0.0  # Reset transition

        if mesh_mode and contours:
            current_contour = max(contours, key=cv2.contourArea)
            # Initialize slow velocities for all points
            point_velocities = []
            for i in range(len(line_points)):
                angle = random.uniform(0, 2 * math.pi)
                speed = INITIAL_MESH_SPEED * 0.5
                point_velocities.append((
                    speed * math.cos(angle),
                    speed * math.sin(angle)
                ))
        else:
            overlay = np.zeros((height, width, 3), dtype=np.uint8)
            point_velocities = []
            current_contour = None

    elif key == ord('m'):
        if mesh_mode:  # Only toggle if in mesh mode
            manual_move = not manual_move
            if manual_move:
                # Initialize movement direction
                current_direction = (random.uniform(-1, 1),
                                     random.uniform(-1, 1))
                # Normalize direction vector
                magnitude = math.sqrt(
                    current_direction[0]**2 + current_direction[1]**2)
                if magnitude > 0:
                    current_direction = (current_direction[0]/magnitude,
                                         current_direction[1]/magnitude)
                # Initialize velocities in the current direction
                point_velocities = [(current_direction[0] * MANUAL_MOVE_SPEED,
                                     current_direction[1] * MANUAL_MOVE_SPEED)
                                    for _ in range(len(line_points))]
            else:
                # Reset to random movement
                current_direction = None
                point_velocities = [(random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2))
                                    for _ in range(len(line_points))]

    elif key == ord('f'):
        if mesh_mode:
            mouse_follow_mode = not mouse_follow_mode
            if mouse_follow_mode:
                # Create explosive velocities from center
                point_velocities = create_explosion_velocities(line_points)
                # Reset transition alpha to ensure mesh is visible
                transition_alpha = 1.0
            else:
                # Reset to normal movement
                point_velocities = [(random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2))
                                    for _ in range(len(line_points))]

    # Add after other key handlers in the main loop
    elif key == ord('r'):
        if mesh_mode and mouse_follow_mode:
            # Create new explosive velocities from current positions
            point_velocities = create_explosion_velocities(line_points)
            # Reset transition alpha to ensure mesh is visible
            transition_alpha = 1.0
            # Add a small random displacement to each point for more variety
            for i in range(len(line_points)):
                x, y = line_points[i]
                line_points[i] = (
                    x + random.randint(-5, 5),
                    y + random.randint(-5, 5)
                )

    # Modified fade effect
    if show_only_lines and fade_alpha > 0:
        fade_alpha = max(0, fade_alpha - 0.05)  # Slowed down fade out

    # Add this inside the main loop where mesh_mode movement is handled (after update_mesh_points call):
        if mesh_mode and manual_move and current_direction:
            # Change direction randomly based on probability
            if random.random() < DIRECTION_CHANGE_PROB:
                new_direction = (random.uniform(-1, 1), random.uniform(-1, 1))
                magnitude = math.sqrt(
                    new_direction[0]**2 + new_direction[1]**2)
                if magnitude > 0:
                    current_direction = (new_direction[0]/magnitude,
                                         new_direction[1]/magnitude)

            # Update positions using manual movement
            if not update_manual_movement(line_points, point_velocities,
                                          current_direction, largest_contour):
                # If movement failed, try a new direction
                current_direction = (-current_direction[0], -
                                     current_direction[1])

    # Modify main loop for memory management
    frame_counter += 1
    if frame_counter - last_cleanup >= CLEANUP_INTERVAL:
        cleanup_resources()
        last_cleanup = frame_counter

# Add before final cleanup
syphon_server.stop()  # Stop Syphon server before exiting

cap.release()
cv2.destroyAllWindows()
gc.collect()
