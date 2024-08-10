import matplotlib.pyplot as plt
import numpy as np
from heapq import heappop, heappush
from collections import deque

# Set the random seed for reproducibility
np.random.seed(100)
countStep = 0
# Define the size of the 2D space
WIDTH = 10
HEIGHT = 10
# Define the size of each grid cell
CELL_SIZE = 1
# Define the minimum step size for searching
SEARCH_STEP = 1
# Define the possible movement directions (including diagonals)
MOVEMENT_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

# Generate random rectangular obstacles
NUM_OBSTACLES = 5
OBSTACLE_SIZE_RANGE = (2, 4)
obstacles = []
for _ in range(NUM_OBSTACLES):
    size = np.random.randint(*OBSTACLE_SIZE_RANGE, size=2)
    position = np.random.randint(0, WIDTH - size[0]), np.random.randint(0, HEIGHT - size[1])
    obstacles.append((*position, *size))

# Calculate the Euclidean distance between two points 
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_total_euclidean_distance(coordinates):
    total_distance = 0.0
    
    for i in range(len(coordinates) - 1):
        total_distance += calculate_distance(coordinates[i],coordinates[i+1] )
    
    return total_distance

# Check if a point is inside an obstacle
def is_point_inside_obstacle(point):
    x, y = point
    for obstacle in obstacles:
        x_obs, y_obs, width, height = obstacle
        if x_obs <= x < x_obs + width and y_obs <= y < y_obs + height:
            return True
    return False

# Check if a point is valid (inside the boundaries and not inside an obstacle)
def is_valid_point(point):
    x, y = point
    return 0 <= x < WIDTH and 0 <= y < HEIGHT and not is_point_inside_obstacle(point)

# Find the shortest path using the A* algorithm using 
# the Euclidean distance between the start and end points as the heuristic 
def find_shortest_path_Astar(start, end, ax):
    #global countStep
    open_list = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: calculate_distance(start, end)}
    steps = 0
    visitedNode = 0
    
    while open_list:
        steps += 1
        current = heappop(open_list)[1]

        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], steps, g_score[end], visitedNode

        for dx, dy in MOVEMENT_DIRECTIONS:
            next_point = current[0] + dx * SEARCH_STEP, current[1] + dy * SEARCH_STEP

            if is_valid_point(next_point) and is_valid_move(current, next_point):
                new_g_score = g_score[current] + calculate_distance(current, next_point)
                if next_point not in g_score or new_g_score < g_score[next_point]:
                    came_from[next_point] = current
                    g_score[next_point] = new_g_score
                    # Calculate the heuristic score
                    h_score = calculate_distance(next_point, end) 
                    f_score[next_point] = new_g_score + h_score
                    heappush(open_list, (f_score[next_point], next_point))

                    # Plot visited nodes as black dotted lines
                    ax.plot(*zip(current, next_point), color='gray', linestyle='dotted')
                    visitedNode+=1

    return None, steps, float('inf'), visitedNode

# Find the shortest path using the Best-First Search algorithm using 
# the Euclidean distance between the start and end points as the heuristic 
def find_shortest_path_BestFirst(start, end, ax):
    open_list = [(calculate_distance(start, end), start)]
    came_from = {}
    g_score = {start: 0}
    steps = 0
    visitedNode = 0
    while open_list:
        steps += 1
        current = heappop(open_list)[1]

        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], steps, g_score[end], visitedNode

        for dx, dy in MOVEMENT_DIRECTIONS:
            next_point = current[0] + dx * SEARCH_STEP, current[1] + dy * SEARCH_STEP

            if is_valid_point(next_point) and is_valid_move(current, next_point):
                new_g_score = g_score[current] + calculate_distance(current, next_point)
                if next_point not in g_score or new_g_score < g_score[next_point]:
                    came_from[next_point] = current
                    g_score[next_point] = new_g_score
                    priority = calculate_distance(next_point, end)
                    heappush(open_list, (priority, next_point))
                    visitedNode+=1
                    # Plot visited nodes as black solid lines
                    ax.plot(*zip(current, next_point), color='gray', linestyle='dotted')

    return None, steps, float('inf'), visitedNode

# Find the shortest path using Dijkstra's algorithm
def find_shortest_path_Dijkstra(start, end, ax):
    open_list = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    steps = 0
    visitedNode = 0
    
    while open_list:
        steps += 1
        current = heappop(open_list)[1]

        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], steps, g_score[end], visitedNode

        for dx, dy in MOVEMENT_DIRECTIONS:
            next_point = current[0] + dx * SEARCH_STEP, current[1] + dy * SEARCH_STEP

            if is_valid_point(next_point) and is_valid_move(current, next_point):
                new_g_score = g_score[current] + calculate_distance(current, next_point)
                if next_point not in g_score or new_g_score < g_score[next_point]:
                    came_from[next_point] = current
                    g_score[next_point] = new_g_score
                    heappush(open_list, (g_score[next_point], next_point))

                    # Plot visited nodes as black dotted lines
                    ax.plot(*zip(current, next_point), color='gray', linestyle='dotted')
                    visitedNode+=1
                    

    return None, steps, float('inf'), visitedNode



# Find the shortest path using the Breadth-First Search (BFS) algorithm
def find_shortest_path_BFS(start, end, ax):
    queue = deque([start])
    visited = {start}
    came_from = {}
    steps = 0
    visitedNode = 0
    while queue:
        steps += 1
        current = queue.popleft()
        
        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], steps, len(path) - 1, visitedNode

        for dx, dy in MOVEMENT_DIRECTIONS:
            next_point = current[0] + dx * SEARCH_STEP, current[1] + dy * SEARCH_STEP

            if is_valid_point(next_point) and is_valid_move(current, next_point) and next_point not in visited:
                visited.add(next_point)
                came_from[next_point] = current
                queue.append(next_point)

                # Plot visited nodes as black dotted lines
                ax.plot(*zip(current, next_point), color='gray', linestyle='dotted')
                visitedNode+=1

    return None, steps, float('inf'), visitedNode

# Find the shortest path using Depth-First Search (DFS) algorithm
def find_shortest_path_DFS(start, end, ax):
    stack = [(start, [start])]
    visited = set()
    steps = 0
    visitedNode = 0
    while stack:
        steps += 1
        current, path = stack.pop()

        if current == end:
            return path, steps, len(path) - 1, visitedNode

        if current in visited:
            continue

        visited.add(current)

        for dx, dy in MOVEMENT_DIRECTIONS:
            next_point = current[0] + dx * SEARCH_STEP, current[1] + dy * SEARCH_STEP

            # Check if the next point is a valid move and does not cut corners
            if is_valid_point(next_point) and is_valid_move(current, next_point):
                stack.append((next_point, path + [next_point]))

                # Plot visited nodes as black dotted lines
                ax.plot(*zip(current, next_point), color='gray', linestyle='dotted')
                visitedNode+=1

    return None, steps, float('inf'), visitedNode

# Check if a move is valid
def is_valid_move(current, next_point):
    x1, y1 = current
    x2, y2 = next_point
    dx = x2 - x1
    dy = y2 - y1

    if dx != 0 and dy != 0:
        # Check if the move is diagonal
        # Prevent diagonal moves that cut corners by checking if the adjacent cells are obstacles
        if not is_valid_point((x1 + dx, y1)) or not is_valid_point((x1, y1 + dy)):
            return False

    return True

# Plot the obstacles
def plot_obstacles(ax):
    for obstacle in obstacles:
        x, y, width, height = obstacle
        rect = plt.Rectangle((x, y), width, height, fc='gray', ec='black')
        ax.add_patch(rect)

# Plot the start and end points
def plot_start_and_end(ax, start, end):
    ax.plot(*start, 'go', label='Start')
    ax.plot(*end, 'ro', label='End')

# Plot path
def plot_path(ax, path, color):
    x_coords, y_coords = zip(*path)
    ax.plot(x_coords, y_coords, color=color, label='Shortest Path')
        
# Plotting shortest path and statistics
def find_shortest_path(ax, algorithm, start, end):
    if algorithm == 'Breadth First Search':
        path, steps, distance, nodes = find_shortest_path_BFS(start, end, ax)
        distance = calculate_total_euclidean_distance(path)
    elif algorithm == 'Depth First Search':
        path, steps, distance, nodes = find_shortest_path_DFS(start, end, ax)
        distance = calculate_total_euclidean_distance(path)
    elif algorithm == 'A*':
        path, steps, distance, nodes = find_shortest_path_Astar(start, end, ax)
    elif algorithm == 'Best First Search':
        path, steps, distance, nodes = find_shortest_path_BestFirst(start, end, ax)

    if path is not None:
        ax.set_title(f"{algorithm}")
        plot_path(ax, path, 'blue')
        
        distance = round(distance,2)
        ax.set_title(f"{algorithm}- visited nodes: {nodes} distance: {distance}")
    else:
        ax.set_title(f"{algorithm}- Path not found!")

# Main function
def main():
    # Generate random start and end points that are not inside obstacles
    while True:
        start = np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)
        end = np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)
        if is_valid_point(start) and is_valid_point(end):
            break

    # Plotting grid and shortest paths for each algorithm
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    for i, algorithm in enumerate(['Depth First Search', 'Breadth First Search', 'A*', 'Best First Search']):
        ax = axes[i // 2][i % 2]
        ax.grid(True)
        ax.set_xlim([0, WIDTH])
        ax.set_ylim([0, HEIGHT])
        ax.set_aspect('equal')
        ax.set_aspect('equal', adjustable='box')
        
        plot_obstacles(ax)
        plot_start_and_end(ax, start, end)
        find_shortest_path(ax, algorithm, start, end)
        
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
