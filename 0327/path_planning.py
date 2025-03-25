import heapq
import random
import math

from collections import defaultdict

import matplotlib.pyplot as plt

class Path:
    def path_planning(start, goal, obstacles, map_size):
        """
        Parameters:
            start: Tuple[int, int]  # 시작 위치
            goal: Tuple[int, int]   # 목표 위치
            obstacles: Set[Tuple[int, int]]  # 장애물 좌표 집합
            map_size: Tuple[int, int]  # (width, height)

        Returns:
            path: List[Tuple[int, int]]  # start에서 goal까지의 경로
        """
        pass

class A_star(Path):
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos, map_size, obstacles):
        x, y = pos
        w, h = map_size
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in obstacles:
                neighbors.append((nx, ny))
        return neighbors
    
    def path_planning(self, start, goal, obstacles, map_size):
        open_set = [] # (priority = g(n) + h(n), cost_so_far = g(n), position)
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))
        came_from = {} # 경로 복원을 위한 정보 저장
        cost_so_far = {start : 0}  # 지금까지 온 거리 저장
        
        while open_set:
            _, cost, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                
                while current != start:
                    path.append(current)
                    current = came_from[current]
                    
                path.append(start)
                return path[::-1]
        
            for neighbor in self.get_neighbors(current, map_size, obstacles):
                new_cost = cost_so_far[current] + 1
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
            
        return []

class Dijkstra(Path):
    def get_neighbors(self, pos, map_size, obstacles):
        x, y = pos
        w, h = map_size
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in obstacles:
                neighbors.append((nx, ny))
        return neighbors
    
    def path_planning(self, start, goal, obstacles, map_size):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        cost_so_far = {start : 0}
        
        while open_set:
            cost, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                
                while current != start:
                    path.append(current)
                    current = came_from[current]
                
                path.append(start)
                return path[::-1]

            for neighbor in self.get_neighbors(current, map_size, obstacles):
                new_cost = cost_so_far[current] + 1
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(open_set, (new_cost, neighbor))
                    came_from[neighbor] = current
                
        return []
    
class RRT(Path):
    def distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def steer(self, from_node, to_node, step_size = 1):
        if from_node == to_node: return from_node
        
        dx, dy = to_node[0] - from_node[0], to_node[1] - from_node[1]
        length = math.hypot(dx, dy)
        
        if length == 0:
            return from_node
        
        return (
            int(from_node[0] + step_size * dx / length),
            int(from_node[1] + step_size * dy / length)
        )
        
    def collision_free(self, from_node, to_node, obstacles, map_size):
        steps = int(self.distance(from_node, to_node))
        if steps == 0: return True
        
        for i in range(steps + 1):
            x = int(from_node[0] + (to_node[0] - from_node[0]) * i / steps)
            y = int(from_node[1] + (to_node[1] - from_node[1]) * i / steps)

            # 지도 범위 초과 방지
            if not (0 <= x < map_size[0] and 0 <= y < map_size[1]):
                return False

            if (x, y) in obstacles:
                return False

        return True
    
    def path_planning(self, start, goal, obstacles, map_size):
        nodes = {start: None}
        max_iter = 5000
        step_size = 5

        for _ in range(max_iter):
            rand_point = (
                random.randint(0, map_size[0] - 1),
                random.randint(0, map_size[1] - 1)
            )

            nearest = min(nodes.keys(), key=lambda n: self.distance(n, rand_point))
            new_node = self.steer(nearest, rand_point, step_size)

            if new_node in nodes:
                continue

            if not self.collision_free(nearest, new_node, obstacles, map_size):
                continue

            nodes[new_node] = nearest

            if self.distance(new_node, goal) < step_size:
                if self.collision_free(new_node, goal, obstacles, map_size):
                    nodes[goal] = new_node
                    break
        
        if goal not in nodes:
            return []
        print('check')
        # reconstruct path
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = nodes[current]
            
        return path[::-1]
    
class PRM(Path):
    def euclidean(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def is_free(self, p, obstacles):
        return p not in obstacles
    
    def is_collision_free(self, p1, p2, obstacles):
        """Bresenham's line algorithm으로 두 점 사이 직선이 장애물을 통과하는지 확인"""
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if (x, y) in obstacles:
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                if (x, y) in obstacles:
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        # 마지막 점 확인
        return (x2, y2) not in obstacles
    
    def path_planning(self, start, goal, obstacles, map_size):
        num_samples = 200
        k_neighbors = 10
        nodes = [start, goal]
        
        while len(nodes) < num_samples:
            p = (random.randint(0, map_size[0] - 1), random.randint(0, map_size[1] - 1))
            
            if self.is_free(p, obstacles):
                nodes.append(p)
                
        graph = defaultdict(list)
        
        for i, node in enumerate(nodes):
            dists = sorted(((self.euclidean(node, other), other) for other in nodes if other != node), key = lambda x: x[0])
            
            for _, neighbor in dists[:k_neighbors]:
                if self.is_collision_free(node, neighbor, obstacles):
                    graph[node].append(neighbor)
                    graph[neighbor].append(node)
                    
        open_set = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while open_set:
            cost, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                    
                return path[::-1]
            
            for neighbor in graph[current]:
                new_cost = cost + self.euclidean(current, neighbor)
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (new_cost, neighbor))
                    
        return []

def print_path_on_map(path, map_size, obstacles, start, goal):
    for y in range(map_size[1]):
        row = ""
        for x in range(map_size[0]):
            if (x, y) == start:
                row += "S"
            elif (x, y) == goal:
                row += "G"
            elif (x, y) in obstacles:
                row += "#"
            elif (x, y) in path:
                row += "*"
            else:
                row += "."
        print(row)
    print("\n")
    
def plot_path_on_map(path, map_size, obstacles, start, goal):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, map_size[0])
    ax.set_ylim(-1, map_size[1])
    ax.set_xticks(range(map_size[0]))
    ax.set_yticks(range(map_size[1]))
    ax.set_aspect('equal')
    ax.grid(True)

    # 장애물
    for obs in obstacles:
        ax.add_patch(plt.Rectangle(obs, 1, 1, color='black'))

    # 경로
    if path:
        # path_x = [x + 0.5 for x, y in path]
        # path_y = [y + 0.5 for x, y in path]
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, color='blue', linewidth=2, marker='o', label='Path')

    # 시작점
    ax.add_patch(plt.Rectangle(start, 1, 1, color='green'))
    ax.text(start[0]+0.1, start[1]+0.1, 'S', color='white', weight='bold')

    # 도착점
    ax.add_patch(plt.Rectangle(goal, 1, 1, color='red'))
    ax.text(goal[0]+0.1, goal[1]+0.1, 'G', color='white', weight='bold')

    ax.legend(loc='upper right')
    plt.gca().invert_yaxis()  # (0,0)이 좌상단이 되도록
    plt.title("Path Planning Visualization")
    plt.show()

# 각 알고리즘에 대해 실행
algorithms = {
    "A*": A_star().path_planning,
    "Dijkstra": Dijkstra().path_planning,
    "RRT": RRT().path_planning,
    "PRM": PRM().path_planning,
}

map_size = (10, 10) # 맵 크기: 10x10
start = (0, 0)      # 시작점
goal = (9, 9)       # 도착점
obstacles_list = [  # 장애물: 대각선으로 막힌 구간 일부
    # 1. 대각선 장애물
    {(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8)},

    # 2. 가로줄 장애물 (중간 줄을 막음, 하지만 경로는 좌우 끝이 뚫려 있음)
    {(x, 5) for x in range(9)},

    # 3. 정중앙에 통로로
    {(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (6, 5), (7, 5), (8, 5), (9, 5)}
]

for name, algo in algorithms.items():
    print(f"=== {name} ===")
    for obstacles in obstacles_list:
        path = algo(start, goal, obstacles, map_size)
        if path:
            print(f"Found path! Length: {len(path)}")
            print_path_on_map(path, map_size, obstacles, start, goal)
            plot_path_on_map(path, map_size, obstacles, start, goal)
        else:
            print("No path found.\n")
