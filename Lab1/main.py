# TODO: Import libraries
from collections import deque
import numpy as np
import time
import tracemalloc
import heapq
# 1. Search Strategies Implementation
# Define Node for ease handling
class Node:
    def __init__(self, state, path_cost, parent=None):
        self.state = state
        self.path_cost = path_cost
        self.parent = parent
    def __lt__(self, other):
        return self.path_cost < other.path_cost

def goal_test(state, destination):
    return state == destination

def get_actions(state, arr):
    return [i for i, weight in enumerate(arr[state]) if weight > 0] # indices of nodes

def child_node(parent_node, action):
    return Node(state=action, path_cost=parent_node.path_cost + 1, parent=parent_node) # child node

def child_node_cost(parent_node, action, cost):
    return Node(state=action, path_cost=parent_node.path_cost + cost, parent=parent_node) # child node with path cost

def solution(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    path.reverse()
    return path # solution path

# 1.1. Breadth-first search (BFS)
def bfs(arr, source, destination):
    """
    BFS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO

    path = []
    visited = {}
    
    # Initialization
    start_node = Node(state=source, path_cost=0)
    frontier = deque([start_node])
    # Exploration
    while frontier:
        node = frontier.popleft()  # Dequeue node
        if goal_test(node.state, destination):
            return visited, solution(node)

        visited[node.state] = node.parent.state if node.parent else None  # Mark node as visited

        for action in get_actions(node.state, arr):
            if action not in visited and not any(n.state == action for n in frontier):
                child = child_node(node, action)
                if goal_test(child.state, destination):
                    return visited, solution(child)
                frontier.append(child)

    # If no solution is found
    path = [-1]
    return visited, path

# 1.2. Depth-first search (DFS)
def dfs(arr, source, destination):
    """
    DFS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO

    path = []
    visited = {}
    # Initialization
    start_node = Node(state=source, path_cost=0)
    stack = [start_node]
    # Exploration
    while stack:
        node = stack.pop()  # Pop node from the stack
        if goal_test(node.state, destination):
            return visited, solution(node)

        if node.state not in visited:
            visited[node.state] = node.parent.state if node.parent else None  # Mark node as visited with its parent

            for action in get_actions(node.state, arr):
                if action not in visited:
                    child = child_node(node, action)
                    stack.append(child)

    # If no solution is found
    path = [-1]
    return visited, path


# 1.3. Uniform-cost search (UCS)
def ucs(arr, source, destination):
    """
    UCS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO

    path = []
    visited = {}
    # Initialization
    frontier = []
    heapq.heappush(frontier, Node(state=source, path_cost=0))
    # Exploration
    while frontier:
        node = heapq.heappop(frontier)  # Dequeue the node with the lowest path cost
        if goal_test(node.state, destination):
            visited[node.state] = node.parent.state if node.parent else None
            return visited, solution(node)

        if node.state not in visited:
            visited[node.state] = node.parent.state if node.parent else None  # Mark node as visited

            for action in get_actions(node.state, arr):
                if action not in visited:
                    cost = arr[node.state][action]
                    child = child_node_cost(node, action, cost)
                    heapq.heappush(frontier, child)

    # If no solution is found
    path = [-1]
    return visited, path

# 1.4. Iterative deepening search (IDS)
# 1.4.a. Depth-limited search
def dls(arr, source, destination, depth_limit):
    """
    DLS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    depth_limit: integer
        Maximum depth for search
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO

    path = []
    visited = {}
    def recursive_dls(node, depth):
        if goal_test(node.state, destination):
            path.extend(solution(node))
            return True
        
        if depth == 0:
            return False

        visited[node.state] = node.parent.state if node.parent else None  # Mark node as visited
        
        for action in get_actions(node.state, arr):
            if action not in visited:
                child = child_node(node, action)
                if child.state not in visited:
                    visited[child.state] = node.state
                if recursive_dls(child, depth - 1):
                    return True
        
        return False
    
    start_node = Node(state=source, path_cost=0)
    if recursive_dls(start_node, depth_limit):
        return visited, path
    else:
        return visited, [-1]

# 1.4.b. IDS
def ids(arr, source, destination):
    """
    IDS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO

    path = []
    visited = {}

    for depth in range(1, len(arr) + 1):  # The depth goes from 1 -> number of nodes in the graph
        visited, path = dls(arr, source, destination, depth)
        if path != [-1]:  
            return visited, path

    # If no solution is found
    path = [-1] 
    return visited, path


# 1.5. Greedy best first search (GBFS)
def gbfs(arr, source, destination, heuristic):
    """
    GBFS algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    heuristic: list / numpy array
        The heuristic value from the current node to the goal
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO

    path = []
    visited = {}
    # Initialization
    frontier = []
    heapq.heappush(frontier, Node(state=source, path_cost=heuristic[source]))
    # Exploration
    while frontier:
        node = heapq.heappop(frontier)  # Dequeue the node with the lowest heuristic cost
        if goal_test(node.state, destination):
            return visited, solution(node)

        visited[node.state] = node.parent.state if node.parent else None  # Mark node as visited

        for action in get_actions(node.state, arr):
            if action not in visited and not any(n.state == action for n in frontier):
                child = Node(state=action, path_cost=heuristic[action], parent=node)
                if goal_test(child.state, destination):
                    return visited, solution(child)
                heapq.heappush(frontier, child)

    # If no solution is found
    path = [-1]
    return visited, path


# 1.6. Graph-search A* (AStar)
def astar(arr, source, destination, heuristic):
    """
    A* algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    heuristic: list / numpy array
        The heuristic value from the current node to the goal
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO

    path = []
    visited = {}
    # Initialization
    frontier = []
    heapq.heappush(frontier, (heuristic[source], Node(state=source, path_cost=0)))
    # Exploration
    while frontier:
        _, node = heapq.heappop(frontier)  # Dequeue the node with the lowest f(n) = g(n) + h(n)
        if goal_test(node.state, destination):
            visited[node.state] = node.parent.state if node.parent else None
            return visited, solution(node)

        if node.state not in visited:
            visited[node.state] = node.parent.state if node.parent else None  # Mark node as visited.parent else None  # Mark node as visited

        for action in get_actions(node.state, arr):
            if action not in visited:
                cost = arr[node.state][action]
                g_cost = node.path_cost + cost
                f_cost = g_cost + heuristic[action]
                child = Node(state=action, path_cost=g_cost, parent=node)
                heapq.heappush(frontier, (f_cost, child))

    # If no solution is found
    path = [-1]
    return visited, path


# 1.7. Hill-climbing First-choice (HC)
def hc(arr, source, destination, heuristic):
    """
    HC algorithm:
    Parameters:
    ---------------------------
    arr: list / numpy array 
        The graph's adjacency matrix
    source: integer
        Starting node
    destination: integer
        Ending node
    heuristic: list / numpy array
        The heuristic value from the current node to the goal
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO

    path = []
    visited = {}
    # Initialization
    current_node = Node(state=source, path_cost=heuristic[source])
    visited[current_node.state] = current_node.parent.state if current_node.parent else None
    # Exploration
    while True:
        neighbors = get_actions(current_node.state, arr)
        
        if not neighbors:
            break
        
        next_node = None
        for action in neighbors:
            candidate_node = Node(state=action, path_cost=heuristic[action], parent=current_node)
            if goal_test(candidate_node.state, destination):
                return visited, solution(candidate_node)
            if not next_node or candidate_node.path_cost < next_node.path_cost:
                next_node = candidate_node

        if next_node and next_node.path_cost < current_node.path_cost:
            current_node = next_node
            visited[current_node.state] = current_node.parent.state if current_node.parent else None
        else:
            break

    path = solution(current_node)
    if goal_test(current_node.state, destination):
        return visited, path
    else:
        return visited, [-1]

# Read file function
def read_file(directory):
    with open(directory, 'r') as file:
        # number of nodes
        num_nodes = int(file.readline().strip())
        
        # start and goal nodes
        start_node, goal_node = map(int, file.readline().strip().split())
        
        # adjacency matrix
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for i in range(num_nodes):
            adjacency_matrix[i] = np.array(list(map(int, file.readline().strip().split())))
        
        # heuristic
        heuristic_weights = np.array(list(map(int, file.readline().strip().split())))
        
        return start_node, goal_node, adjacency_matrix, heuristic_weights

# Execute each algorithm
def executing_algorithm(directory, algo):
    start, goal, matrix, h = read_file(directory)
    start_time = time.perf_counter()
    tracemalloc.start()
    if algo.__name__ == 'astar' or algo.__name__ == 'gbfs' or algo.__name__ == 'hc':
        visited, path = algo(matrix, start, goal, h)
    else:
        visited, path = algo(matrix, start, goal)
    #print(visited)
    end_time = time.perf_counter()
    run_time = (end_time - start_time) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_usage = peak / 1024
    return path, memory_usage, run_time

# Write the result to .txt file
def output_file(algo, path, memory_usage, run_time, output_filename):
    with open(output_filename, 'a') as file:
        file.write(f"{algo} :\n")
        if not path:
            file.write("Path: -1\n")
        else:
            file.write("Path: ")
            for i in range(len(path)):
                if i < len(path) - 1:
                    file.write(f"{path[i]} -> ")
                else:
                    file.write(f"{path[i]}\n")
        file.write(f"Time: {run_time} ms\n")
        file.write(f"Memory: {memory_usage} KB\n")

def main():
    input_test = 'graph05.txt'
    output_filename = 'result.txt'

    with open(output_filename, 'w') as file:
        file.write("")

    #1. BFS
    path_bfs, memory_bfs, runtime_bfs = executing_algorithm(input_test, bfs)
    output_file('BFS', path_bfs, memory_bfs, runtime_bfs, output_filename)
    #2. DFS
    path_dfs, memory_dfs, runtime_dfs = executing_algorithm(input_test, dfs)
    output_file('DFS', path_dfs, memory_dfs, runtime_dfs, output_filename)
    #3. UCS
    path_ucs, memory_ucs, runtime_ucs = executing_algorithm(input_test, ucs)
    output_file('UCS', path_ucs, memory_ucs, runtime_ucs, output_filename)
    #4. IDS
    path_ids, memory_ids, runtime_ids = executing_algorithm(input_test, ids)
    output_file('IDS', path_ids, memory_ids, runtime_ids, output_filename)
    #5. GBFS
    path_gbfs, memory_gbfs, runtime_gbfs = executing_algorithm(input_test, gbfs)
    output_file('GBFS', path_gbfs, memory_gbfs, runtime_gbfs, output_filename)
    #6. Astar
    path_astar, memory_astar, runtime_astar = executing_algorithm(input_test, astar)
    output_file('Astar', path_astar, memory_astar, runtime_astar, output_filename)
    #7. Hill Climbing:
    path_hc, memory_hc, runtime_hc = executing_algorithm(input_test, hc)
    output_file('HC', path_hc, memory_hc, runtime_hc, output_filename)

# 2. Main function
if __name__ == "__main__":

    # TODO: Read the input data

    # TODO: Start measuring

    # TODO: Call a function to execute the path finding process
    
    # TODO: Stop measuring 

    # TODO: Show the output data
    main()

    pass