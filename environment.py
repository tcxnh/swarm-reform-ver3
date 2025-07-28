import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
import numpy as np
from agent import Agent
from position_class import Position
import random
from matplotlib.text import Text  
import math

class Environment:
    def __init__(self, num_agents=16, width=100, height=100, config=None):
        self.width = width
        self.height = height
        self.comm_range = 30 # Updated communication range
        self.vision_range = 20  # Updated vision range
        self.agents = []
        self.config = config  # Store config for obstacle initialization
        self.global_map, self.wall_mask = self._initialize_global_map()
        self.comm_map = None  # will be initialized per step
        # Initialize rank_config with default values
        self.rank_config = {
            0: 1,  # Entrance can manage 1 subordinate
            1: 1,  # Rank 1 can manage 1 subordinate
            2: 1,  # Rank 2 can manage 1 subordinate
            3: 1   # Rank 3 can manage 1 subordinate
        }
        # Update rank_config from config if provided
        if config and 'rank_config' in config:
            self.rank_config = {int(k): int(v) for k, v in config['rank_config'].items()}
        self.rank_config_modified = False  # Flag to track if rank_config has been modified
        
        # Steel mill environmental hazard zones
        self._initialize_environmental_zones()
        
        self.initialize_agents(num_agents)
        self.history = []  # save all the step information of the simulation
        self.is_running = False  # Control whether agents should move

    def _initialize_global_map(self):
        """
        Create an empty global map with walls on all borders.
        Also return a static wall mask (1 = wall, 0 = empty) for visualization.
        """
        global_map = []
        wall_mask = []

        for y in range(self.height):
            row = []
            mask_row = []
            for x in range(self.width):
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    row.append('#')    # wall
                    mask_row.append(1) # wall mask
                else:
                    row.append(' ')    # empty
                    mask_row.append(0) # empty mask
            global_map.append(row)
            wall_mask.append(mask_row)

        # Add obstacles if configured
        if hasattr(self, 'config') and self.config.get('add_obstacles', False):
            # 先整体填满障碍（墙），再挖出Y型通路
            for y in range(self.height):
                for x in range(self.width):
                    if 0 < x < self.width-1 and 0 < y < self.height-1:
                        global_map[y][x] = '#'
                        wall_mask[y][x] = 1
            # Y型通路参数
            cx = self.width // 2
            cy = self.height // 2
            # 极大加宽
            main_width = 27
            branch_width = 22
            # 下干（竖线）——从cy一直挖到入口agent的y坐标
            entrance_y = self.height - 2
            for y in range(cy, entrance_y+1):
                for dx in range(-main_width//2, main_width//2+1):
                    global_map[y][cx+dx] = ' '
                    wall_mask[y][cx+dx] = 0
            # 左分支（左上斜线，短，夹角更大）
            for i in range(0, 25):
                x = cx - 12 - int(i*1.2)  # 斜率更大
                y = cy - i
                for dx in range(-branch_width//2, branch_width//2+1):
                    if 0 < x+dx < self.width-1 and 0 < y < self.height-1:
                        global_map[y][x+dx] = ' '
                        wall_mask[y][x+dx] = 0
            # 右分支（右上斜线，更长更宽，夹角更大）
            for i in range(0, 90):
                x = cx + 12 + int(i*1.8)  # 斜率更大，右分支更长
                y = cy - i
                for dx in range(-branch_width, branch_width+1):
                    if 0 < x+dx < self.width-1 and 0 < y < self.height-1:
                        global_map[y][x+dx] = ' '
                        wall_mask[y][x+dx] = 0
            # 保证入口agent初始位置没有障碍物
            entrance_x = self.width // 2
            for dx in range(-main_width//2, main_width//2+1):
                global_map[entrance_y][entrance_x+dx] = ' '
                wall_mask[entrance_y][entrance_x+dx] = 0

        # Add target point if configured (independent of obstacles)
        if hasattr(self, 'config') and self.config.get('add_target', False):
            target_position = self.config.get('target_position', 'right')
            if target_position == 'right':
                target_x, target_y = 120, 60  # Right position
            else:  # top
                target_x, target_y = 75, 20   # Top position
            global_map[target_y][target_x] = '*'  # Set target point
            wall_mask[target_y][target_x] = 0  # Target is not a wall

        # Add fork point if configured (independent of obstacles)
        if hasattr(self, 'config') and self.config.get('add_fork', False):
            fork_position = self.config.get('fork_position', 'right')
            if fork_position == 'right':
                fork_x, fork_y = 120, 80  # Right position (和target错开)
            elif fork_position == 'middle':
                fork_x, fork_y = self.width // 2, self.height // 2  # Middle position
            else:
                fork_x, fork_y = 75, 40   # fallback for legacy 'top'
            global_map[fork_y][fork_x] = '^'  # Set fork point
            wall_mask[fork_y][fork_x] = 0  # Fork is not a wall

        # Add fork and dead end markers if configured
        if hasattr(self, 'config') and self.config.get('add_fork_deadend', False):
            # 左上分支死路
            cx = self.width // 2
            cy = self.height // 2
            deadend1_x, deadend1_y = cx - 40, cy - 22
            if 0 < deadend1_x < self.width-1 and 0 < deadend1_y < self.height-1:
                global_map[deadend1_y][deadend1_x] = 'X'
                wall_mask[deadend1_y][deadend1_x] = 0
            # 右上分支死路
            deadend2_x, deadend2_y = cx + 90, cy - 70
            if 0 < deadend2_x < self.width-1 and 0 < deadend2_y < self.height-1:
                global_map[deadend2_y][deadend2_x] = 'X'
                wall_mask[deadend2_y][deadend2_x] = 0

        return global_map, wall_mask

    def _initialize_environmental_zones(self):
        """
        Initialize environmental hazard zones for steel mill simulation.
        These zones simulate real-world conditions that affect robot communication and operation.
        """
        # High temperature zones (around furnaces, molten metal areas)
        # Format: (center_x, center_y, radius, intensity)
        self.high_temp_zones = [
            (30, 30, 15, 0.8),   # Main furnace area - high intensity
            (70, 40, 12, 0.6),   # Secondary heating zone
            (50, 70, 10, 0.5),   # Cooling area - lower intensity
            (80, 80, 8, 0.7),    # Hot metal processing
        ]
        
        # Magnetic interference zones (around large equipment, electromagnetic systems)
        # Format: (center_x, center_y, radius, intensity)
        self.magnetic_zones = [
            (40, 50, 20, 0.7),   # Large electromagnetic crane
            (60, 20, 15, 0.5),   # Induction heating system
            (20, 60, 12, 0.6),   # Electric arc furnace
            (85, 60, 10, 0.4),   # Motor control center
        ]
        
        # GPS obstruction zones (steel structures, large buildings)
        # Format: (center_x, center_y, radius, intensity)
        self.gps_obstruction_zones = [
            (50, 50, 25, 0.9),   # Main building structure - very high obstruction
            (75, 75, 18, 0.8),   # Secondary structure
            (25, 75, 15, 0.7),   # Storage facility
            (75, 25, 20, 0.8),   # Manufacturing hall
        ]
        
        # Dynamic zones that can change over time
        self.dynamic_zones = {
            'radiation': [],      # Radiation zones (can appear during certain processes)
            'gas_leak': [],       # Gas leak zones (emergency situations)
            'equipment_failure': [] # Equipment failure zones
        }
        
        # Zone update parameters
        self.zone_update_interval = 50  # Update zones every 50 steps
        self.last_zone_update = 0

    def _update_dynamic_zones(self, current_step):
        """
        Update dynamic environmental zones based on simulation time.
        Simulates changing conditions in the steel mill.
        """
        if current_step - self.last_zone_update < self.zone_update_interval:
            return
        
        self.last_zone_update = current_step
        
        # Random radiation zones (during certain processes)
        if random.random() < 0.1:  # 10% chance per update
            radiation_x = random.randint(10, self.width - 10)
            radiation_y = random.randint(10, self.height - 10)
            self.dynamic_zones['radiation'].append((radiation_x, radiation_y, 8, 0.6, current_step + 30))  # Lasts 30 steps
        
        # Random gas leak simulation
        if random.random() < 0.05:  # 5% chance per update
            leak_x = random.randint(10, self.width - 10)
            leak_y = random.randint(10, self.height - 10)
            self.dynamic_zones['gas_leak'].append((leak_x, leak_y, 12, 0.8, current_step + 100))  # Lasts 100 steps
            print(f"Gas leak detected at ({leak_x}, {leak_y})!")
        
        # Equipment failure zones
        if random.random() < 0.03:  # 3% chance per update
            failure_x = random.randint(10, self.width - 10)
            failure_y = random.randint(10, self.height - 10)
            self.dynamic_zones['equipment_failure'].append((failure_x, failure_y, 10, 0.7, current_step + 80))
            print(f"Equipment failure at ({failure_x}, {failure_y})!")
        
        # Remove expired zones
        for zone_type in self.dynamic_zones:
            self.dynamic_zones[zone_type] = [
                zone for zone in self.dynamic_zones[zone_type]
                if zone[4] > current_step  # Check expiration time
            ]

    def initialize_agents_random(self, num_agents, config=None):
        """
        Randomly place agents in the environment while ensuring they are within safety range of each other.
        Safety range is (comm_range + vision_range)/2.
        """
        self.agents = []
        occupied = set()
        
        # First place the entrance agent
        entrance_x = self.width // 2
        entrance_y = self.height - 2
        entrance_pos = Position(entrance_x, entrance_y)
        function_mode = config.get('function_mode', 'flex-search') if config else 'flex-search'
        self.agents.append(Agent(position=entrance_pos, agent_id=0,
                                vision_range=self.vision_range,
                                comm_range=self.comm_range,
                                is_entrance=True,
                                function_mode=function_mode))
        occupied.add((entrance_x, entrance_y))

        # Calculate safety range
        safety_range = (self.comm_range + self.vision_range) / 2

        # Function to check if a position is valid (within bounds and not occupied)
        def is_valid_position(x, y):
            return (1 <= x < self.width - 1 and 
                   1 <= y < self.height - 1 and 
                   (x, y) not in occupied)

        # Function to check if a position is within safety range of any existing agent
        def is_within_safety_range(x, y):
            for agent in self.agents:
                dist = math.sqrt((x - agent.position.x)**2 + (y - agent.position.y)**2)
                if dist <= safety_range:
                    return True
            return False

        # Place remaining agents
        agent_id = 1
        max_attempts = 5000  # Increased max attempts due to stricter placement requirements
        attempts = 0

        while agent_id < num_agents and attempts < max_attempts:
            # Generate random position
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            
            if is_valid_position(x, y) and is_within_safety_range(x, y):
                agent = Agent(position=Position(x, y), agent_id=agent_id,
                            vision_range=self.vision_range,
                            comm_range=self.comm_range,
                            function_mode=function_mode)
                self.agents.append(agent)
                occupied.add((x, y))
                agent_id += 1
            attempts += 1

        if agent_id < num_agents:
            print(f"Warning: Could only place {agent_id} agents out of {num_agents} requested")

    def initialize_agents_grid(self, num_agents, config=None):
        """
        Automatically place agents in adjacent 1x1 grid cells expanding around entrance.
        Ensures no overlap and good spacing, without assuming square layout.
        """
        entrance_x = self.width // 2
        entrance_y = self.height - 2
        entrance_pos = Position(entrance_x, entrance_y)
        occupied = set()
        self.agents = []
        function_mode = config.get('function_mode', 'flex-search') if config else 'flex-search'

        # Place entrance
        self.agents.append(Agent(position=entrance_pos, agent_id=0,
                                vision_range=self.vision_range,
                                comm_range=self.comm_range,
                                is_entrance=True,
                                function_mode=function_mode))
        occupied.add((entrance_x, entrance_y))

        # Directions: 8-connected neighborhood
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Use a queue to expand outward
        from collections import deque
        queue = deque()
        queue.append((entrance_x, entrance_y))

        agent_id = 1
        while queue and agent_id < num_agents:
            x, y = queue.popleft()

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < self.width - 1 and 1 <= ny < self.height - 1:
                    if (nx, ny) not in occupied:
                        agent = Agent(position=Position(nx, ny), agent_id=agent_id,
                                    vision_range=self.vision_range,
                                    comm_range=self.comm_range,
                                    function_mode=function_mode)
                        self.agents.append(agent)
                        occupied.add((nx, ny))
                        queue.append((nx, ny))
                        agent_id += 1
                        if agent_id >= num_agents:
                            break

    def initialize_agents_binary_tree(self, num_agents, config=None):
        """
        Initialize agents in a binary tree structure.
        Each agent (except leaf nodes) will have two children.
        The tree will be placed starting from the entrance.
        Each agent must be within safety range of at least one other agent.
        """
        self.agents = []
        occupied = set()
        function_mode = config.get('function_mode', 'flex-search') if config else 'flex-search'
        
        # First place the entrance agent (root of the tree)
        entrance_x = self.width // 2
        entrance_y = self.height - 2
        entrance_pos = Position(entrance_x, entrance_y)
        self.agents.append(Agent(position=entrance_pos, agent_id=0,
                                vision_range=self.vision_range,
                                comm_range=self.comm_range,
                                is_entrance=True,
                                function_mode=function_mode))
        occupied.add((entrance_x, entrance_y))

        # Calculate spacing between levels and nodes
        safety_range = (self.comm_range + self.vision_range) / 2
        level_spacing = safety_range * 0.8  # Vertical spacing between levels
        node_spacing = safety_range * 0.8   # Horizontal spacing between nodes

        # Function to check if a position is valid
        def is_valid_position(x, y):
            return (1 <= x < self.width - 1 and 
                1 <= y < self.height - 1 and 
                (x, y) not in occupied)

        # Function to check if a position is within safety range of any existing agent
        def is_within_safety_range(x, y):
            for agent in self.agents:
                dist = math.sqrt((x - agent.position.x)**2 + (y - agent.position.y)**2)
                if dist <= safety_range:
                    return True
            return False

        # Function to create a binary tree level by level
        def create_binary_tree():
            current_level = [(entrance_x, entrance_y, 0)]  # (x, y, parent_id)
            agent_id = 1
            level = 0

            while current_level and agent_id < num_agents:
                next_level = []
                level_width = len(current_level)
                start_x = entrance_x - (level_width - 1) * node_spacing / 2

                for i, (parent_x, parent_y, parent_id) in enumerate(current_level):
                    # Calculate position for left child
                    left_x = start_x + i * node_spacing
                    left_y = parent_y - level_spacing

                    if is_valid_position(left_x, left_y) and is_within_safety_range(left_x, left_y) and agent_id < num_agents:
                        left_agent = Agent(position=Position(left_x, left_y), 
                                        agent_id=agent_id,
                                        vision_range=self.vision_range,
                                        comm_range=self.comm_range,
                                        function_mode=function_mode)
                        left_agent.direct_sup_id = parent_id
                        self.agents.append(left_agent)
                        occupied.add((left_x, left_y))
                        next_level.append((left_x, left_y, agent_id))
                        agent_id += 1

                    # Calculate position for right child
                    right_x = left_x + node_spacing
                    right_y = left_y

                    if is_valid_position(right_x, right_y) and is_within_safety_range(right_x, right_y) and agent_id < num_agents:
                        right_agent = Agent(position=Position(right_x, right_y), 
                                        agent_id=agent_id,
                                        vision_range=self.vision_range,
                                        comm_range=self.comm_range,
                                        function_mode=function_mode)
                        right_agent.direct_sup_id = parent_id
                        self.agents.append(right_agent)
                        occupied.add((right_x, right_y))
                        next_level.append((right_x, right_y, agent_id))
                        agent_id += 1

                current_level = next_level
                level += 1

        create_binary_tree()
        
        if len(self.agents) < num_agents:
            print(f"Warning: Could only place {len(self.agents)} agents out of {num_agents} requested")

    def initialize_agents(self, num_agents, config=None):
        """
        Initialize agents based on the configuration
        """
        if config is None:
            config = {}
            
        init_method = config.get('init_method', 'random')
        function_mode = config.get('function_mode', 'flex-search')
        
        if init_method == 'random':
            self.initialize_agents_random(num_agents, config)
        elif init_method == 'grid':
            self.initialize_agents_grid(num_agents, config)
        elif init_method == 'binary_tree':
            self.initialize_agents_binary_tree(num_agents, config)
            
        # Set environment reference for all agents
        for agent in self.agents:
            agent.environment = self

    def _initialize_comm_map(self):
        """
        Create a blank comm_map grid. Each cell stores a list of messages.
        """
        return [[[] for _ in range(self.width)] for _ in range(self.height)]

    def step(self):
        # Track current simulation step
        if not hasattr(self, 'current_step'):
            self.current_step = 0
        self.current_step += 1
        
        # Update dynamic environmental zones
        self._update_dynamic_zones(self.current_step)
        
        self.comm_map = self._initialize_comm_map()

        # Shuffle update order
        agent_list = self.agents.copy()
        random.shuffle(agent_list)

        # Phase 1: Everyone broadcasts
        for agent in agent_list:
            agent._broadcast_to_comm_map(self.comm_map, self.global_map)

        # Phase 2: Everyone moves (but in random order) only if simulation is running
        if self.is_running:
            for agent in agent_list:
                agent.move(self.global_map, self.comm_map)

        # Frame: wall + agents + target
        frame = [row.copy() for row in self.wall_mask]

        # Add target point to frame
        for y in range(self.height):
            for x in range(self.width):
                if self.global_map[y][x] == '*':
                    frame[y][x] = 3  # Use 3 for target point
                if self.global_map[y][x] == '^':
                    frame[y][x] = 4 # Use 4 for fork point
                if self.global_map[y][x] == 'X':
                    frame[y][x] = 5 # Use 5 for dead end (deep blue)

        # Add agents to frame
        agent_overlay = []
        for agent in self.agents:
            x, y = int(agent.position.x), int(agent.position.y)
            if 0 <= x < self.width and 0 <= y < self.height:
                # Draw a larger point for the agent by filling surrounding cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            frame[ny][nx] = 2

            # Record agent overlay info for this step
            agent_overlay.append((
                agent.position.x, agent.position.y,
                agent.vision_range, agent.comm_range, agent.safety_range, 
                agent.rank, agent.id, agent.direct_sup_id, agent.step_number,
                getattr(agent, 'split_tag', None)  # 新增分流标签
            ))

        self.history.append((frame, agent_overlay))

    def start_simulation(self):
        """Start the simulation - agents will begin moving"""
        self.is_running = True

    def stop_simulation(self):
        """Stop the simulation - agents will stop moving"""
        self.is_running = False

    def run(self, steps=100):
        """Run the simulation for a specified number of steps"""
        self.history = []
        self.is_running = True  # Ensure simulation is running
        for _ in range(steps):
            self.step()
        self.is_running = False  # Stop simulation after steps are complete

    def render_grid_animation(self):
        """Render the current state of the simulation"""
        if not hasattr(self, 'fig') or not hasattr(self, 'ax'):
            # Create figure and axes only if they don't exist
            self.fig, self.ax = plt.subplots(figsize=(10, 10), dpi=100)
            self.cmap = ListedColormap(['white', 'black', '#b58900', 'red', 'green', '#1f3b99'])  # white, black, yellow, red, green, deep blue
            self.im = self.ax.imshow(self.history[-1][0], cmap=self.cmap, vmin=0, vmax=5) # Updated vmax to 5
            
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()
            plt.title("FlexSwarm")
            
            # Initialize lists to store visualization elements
            self.vision_circles = []
            self.comm_circles = []
            self.safety_circles = []
            self.text_labels = []
            self.connection_lines = []
            self.target_star = None  # Add target star to visualization elements
            self.fork_triangle = None  # Add fork triangle to visualization elements
        
        # Clear previous elements
        for patch in self.vision_circles + self.comm_circles + self.safety_circles + self.connection_lines:
            if patch in self.ax.patches:
                patch.remove()
        if self.target_star and self.target_star in self.ax.patches:
            self.target_star.remove()
        if hasattr(self, 'fork_triangle') and self.fork_triangle and self.fork_triangle in self.ax.patches:
            self.fork_triangle.remove()
        for txt in self.text_labels:
            if txt in self.ax.texts:
                txt.remove()
        
        # Clear all lines from the axes
        for line in self.ax.lines:
            line.remove()
        
        self.vision_circles.clear()
        self.comm_circles.clear()
        self.safety_circles.clear()
        self.text_labels.clear()
        self.connection_lines.clear()
        
        # Update the grid
        frame, overlays = self.history[-1]
        self.im.set_array(frame)
        
        # Find target position
        target_pos = None
        fork_pos = None
        for y in range(self.height):
            for x in range(self.width):
                if self.global_map[y][x] == '*':
                    target_pos = (x, y)
                if self.global_map[y][x] == '^':
                    fork_pos = (x, y)
            if target_pos and fork_pos:
                break
        # Draw target as a star if found
        if target_pos:
            x, y = target_pos
            # Create a proper five-pointed star
            # Calculate points for a five-pointed star
            outer_radius = 1.5
            inner_radius = 0.6
            points = []
            for i in range(10):
                angle = math.pi/2 + i * math.pi/5
                radius = outer_radius if i % 2 == 0 else inner_radius
                px = x + radius * math.cos(angle)
                py = y + radius * math.sin(angle)
                points.append((px, py))
            
            star = plt.Polygon(points, fill=True, color='red', alpha=0.8)
            self.ax.add_patch(star)
            self.target_star = star
        # Draw fork as a triangle if found
        if fork_pos:
            x, y = fork_pos
            # 画等边三角形
            triangle_radius = 1.5
            angle_offset = math.pi / 2
            points = []
            for i in range(3):
                angle = angle_offset + i * 2 * math.pi / 3
                px = x + triangle_radius * math.cos(angle)
                py = y + triangle_radius * math.sin(angle)
                points.append((px, py))
            triangle = plt.Polygon(points, fill=True, color='green', alpha=0.8)
            self.ax.add_patch(triangle)
            self.fork_triangle = triangle
        
        # Draw dead end markers as black triangles
        for y in range(self.height):
            for x in range(self.width):
                if self.global_map[y][x] == 'X':
                    # Create black triangle for dead end
                    triangle_radius = 1.5
                    angle_offset = math.pi / 2
                    points = []
                    for i in range(3):
                        angle = angle_offset + i * 2 * math.pi / 3
                        px = x + triangle_radius * math.cos(angle)
                        py = y + triangle_radius * math.sin(angle)
                        points.append((px, py))
                    deadend_triangle = plt.Polygon(points, fill=True, color='black', alpha=0.8)
                    self.ax.add_patch(deadend_triangle)
        
        # Create a dictionary to store agent positions by ID
        agent_positions = {}
        for x, y, vr, cr, sr, rank, agent_id, sup_id, step_num, split_tag in overlays:
            agent_positions[agent_id] = (x, y)
        
        # Draw all elements
        for x, y, vr, cr, sr, rank, agent_id, sup_id, step_num, split_tag in overlays:
            # Get agent's task completion status from comm_map
            task_completed = False
            for msg in self.comm_map[int(y)][int(x)]:
                if msg['type'] == 'status' and msg['id'] == agent_id:
                    task_completed = msg.get('task_completion', 0)
                    break

            # Draw circles with color based on split_tag and task completion
            if split_tag == 'left':
                vis_color = 'cyan'     # Left split
            elif split_tag == 'right':
                vis_color = 'magenta'  # Right split
            elif task_completed == 1:
                vis_color = 'lime'     # Target found
            elif task_completed == 2:
                vis_color = 'green'    # Fork discovered
            elif task_completed == 3:
                vis_color = 'brown'    # Dead end detected
            else:
                vis_color = 'blue'     # Normal operation
            vis = Circle((x, y), vr, fill=True, color=vis_color, alpha=0.05)
            comm = Circle((x, y), cr, fill=False, linestyle='--', color='gray', alpha=0.2)
            safe = Circle((x, y), sr, fill=False, linestyle='-', color='green', alpha=0.3)
            
            for c in [vis, comm, safe]:
                self.ax.add_patch(c)
            
            self.vision_circles.append(vis)
            self.comm_circles.append(comm)
            self.safety_circles.append(safe)
            
            # Draw connection line if there's a superior
            if sup_id >= 0 and sup_id in agent_positions:
                sup_x, sup_y = agent_positions[sup_id]
                line = plt.Line2D([x, sup_x], [y, sup_y], 
                                color='red', alpha=0.5, 
                                linestyle='-', linewidth=1)
                self.ax.add_line(line)
                self.connection_lines.append(line)
            
            # Text: Show ID, rank and step number
            label = self.ax.text(x, y + 0.2, f"ID:{agent_id} R:{rank} S:{step_num}", 
                               fontsize=8, color='black', ha='center', va='bottom', zorder=3)
            self.text_labels.append(label)
        
        # Update the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_rank_config(self, new_config):
        """
        Update the rank configuration during simulation.
        new_config should be a dictionary with rank numbers as keys and max subordinates as values.
        Example: {0: 2, 1: 3, 2: 2}
        """
        print("Environment: Received new config:", new_config)  # Debug print
        # Convert string keys to integers if necessary
        new_rank_config = {int(k): int(v) for k, v in new_config.items()}
        
        # Check if the config has actually changed
        if new_rank_config != self.rank_config:
            self.rank_config = new_rank_config
            self.rank_config_modified = True
            print("Environment: Updated rank_config:", self.rank_config)  # Debug print
            
            # Update environment reference in all agents
            for agent in self.agents:
                agent.environment = self

# Run when this script is executed
if __name__ == '__main__':
    env = Environment()
    env.run(steps=200)               # Run the full simulation
    env.render_grid_animation()      # Then visualize the recorded history