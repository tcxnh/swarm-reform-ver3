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
            def add_barrier(x1, y1, x2, y2):
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        if 0 <= x < self.width and 0 <= y < self.height:
                            global_map[y][x] = '#'
                            wall_mask[y][x] = 1

            # Define maze patterns
            maze_patterns = [
                lambda: [
                    # ===== 外边框（厚度5）=====
                    (0, 0, 150, 5),  # 上墙
                    (0, 0, 5, 150),  # 左墙
                    (145, 0, 150, 150),  # 右墙

                    # ===== Z字形主干道（厚度5）=====
                    # # 第一横（从左到右）
                    # (5, 40, 50, 45),
                    # # 第一斜（向下）
                    # (45, 45, 50, 90),
                    # 第二横（从右到左）
                    (120, 90, 145, 100),
                    # 第三横（从左到右）
                    (95, 0, 145, 45),

                    # ===== 大块洞穴障碍物 =====
                    # 左上区域（不规则大块）
                    (0, 0, 40, 35),
                    (20, 35, 45, 70),
                    # 中间区域（阻挡Z字路径）
                    (60, 70, 80, 120),
                    # 右下区域（缩小后的岩石群）
                    # (110, 110, 130, 130),  # 主岩石缩小
                    # (120, 90, 140, 110),   # 新增小岩石
                    # (80, 10, 120, 40),
                    # 左下入口区域（洞穴结构）
                    (5, 120, 30, 150),
                    # 右上出口区域（狭窄通道）
                    (120, 140, 140, 150),

                    # ===== 入口和出口（厚度5）=====
                    # 入口（左下，从洞穴进入）
                    (0, 125, 5, 130),
                    # 出口（右上，狭窄通道）
                    (120,140,140,150)
                ]
            ]

            # Randomly select and apply one maze pattern
            selected_maze = random.choice(maze_patterns)
            for x1, y1, x2, y2 in selected_maze():
                add_barrier(x1, y1, x2, y2)

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

        return global_map, wall_mask

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
                agent.rank, agent.id, agent.direct_sup_id, agent.step_number
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
            self.cmap = ListedColormap(['white', 'black', '#b58900', 'red'])  # white, black, yellow, red
            self.im = self.ax.imshow(self.history[-1][0], cmap=self.cmap, vmin=0, vmax=4) # Updated vmax to 4
            
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
        
        # Create a dictionary to store agent positions by ID
        agent_positions = {}
        for x, y, vr, cr, sr, rank, agent_id, sup_id, step_num in overlays:
            agent_positions[agent_id] = (x, y)
        
        # Draw all elements
        for x, y, vr, cr, sr, rank, agent_id, sup_id, step_num in overlays:
            # Get agent's task completion status from comm_map
            task_completed = False
            for msg in self.comm_map[int(y)][int(x)]:
                if msg['type'] == 'status' and msg['id'] == agent_id:
                    task_completed = msg.get('task_completion', 0) == 1
                    break

            # Draw circles with color based on task completion
            vis = Circle((x, y), vr, fill=True, 
                        color='red' if task_completed else 'blue', 
                        alpha=0.05)  # Reduced alpha from 0.1 to 0.05
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