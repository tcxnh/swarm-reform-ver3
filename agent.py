import math
from position_class import Position
import random
import json

class Agent:

   
    def __init__(self, position: Position, agent_id: int,
                 vision_range: float, comm_range: float, is_entrance=False, function_mode="flex-search"):
        """
        Initialize the agent's attributes and information.
        """
        # --- Attributes ---
        self.id = agent_id
        self.vision_range = vision_range
        self.comm_range = comm_range
        self.safety_range = (comm_range+vision_range)/2
        # Attributes only for simulation
        self.position = position 
        self.speed = 1 # the speed must >=1 in the simulation
        
        # Attributes for rank based swarm algorithm 
        self.function_mode = function_mode
        self.is_entrance = is_entrance
        self.rank = 0 if is_entrance else 1000 # Starts at 0; dynamic update 
        self.step_number = 0  # Initialize step_number
        self.direct_sub_id = []  # List of direct subordinate agent IDs 
        self.direct_sup_id = -1 # When not determine direct superior, assign as -1

        # --- Information ---
        self.perceived_environment = [] # information from environment
        self.received_messages = [] # information from nearby agents
        self.task_completion_index = 0  # initial as not completed
        # Dead end tracking for smart flood search
        self.deadend_detected = False  # Flag to track if agent detected a dead end
        self.known_deadend_areas = set()  # Set of (x, y) coordinates known to be dead end areas
        # Direction tracking for fork-based navigation
        self.fork_direction = None  # Which direction this agent is relative to fork ('left', 'right', 'center')
        self.deadend_direction = None  # Which direction leads to dead end
        self.blocked_directions = set()  # Set of directions known to be blocked/dead ends



    def _get_max_subordinates(self):
        """
        Get the maximum number of subordinates based on rank or custom override.
        """
        try:
            # Get rank_config from environment if available
            if hasattr(self, 'environment') and hasattr(self.environment, 'rank_config'):
                rank_config = self.environment.rank_config
            else:
                # Fallback to reading from file
                with open('swarm_config.json', 'r') as f:
                    config = json.load(f)
                    rank_config = config.get('rank_config', {})
                    # Convert string keys to integers
                    rank_config = {int(k): int(v) for k, v in rank_config.items()}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Default configuration if file not found or invalid
            rank_config = {
                0: 1,  # Entrance can manage 1 subordinate
                1: 1,  # Rank 1 can manage 1 subordinate
                2: 1,  # Rank 2 can manage 1 subordinate
                3: 1   # Rank 3 can manage 1 subordinate
            }
        
        # For entrance, use rank 0 configuration
        if self.is_entrance:
            return rank_config.get(0, 1)
        # For other agents, use their rank configuration, default to 1 if rank not found
        return rank_config.get(self.rank, 1)

    # =====================
    # General Behavior Rule
    # =====================
    def _general_behavior(self, global_map,comm_map):
        """
        General decentralized behavior:
        1. Perceive environment
        2. Broadcast information (in simulation, include two parts:remove and broadcast)
        3. Receive information from local communication
        4. Update function mode
        5. Network self-healing and dynamic topology optimization
        """
        # Update current step for tracking
        if not hasattr(self, 'current_step'):
            self.current_step = 0
        self.current_step += 1
        
        self._update_perceived_environment(global_map)
        self._receive_from_comm_map(comm_map, global_map)
        self._remove_old_broadcast(comm_map) # simulation only
        self._update_function_mode()
        


    def _update_perceived_environment(self, global_map):
        """
        Store the location of nearby walls relative to the agent.
        """
        self.perceived_environment = []
        x0, y0 = int(self.position.x), int(self.position.y)
        vision_radius = int(self.vision_range)

        for dx in range(-vision_radius, vision_radius + 1):
            for dy in range(-vision_radius, vision_radius + 1):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < len(global_map[0]) and 0 <= y < len(global_map):
                    if global_map[y][x] == '#':
                        self.perceived_environment.append({
                            'type': 'wall',
                            'relative_position': (dx, dy)
                        })
                    
                    # targeted object find
                    if global_map[y][x] == '*':
                        self.perceived_environment.append({
                            'type': 'target',
                            'relative_position': (dx, dy)
                        })
                        self.task_completion_index = 1 
                    # fork object find
                    if global_map[y][x] == '^':
                        self.perceived_environment.append({
                            'type': 'fork',
                            'relative_position': (dx, dy)
                        })
                    # dead end object find
                    if global_map[y][x] == 'X':
                        self.perceived_environment.append({
                            'type': 'deadend',
                            'relative_position': (dx, dy)
                        })
                else:
                    # Treat out-of-bounds as wall
                    self.perceived_environment.append({
                        'type': 'wall',
                        'relative_position': (dx, dy)
                    })
 
    def _check_line_of_sight(self, target_x, target_y, global_map):
        """
        Check if there is a clear line of sight between current position and target position.
        Returns True if there is no obstacle between the two points.
        """
        x0, y0 = int(self.position.x), int(self.position.y)
        x1, y1 = int(target_x), int(target_y)
        
        # Get all points along the line using Bresenham's line algorithm
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            points.append((x, y))
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        # Check if any point along the line is a wall
        for x, y in points:
            if 0 <= x < len(global_map[0]) and 0 <= y < len(global_map):
                if global_map[y][x] == '#':
                    return False
            else:
                return False  # Out of bounds is considered blocked
        return True

    def _broadcast_to_comm_map(self, comm_map, global_map): # broadcast the information after the agent move 
        """
        Broadcast the agent's information to the comm_map grid cells within communication range.
        Only broadcast to cells that have a clear line of sight.
        """
        radius = int(self.comm_range)
        x0, y0 = int(self.position.x), int(self.position.y)

        message = {
            'type': 'status',
            'id': self.id,
            'rank': self.rank,
            'task_completion': self.task_completion_index,
            'function_mode': self.function_mode,
            'position': self.position.copy(),  # just used for simulation
            'direct_sub_id': self.direct_sub_id.copy(),
            'direct_sup_id': self.direct_sup_id,
            # --- dead end tracking ---
            'deadend_detected': self.deadend_detected,
            'known_deadend_areas': list(self.known_deadend_areas),  # Convert set to list for message
            # --- direction tracking for fork-based navigation ---
            'fork_direction': self.fork_direction,
            'deadend_direction': self.deadend_direction,
            'blocked_directions': list(self.blocked_directions),
        }
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < len(comm_map[0]) and 0 <= y < len(comm_map):
                    if self._check_line_of_sight(x, y, global_map):
                        comm_map[y][x].append(message.copy())

    def _remove_old_broadcast(self, comm_map): # remove the broadcast before the agent move 
        radius = int(self.comm_range)
        x0, y0 = int(self.position.x), int(self.position.y)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < len(comm_map[0]) and 0 <= y < len(comm_map):
                    comm_map[y][x] = [msg for msg in comm_map[y][x] if msg.get('id') != self.id]

    def _receive_from_comm_map(self, comm_map, global_map):
        """
        Receive messages from the agent's current cell,
        only from agents that have a clear line of sight.
        Categorize surrounding agents into superiors, peers, and subordinates,
        and propagate task completion status.
        """
        x, y = int(self.position.x), int(self.position.y)
        self.received_messages = []    # Clear old messages
        self.subordinates = []         # Messages from agents with larger rank 
        self.superiors = []            # Messages from agents with lower rank
        self.peers = []                # Messages from agents with same rank
        self.assigned_direction = None  # 每步重置，防止旧信号残留

        for msg in comm_map[y][x]:
            if msg['id'] == self.id:
                continue

            # 分流过滤：只接收同一路的消息
            if msg.get('split_tag') is not None: # 移除 split_tag 过滤
                continue

            # Check if there is a clear line of sight to the sender
            if not self._check_line_of_sight(msg['position'].x, msg['position'].y, global_map):
                continue



            self.received_messages.append(msg)

            if msg['type'] != 'status':
                continue

            # If any agent has found the target, set our task_completion to 1
            if msg.get('task_completion', 0) == 1:
                self.task_completion_index = 1

            # Process dead end information from other agents
            if msg.get('deadend_detected', False):
                self.deadend_detected = True
            # Update known dead end areas
            if 'known_deadend_areas' in msg:
                for deadend_pos in msg['known_deadend_areas']:
                    if isinstance(deadend_pos, (list, tuple)) and len(deadend_pos) == 2:
                        self.known_deadend_areas.add(tuple(deadend_pos))

            # Process direction information for fork-based navigation
            if msg.get('deadend_direction') and msg['deadend_direction'] != 'unknown':
                self.blocked_directions.add(msg['deadend_direction'])
                print(f"Agent {self.id} learned that {msg['deadend_direction']} direction is blocked from agent {msg['id']}")
            
            # Update blocked directions
            if 'blocked_directions' in msg:
                for blocked_dir in msg['blocked_directions']:
                    if blocked_dir and blocked_dir != 'unknown':
                        self.blocked_directions.add(blocked_dir)

            # fork leader分流信号
            # (Removed fork leader and assigned_direction logic)

            # categorize
            if msg['rank'] < self.rank and msg['rank'] >= 0:
                self.superiors.append(msg)
            elif msg['rank'] == self.rank and msg['rank'] >= 0:
                self.peers.append(msg)
            elif msg['rank'] > self.rank and msg['rank'] >= 0: 
                self.subordinates.append(msg)
                
        self.subordinates.sort(
            key=lambda m: math.sqrt(
               (self.position.x - m['position'].x) ** 2 +
               (self.position.y - m['position'].y) ** 2
            )
        )       

    def _update_function_mode(self):
        """
        Adopt the function mode of lowest-rank agent within communication range.
        Only updates if target is not found and we're in search mode.
        """
        if self.task_completion_index == 1:  # If target found, don't update function mode
            return
            
        # Only update mode if we're in a search mode
        if self.function_mode not in ["flex-search", "flood-search"]:
            return
            
        lowest_rank = float('inf')
        for msg in self.received_messages:
            if msg['type'] == 'status' and msg['rank'] < self.rank and msg['rank'] >= 0:
                if msg['rank'] < lowest_rank:
                    lowest_rank = msg['rank']
                    self.function_mode = msg['function_mode']
    
    # =============================
    # Function-mode Specific Behavior
    # =============================
    def _mode_based_behavior(self):
        if self.task_completion_index == 1:  # Target found
            self._update_rank_return()
            self._move_return()
        elif self.function_mode == "flex-search":
            self._update_rank_search()
            self._move_rank_search()
        elif self.function_mode == "flood-search":
            self._update_flood_search()
            self._move_flood_search()
        elif self.function_mode == "flood-evol-search":
            self._update_flood_evol_search()
            self._move_flood_evol_search()
        else:
            print(f"Unknown function mode: {self.function_mode}")

    def _determine_direction_from_fork(self, fork_relative_pos):
        """
        Determine which direction this agent is relative to the fork.
        Returns 'left', 'right', or 'center' based on relative position.
        """
        dx, dy = fork_relative_pos
        
        # If fork is very close, consider as center
        if abs(dx) <= 2 and abs(dy) <= 2:
            return 'center'
        
        # Determine direction based on relative position
        # Positive dx means fork is to the right of agent, so agent is on left side
        # Negative dx means fork is to the left of agent, so agent is on right side
        if dx > 2:  # Fork is to the right, agent is on left branch
            return 'left'
        elif dx < -2:  # Fork is to the left, agent is on right branch  
            return 'right'
        else:
            # Close to center line, use y-coordinate to determine
            if dy > 0:  # Fork is below, agent is approaching from above
                return 'center'
            else:
                return 'center'

    def _update_flood_evol_search(self):
        """
        Enhanced flood search that can detect dead ends and avoid them.
        - Similar to flood-search, but with smart dead end avoidance
        - If fork is found in vision, propagate like target and enter return
        - If dead end is found, mark area as blocked and signal back to other agents
        - Avoid moving towards known dead end areas
        """
        # First sync messages for target completion only (not dead end)
        for msg in self.received_messages:
            if msg.get('task_completion', 0) == 1:  # Only target found (1)
                self.task_completion_index = msg['task_completion']

        # Check own vision for fork discovery
        fork_found = False
        for info in self.perceived_environment:
            if info['type'] == 'fork':
                fork_found = True
                
                # Determine and store our direction relative to fork
                self.fork_direction = self._determine_direction_from_fork(info['relative_position'])
                print(f"Agent {self.id} discovered fork, positioned on {self.fork_direction} side")
                # No need to pause or set task_completion_index = 2
                break

        # Check own vision for dead end discovery
        deadend_found = False
        for info in self.perceived_environment:
            if info['type'] == 'deadend':
                deadend_found = True
                # Mark the dead end location in our known areas
                abs_x = int(self.position.x + info['relative_position'][0])
                abs_y = int(self.position.y + info['relative_position'][1])
                self.known_deadend_areas.add((abs_x, abs_y))
                
                # Also mark the surrounding area as potentially blocked
                for dx in range(-3, 4):  # Mark a larger area around dead end
                    for dy in range(-3, 4):
                        area_x, area_y = abs_x + dx, abs_y + dy
                        self.known_deadend_areas.add((area_x, area_y))
                
                # CRITICAL: Report which direction leads to dead end
                if hasattr(self, 'fork_direction'):
                    self.deadend_direction = self.fork_direction
                    print(f"Agent {self.id} detected dead end in {self.fork_direction} branch!")
                else:
                    # Try to infer direction from known fork agents
                    self.deadend_direction = self._infer_deadend_direction()
                break

        # If dead end found, set task completion to signal back to other agents
        if deadend_found:
            self.deadend_detected = True
            self.task_completion_index = 3  # 3 = dead end discovered (blocks movement)
            return

        # If in a dead end area (detected by self or others), try to move out
        current_pos = (int(self.position.x), int(self.position.y))
        if current_pos in self.known_deadend_areas:
            # Move away from dead end areas - this will be handled in movement
            pass

        # Otherwise continue normal flood search behavior
        if self.task_completion_index == 0:
            self._update_flood_search()

    def _infer_deadend_direction(self):
        """
        Try to infer which direction the dead end is in based on communication with other agents.
        """
        # Look for agents that have discovered fork and determine our relative position
        for msg in self.received_messages:
            if msg.get('task_completion', 0) == 2:  # Fork discovered
                # Calculate relative position to this fork-discovering agent
                dx = self.position.x - msg['position'].x
                dy = self.position.y - msg['position'].y
                
                if abs(dx) > 5:  # Significant horizontal separation
                    return 'left' if dx < 0 else 'right'
        
        return 'unknown'

    def _move_flood_evol_search(self):
        """
        Enhanced flood search movement that avoids dead end areas.
        PRIORITY 1: Safety Rule - maintain connectivity (highest priority)
        PRIORITY 2: Dead end avoidance
        PRIORITY 3: Normal flood search
        """
        # If dead end discovered, stop moving to signal back to other agents
        if self.task_completion_index == 3:
            return

        # PRIORITY 1: Safety Rule - ensure connectivity FIRST (overrides dead end avoidance)
        if not self.is_entrance:
            direct_superior = self._get_direct_superior()
            if self._handle_safety_rule_for_flood_evol(direct_superior):
                return  # Connectivity takes precedence over everything else

        # PRIORITY 2: Check if currently in a dead end area
        current_pos = (int(self.position.x), int(self.position.y))
        if current_pos in self.known_deadend_areas:
            # Try to escape from dead end area towards lower rank agents
            self._move_escape_deadend()
            return

        # PRIORITY 3: Normal flood search but with dead end avoidance
        self._move_flood_search_with_deadend_avoidance()

    def _handle_safety_rule_for_flood_evol(self, direct_superior):
        """
        Handle Safety Rule for flood-evol search (flood search style): maintain connectivity.
        This is the HIGHEST PRIORITY - if no agents in vision range, stop moving to maintain connectivity.
        Returns True if safety rule activated (no movement), False to continue to next priority.
        """
        # Get agents in vision range
        visible_agents = []
        for msg in self.received_messages:
            if msg['type'] != 'status':
                continue

            dist = self._calculate_distance(msg)
            if dist <= self.vision_range:
                visible_agents.append(msg)

        # FLOOD SEARCH SAFETY RULE: If no agents in vision range, stop moving to maintain connectivity
        if not visible_agents:
            print(f"Agent {self.id} maintaining connectivity - no agents in vision range, staying put")
            return True  # Safety rule activated, stop movement
        
        return False  # Safety rule not activated, continue to next priority

    def _move_escape_deadend(self):
        """
        Escape movement when agent is in a dead end area.
        Move towards the lowest rank agent that is NOT in a dead end area.
        """
        if self.is_entrance:
            return

        # Find agents not in dead end areas, prioritize lower rank
        escape_targets = []
        for msg in self.received_messages:
            if msg['type'] != 'status':
                continue
            
            msg_pos = (int(msg['position'].x), int(msg['position'].y))
            # Only consider agents not in dead end areas
            if msg_pos not in self.known_deadend_areas:
                dist = self._calculate_distance(msg)
                if dist <= self.vision_range:
                    escape_targets.append((msg, dist))

        if not escape_targets:
            # No escape targets visible, try random movement away from dead end center
            self._move_random_escape()
            return

        # Move towards the lowest rank agent not in dead end area
        escape_targets.sort(key=lambda x: (x[0]['rank'], x[1]))  # Sort by rank first, then distance
        target_msg = escape_targets[0][0]
        
        # Calculate movement towards escape target
        dx = target_msg['position'].x - self.position.x
        dy = target_msg['position'].y - self.position.y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            move_x = round((dx / dist) * self.speed)
            move_y = round((dy / dist) * self.speed)
            self._attempt_move(move_x, move_y)

    def _move_random_escape(self):
        """
        Random movement to escape when no clear escape path is visible.
        """
        import random
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            new_pos = (int(self.position.x + dx), int(self.position.y + dy))
            if new_pos not in self.known_deadend_areas:
                self._attempt_move(dx, dy)
                break

    def _move_flood_search_with_deadend_avoidance(self):
        """
        Normal flood search movement but avoid known dead end areas and blocked directions.
        """
        if self.is_entrance:
            return

        # Get agents in vision range, excluding those in dead end areas
        visible_agents = []
        for msg in self.received_messages:
            if msg['type'] != 'status':
                continue

            dist = self._calculate_distance(msg)
            if dist <= self.vision_range:
                msg_pos = (int(msg['position'].x), int(msg['position'].y))
                # Include agents not in dead end areas, or agents with significantly lower rank
                if msg_pos not in self.known_deadend_areas or msg['rank'] < self.rank - 2:
                    visible_agents.append(msg)

        # Stop if no suitable agents in vision range
        if not visible_agents:
            return

        # Find the lowest rank agent in vision range to repel from
        lowest_rank_agent = min(visible_agents, key=lambda x: x['rank'])
        
        # Calculate repulsion vector but avoid dead end areas and blocked directions
        dx = self.position.x - lowest_rank_agent['position'].x
        dy = self.position.y - lowest_rank_agent['position'].y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            # Normalize movement direction
            move_x = round((dx / dist) * self.speed)
            move_y = round((dy / dist) * self.speed)
            
            # Check if movement would lead to dead end area
            new_pos = (int(self.position.x + move_x), int(self.position.y + move_y))
            if new_pos in self.known_deadend_areas:
                # Try alternative directions that avoid dead end areas
                alternative_moves = [
                    (move_x, 0), (0, move_y), (-move_x, 0), (0, -move_y),
                    (move_x, move_y//2), (move_x//2, move_y)
                ]
                for alt_x, alt_y in alternative_moves:
                    alt_pos = (int(self.position.x + alt_x), int(self.position.y + alt_y))
                    if alt_pos not in self.known_deadend_areas:
                        # Additional check: avoid blocked directions if near fork
                        if not self._would_enter_blocked_direction(alt_x, alt_y):
                            self._attempt_move(alt_x, alt_y)
                            return
                # If all alternatives lead to dead end, don't move
                return
            else:
                # Check if this movement would enter a blocked direction
                if self._would_enter_blocked_direction(move_x, move_y):
                    # Try alternative directions that avoid blocked directions
                    alternative_moves = [
                        (0, move_y), (move_x, 0), (-move_x, 0), (0, -move_y),
                        (move_x//2, move_y), (move_x, move_y//2)
                    ]
                    for alt_x, alt_y in alternative_moves:
                        if not self._would_enter_blocked_direction(alt_x, alt_y):
                            alt_pos = (int(self.position.x + alt_x), int(self.position.y + alt_y))
                            if alt_pos not in self.known_deadend_areas:
                                self._attempt_move(alt_x, alt_y)
                                return
                    # If no good alternative, stay put
                    print(f"Agent {self.id} avoiding blocked direction, staying put")
                    return
                else:
                    self._attempt_move(move_x, move_y)

    def _would_enter_blocked_direction(self, move_x, move_y):
        """
        Check if a movement would lead the agent into a direction known to be blocked.
        Enhanced to handle Y-shaped paths with diagonal directions.
        """
        if not self.blocked_directions:
            return False
        
        # Check if we are near fork area (only then should we care about directions)
        near_fork = (
            self.fork_direction is not None or  # We discovered a fork ourselves
            any(msg.get('fork_direction') for msg in self.received_messages) or  # Others in comm range discovered fork
            self._can_see_fork_in_vision()  # We can see fork marker in our vision
        )
        
        if not near_fork:
            return False  # Only apply direction blocking when near fork area
        
        # Determine the general direction we would be moving towards
        # Handle both simple left/right and more complex Y-path directions
        move_directions = set()
        
        # Horizontal component
        if move_x > 1:
            move_directions.add('right')
        elif move_x < -1:
            move_directions.add('left')
            
        # Vertical component (for Y-shaped paths)
        if move_y < -1:  # Moving upward (towards fork branches)
            move_directions.add('up')
        elif move_y > 1:  # Moving downward (towards entrance)
            move_directions.add('down')
            
        # Diagonal combinations for Y-paths
        if move_x > 1 and move_y < -1:
            move_directions.add('right-up')
        elif move_x < -1 and move_y < -1:
            move_directions.add('left-up')
        
        # Check if any of the movement directions are blocked
        for direction in move_directions:
            if direction in self.blocked_directions:
                print(f"Agent {self.id} avoiding movement towards blocked {direction} direction")
                return True
        
        return False


    def _can_see_fork_in_vision(self):
        """
        Check if agent can see fork marker in its vision range.
        Only agents near fork should care about direction blocking.
        """
        for info in self.perceived_environment:
            if info['type'] == 'fork':
                return True
        return False
    
    # ---------- Rank-based Search Behavior ---------
    def _update_rank_search(self):
        # general steps: decide rank; decide direct_sup_id; decide direct_sub_id
        # Entrance never change rank and always considered as linked
        
        if self.is_entrance:
            self.rank = 0
            self.direct_sup_id = 0 # superior is itself
            # Update direct_sub_id using valid existing ones and add new if needed
            current_ids = set(self.direct_sub_id)
            valid_existing = [
                sub for sub in self.subordinates
                if sub['id'] in current_ids
            ]

            # 使用当前rank对应的max_subordinates
            current_max = self._get_max_subordinates()
            if len(valid_existing) >= current_max:
                self.direct_sub_id = [sub['id'] for sub in valid_existing[:current_max]]
                return

            additional = []
            for sub in self.subordinates:
                if sub['id'] not in current_ids and sub.get('direct_sup_id') == -1:
                    additional.append(sub)
                if len(valid_existing) + len(additional) >= current_max:
                    break

            self.direct_sub_id = [sub['id'] for sub in valid_existing + additional]
            return

        # ---- Non-entrance agents ----
        # situation 1: if superior has manage slots or the agent is in the superior's direct_sub_id 
        candidates = []
        for msg in self.superiors:
            if msg['direct_sup_id'] >= 0: # only consider connected superior
                sub_ids = msg.get('direct_sub_id', [])
                current_max = self._get_max_subordinates()  # 获取当前rank对应的max_subordinates
                if len(sub_ids) < current_max or self.id in sub_ids:
                    candidates.append(msg)

        if candidates:
            best_superior = min(
                candidates,
                key=lambda m: (m['rank'])
            )
            # update rank 
            self.rank = best_superior['rank'] + 1
            
            # update direct_sup_id
            self.direct_sup_id = -1
            for msg in candidates:
                sub_ids = msg.get('direct_sub_id', [])
                if self.id in sub_ids and self.rank == msg.get('rank') + 1: # this is for double check
                    self.direct_sup_id = msg['id']

            # update direct_sub_id
            if self.direct_sup_id >= 0:
                current_ids = set(self.direct_sub_id)
                valid_existing = [
                    sub for sub in self.subordinates
                    if sub['id'] in current_ids and sub.get('direct_sup_id', -1) == self.id
                ]

                # 使用当前rank对应的max_subordinates
                current_max = self._get_max_subordinates()
                # If already have max subordinates, just keep first max_subordinates
                if len(valid_existing) >= current_max:
                    self.direct_sub_id = [sub['id'] for sub in valid_existing[:current_max]]
                    return

                # Otherwise, fill the rest with new closest ones not already included
                additional = []
                for sub in self.subordinates:
                    if sub['id'] not in current_ids and sub['direct_sup_id'] == -1:
                        additional.append(sub)
                    if len(valid_existing) + len(additional) >= current_max:
                        break

                self.direct_sub_id = [sub['id'] for sub in valid_existing + additional]
                return
            else:
                self.direct_sub_id=[]
                return
        
        # Situation 2: no superior has slot, and not in any of the superior's direct_sub_id
        # in this situation, as it does not determine direct_sup_id, it will not allow to have direct_sub_id
        # if no superior has rank == self.rank - 1 (expected manager)
        expected_managers = [
            msg for msg in self.superiors
            if msg['rank'] == self.rank - 1 and msg.get('direct_sup_id', -1) >= 0
        ]

        if not expected_managers:
            # Only consider superiors who themselves are linked to a superior
            connected_superiors = [
                msg for msg in self.superiors
                if msg.get('direct_sup_id', -1) >= 0
            ]
            
            if connected_superiors:  # avoid crash if list is empty
                max_rank = max(msg['rank'] for msg in connected_superiors)
                self.rank = max_rank + 1
                self.direct_sup_id = -1
                self.direct_sub_id = []
            else:
                self.direct_sup_id = -1
                self.direct_sub_id = []
            return
        
        # if has expected manager but not accepted by any superior, check if peers are in any superior's sub list
        if any(peer.get('direct_sup_id', 0) >= 0 for peer in self.peers):
            self.rank += 1
            self.direct_sup_id = -1
            self.direct_sub_id = []
            return

        #If none of the above 
        self.direct_sup_id = -1
        self.direct_sub_id = []
        #self.rank = 1000  # 设置一个明显的高等级

    def _move_rank_search(self):
        """
        Rank-based movement with three-stage behavior:
        - Safety Rule: Maintain link to direct superior, ensure connectivity.
        - Rank Management: Move toward superior if under- or over-managed.
        - Stage-Aware Repulsion: Avoid conflicts with non-direct superiors and peers.
        """
        if self.is_entrance:
            self.step_number = 0
            return

        # 获取direct superior信息
        direct_superior = self._get_direct_superior()
        
        # Stage 1: Safety Rule - 确保与上级的连接
        if self._handle_safety_rule(direct_superior):
            return
        
        # Stage 2: Rank Management - 处理管理关系
        if self._handle_rank_management(direct_superior):
            return
        
        # Stage 3: Stage-Aware Repulsion - 避免冲突
        self._handle_stage_aware_repulsion(direct_superior)

    def _get_direct_superior(self):
        """Get the message of direct superior agent"""
        if self.direct_sup_id == -1:
            return None
        return next((msg for msg in self.superiors if msg['id'] == self.direct_sup_id), None)

    def _calculate_distance(self, target_msg):
        """Calculate Euclidean distance to target agent"""
        if target_msg is None:
            return float('inf')
        return math.hypot(
            self.position.x - target_msg['position'].x,
            self.position.y - target_msg['position'].y
        )

    def _calculate_move_vector(self, target_msg, attraction=True):
        """
        Calculate movement vector towards or away from target
        attraction=True: move toward target
        attraction=False: move away from target
        """
        if target_msg is None:
            return 0, 0
        
        dx = self.position.x - target_msg['position'].x
        dy = self.position.y - target_msg['position'].y
        dist = math.hypot(dx, dy)
        
        if dist < 1e-6:
            return 0, 0
        
        factor = -1 if attraction else 1
        move_x = round((factor * dx / dist) * self.speed)
        move_y = round((factor * dy / dist) * self.speed)
        
        return move_x, move_y

    def _handle_safety_rule(self, direct_superior):
        """
        Handle Stage 1: Safety Rule - ensure connection to direct superior
        Returns True if movement is executed, False to continue to next stage
        """
        if direct_superior:
            dist = self._calculate_distance(direct_superior)
            if dist > self.safety_range:
                # Move toward direct superior to reconnect
                move_x, move_y = self._calculate_move_vector(direct_superior, attraction=True)
                self._attempt_move(move_x, move_y)
                return True
        else:
            # When no direct superior, first try to find any superior in communication range
            comm_superiors = [s for s in self.superiors if self._calculate_distance(s) <= self.comm_range]
            if comm_superiors:
                nearest_superior = min(comm_superiors, key=self._calculate_distance)
                dist = self._calculate_distance(nearest_superior)
                if dist > self.safety_range:
                    move_x, move_y = self._calculate_move_vector(nearest_superior, attraction=True)
                    self._attempt_move(move_x, move_y)
                    return True
            
            # If no superior in comm range or all superiors are too far, try any agent with lower rank
            lower_rank_agents = [
                msg for msg in self.received_messages
                if msg['type'] == 'status' and msg['rank'] < self.rank
            ]

            if lower_rank_agents:
                nearest_lower_rank = min(lower_rank_agents, key=self._calculate_distance)
                dist = self._calculate_distance(nearest_lower_rank)
                if dist > self.safety_range:
                    move_x, move_y = self._calculate_move_vector(nearest_lower_rank, attraction=True)
                    self._attempt_move(move_x, move_y)
                    return True
            else: 
                # no agent can chase try to find its own way 
                return False
            # # If no agents with lower rank in comm range or all are too far, try any agent in comm range
            # all_agents = self.superiors + self.peers + self.subordinates
            # comm_agents = [a for a in all_agents if self._calculate_distance(a) <= self.comm_range]
            # if comm_agents:
            #     nearest_agent = min(comm_agents, key=self._calculate_distance)
            #     dist = self._calculate_distance(nearest_agent)
            #     if dist > self.safety_range:
            #         move_x, move_y = self._calculate_move_vector(nearest_agent, attraction=True)
            #         self._attempt_move(move_x, move_y)
            #         return True
            #     else:
            #         # Agent is within safety range, proceed to rank management
            #         return False
            # else:
            #     # If no agents found in comm range at all
            #     return True
        
        return False

    def _handle_rank_management(self, direct_superior):
        """
        Handle Stage 2: Rank Management - move if superior is under/over-managed
        Only uses information from communication range (self.subordinates and direct_superior)
        Returns True if movement is executed, False to continue to next stage
        """
        # Only use information from communication range
        if direct_superior:  # direct_superior is from comm range
            sub_count = len(direct_superior.get('direct_sub_id', []))
            if sub_count > self._get_max_subordinates():
                # Over-managed, try to approach one of the superior's direct subordinates
                # Only consider subordinates that are in our communication range
                for sub_id in direct_superior.get('direct_sub_id', []):
                    sub_msg = next((msg for msg in self.subordinates if msg['id'] == sub_id), None)
                    if sub_msg:  # sub_msg is from comm range
                        move_x, move_y = self._calculate_move_vector(sub_msg, attraction=True)
                        self._attempt_move(move_x, move_y)
                        self.step_number = 2
                        return True
        else:
            # Find all agents in communication range that have direct superiors
            agents_with_superior = [
                msg for msg in self.received_messages
                if msg['type'] == 'status' and msg.get('direct_sup_id', -1) >= 0
            ]
            
            if agents_with_superior:
                # Sort agents by rank to find the one with highest rank
                agents_with_superior.sort(key=lambda x: x['rank'])
                highest_rank_agent = agents_with_superior[-1]
                
                # First check if we can see any of the lowest rank agent's direct subordinates
                visible_subordinates = []
                if highest_rank_agent.get('direct_sub_id'):
                    for sub_id in highest_rank_agent['direct_sub_id']:
                        sub_msg = next((msg for msg in self.received_messages 
                                      if msg['type'] == 'status' and msg['id'] == sub_id), None)
                        if sub_msg:
                            visible_subordinates.append(sub_msg)
                
                # If we can see any subordinates, approach the closest one
                if visible_subordinates:
                    print(f"id {self.id}")
                    # print("If we can see any subordinates, approach the closest one")
                    closest_sub = min(visible_subordinates, 
                                   key=lambda x: self._calculate_distance(x))
                    move_x, move_y = self._calculate_move_vector(closest_sub, attraction=True)
                    self._attempt_move(move_x, move_y)
                    self.step_number = 2
                    return True
                else:
                    # If we can't see any subordinates, approach the lowest rank agent
                    # print("If we can't see any subordinates, approach the lowest rank agent")
                    move_x, move_y = self._calculate_move_vector(highest_rank_agent, attraction=True)
                    self._attempt_move(move_x, move_y)
                    self.step_number = 2
                    return True
        
        return False

    def _handle_stage_aware_repulsion(self, direct_superior):
        """Handle Stage 3: Stage-Aware Repulsion to avoid conflicts"""
        non_direct_superiors = [msg for msg in self.superiors if msg['id'] != self.direct_sup_id]
        
        # Find nearest non-direct superior
        nearest_nds = self._get_nearest_in_range(non_direct_superiors, self.safety_range)
        nearest_peer = self._get_nearest_agent(self.peers)
        
        # Step 1: non-direct superior in vision range
        if nearest_nds and self._calculate_distance(nearest_nds) <= self.vision_range:
            self._handle_repulsion_step1(nearest_nds, direct_superior)
            return
        
        # Step 2: non-direct superior in safety range
        if nearest_nds and self._calculate_distance(nearest_nds) <= self.safety_range:
            move_x, move_y = self._calculate_move_vector(nearest_nds, attraction=False)
            self._attempt_move(move_x, move_y)
            self.step_number = 4
            return
        
        # Step 3: handle direct superior and peer repulsion
        self._handle_repulsion_step3(direct_superior, nearest_peer)

    def _handle_repulsion_step1(self, nearest_nds, direct_superior):
        """Handle Stage 3 Step 1 logic for non-direct superior in vision"""
        move_x, move_y = self._calculate_move_vector(nearest_nds, attraction=False)
        self._attempt_move(move_x, move_y)
        self.step_number = 3

    def _handle_repulsion_step3(self, direct_superior, nearest_peer):
        """Handle Stage 3 Step 3 logic for direct superior and peer repulsion"""
        # 首先检查是否存在direct superior
        if not direct_superior:
            self.step_number = 0  # 如果没有direct superior，重置step_number
            return
        
        # 检查direct superior是否在安全范围内
        if self._calculate_distance(direct_superior) > self.safety_range:
            return
        
        # Check for visible agents
        visible_agents = self._get_visible_agents()
        if not visible_agents:
            self.step_number = 0
            return
        
        # Decide movement direction
        if self._calculate_distance(direct_superior) <= self.vision_range:
            move_x, move_y = self._calculate_move_vector(direct_superior, attraction=False)
        elif nearest_peer and self._calculate_distance(nearest_peer) <= self.vision_range:
            move_x, move_y = self._calculate_move_vector(nearest_peer, attraction=False)
        else:
            move_x, move_y = self._calculate_move_vector(direct_superior, attraction=False)
        
        self._attempt_move(move_x, move_y)
        self.step_number = 5

    # Helper methods

    def _get_nearest_in_range(self, agents, max_range):
        """Get nearest agent within range, sorted by rank"""
        candidates = [
            msg for msg in agents 
            if self._calculate_distance(msg) <= max_range
        ]
        
        if not candidates:
            return None
        
        return min(candidates, key=lambda m: m['rank'])

    def _get_nearest_agent(self, agents):
        """Get the nearest agent from a list"""
        if not agents:
            return None
        return min(agents, key=self._calculate_distance)

    def _get_visible_agents(self):
        """Get all agents within vision range"""
        return [
            msg for msg in self.received_messages 
            if (msg['type'] == 'status' and 
                0 < self._calculate_distance(msg) <= self.vision_range)
        ]
                
    # ---------- Return Behavior ----------
    def _update_rank_return(self):
        """
        Update rank during return phase:
        - Entrance agent always has rank 0
        - Other agents take rank of minimum rank in communication range + 1
        - If no lower rank agents found, preserve current rank
        """
        if self.is_entrance:
            self.rank = 0
            return

        # Find minimum rank among agents in communication range
        min_rank = float('inf')
        for msg in self.received_messages:
            if msg['type'] == 'status' and msg['rank'] >= 0:
                if msg['rank'] < min_rank:
                    min_rank = msg['rank']
        
        # Update rank if found lower rank agent, otherwise preserve current rank
        if min_rank != float('inf'):
            self.rank = min_rank + 1

    def _move_return(self):
        """
        Return mode behavior:
        1. First try to find and move towards the nearest lower-rank agent that has clear line of sight
        2. If no lower-rank agent found, move towards a peer agent that has clear line of sight
        3. Movement is directly towards the target agent
        """
        best_target = None
        min_distance = float('inf')

        # First try to find nearest lower-rank agent with clear line of sight
        for msg in self.received_messages:
            if msg['type'] != 'status':
                continue

            # Check if there is clear line of sight to this agent
            if not self._check_line_of_sight(msg['position'].x, msg['position'].y, self.environment.global_map):
                continue

            dx = self.position.x - msg['position'].x
            dy = self.position.y - msg['position'].y
            dist = math.sqrt(dx**2 + dy**2)

            if msg['rank'] < self.rank and dist < min_distance:
                min_distance = dist
                best_target = (dx, dy, dist)

        # If no lower-rank agent found, try to find a peer with clear line of sight
        if best_target is None:
            for msg in self.received_messages:
                if msg['type'] != 'status':
                    continue

                # Check if there is clear line of sight to this agent
                if not self._check_line_of_sight(msg['position'].x, msg['position'].y, self.environment.global_map):
                    continue

                dx = self.position.x - msg['position'].x
                dy = self.position.y - msg['position'].y
                dist = math.sqrt(dx**2 + dy**2)

                if msg['rank'] == self.rank and dist < min_distance:
                    min_distance = dist
                    best_target = (dx, dy, dist)

        if best_target is None:
            return

        # Compute movement vector
        dx, dy, dist = best_target
        move_x = round((-dx / (dist + 1e-6))*self.speed)
        move_y = round((-dy / (dist + 1e-6))*self.speed)

        # Use _attempt_move to handle occupied cells
        self._attempt_move(move_x, move_y)

    def _attempt_move(self, move_x, move_y):
        """
        Attempt to move to a new position, avoiding walls and occupied cells.
        Includes fallback options if the main target is occupied.
        """
        if move_x == 0 and move_y == 0:
            return
        
        # Occupied cells from perceived walls
        occupied = {
            (int(self.position.x + info['relative_position'][0]),
            int(self.position.y + info['relative_position'][1]))
            for info in self.perceived_environment if info['type'] == 'wall'
        }

        # Occupied by nearby agents
        occupied.update({
            (int(msg['position'].x), int(msg['position'].y))
            for msg in self.received_messages
        })

        px, py = int(self.position.x), int(self.position.y)
        
        # Generate more movement candidates with randomization
        candidates = [
            (px + move_x, py + move_y),                 # primary target
            (px + move_x - 1, py + move_y - 1),         # diagonal fallback
            (px + move_x - 1, py + move_y),             # left fallback
            (px + move_x, py + move_y - 1),             # up fallback
            (px + move_x + 1, py + move_y),             # right fallback
            (px + move_x, py + move_y + 1),             # down fallback
            (px + move_x + 1, py + move_y + 1),         # other diagonal
            (px + move_x - 1, py + move_y + 1),         # other diagonal
            (px + move_x + 1, py + move_y - 1)          # other diagonal
        ]
        
        # Shuffle candidates to add randomness
        random.shuffle(candidates)
        
        for tx, ty in candidates:
            # 检查路径是否被阻挡
            path_blocked = False
            steps = max(abs(tx - px), abs(ty - py))
            if steps > 0:
                for i in range(1, steps + 1):
                    check_x = px + int((tx - px) * i / steps)
                    check_y = py + int((ty - py) * i / steps)
                    if (check_x, check_y) in occupied:
                        path_blocked = True
                        break
            
            if not path_blocked and (tx, ty) not in occupied:
                self.position.x = tx
                self.position.y = ty
                return

    # ---------- Flood Search Behavior ---------
    def _update_flood_search(self):
        """
        Update rank based on nearby agents in communication range.
        Lower rank means closer to entrance.
        """
        if self.is_entrance:
            self.rank = 0
            return

        # If target found, don't modify rank
        if self.task_completion_index == 1:
            return

        # Find minimum rank among nearby agents
        min_rank = float('inf')
        for msg in self.received_messages:
            if msg['type'] == 'status' and msg['rank'] >= 0:
                if msg['rank'] < min_rank:
                    min_rank = msg['rank']
        
        # Update rank to be one more than the minimum rank found
        if min_rank != float('inf'):
            self.rank = min_rank + 1
        else:
            # If no agents in communication range, keep current rank
            pass

    def _move_flood_search(self):
        """
        Flood search movement behavior:
        1. Stop if no agents in vision range
        2. Move away from the lowest rank agent in vision range
        """
        if self.is_entrance:
            return

        # Get agents in vision range
        visible_agents = []
        for msg in self.received_messages:
            if msg['type'] != 'status':
                continue

            dist = self._calculate_distance(msg)
            if dist <= self.vision_range:
                visible_agents.append(msg)

        # Stop if no agents in vision range
        if not visible_agents:
            return

        # Find the lowest rank agent in vision range
        lowest_rank_agent = min(visible_agents, key=lambda x: x['rank'])
        
        # Calculate repulsion vector from the lowest rank agent
        dx = self.position.x - lowest_rank_agent['position'].x
        dy = self.position.y - lowest_rank_agent['position'].y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            # Normalize and apply movement
            move_x = round((dx / dist) * self.speed)
            move_y = round((dy / dist) * self.speed)
            self._attempt_move(move_x, move_y)



    # =====================
    # Simulation Interface
    # =====================
    def move(self, global_map=None, comm_map=None) -> None:
        """
        Simulation interface to update one agent step.
        Performs general update + function-mode behavior.
        """
        self._general_behavior(global_map, comm_map)
        
        # Perform movement behavior for all agents
        self._mode_based_behavior()
        
        self._broadcast_to_comm_map(comm_map, global_map) # simulation only
