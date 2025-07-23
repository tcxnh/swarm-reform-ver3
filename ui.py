import tkinter as tk
from tkinter import ttk
import json
from environment import Environment
import matplotlib.pyplot as plt
import tkinter.messagebox as messagebox

class SwarmConfigUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Swarm Configuration")
        self.root.geometry("600x900")
        
        # Initialize default values
        self.rank_config = {
            0: 1,  # Entrance
            1: 1,  # Rank 1
            2: 1,  # Rank 2
            3: 1   # Rank 3
        }
        
        self.init_method = "random"  # Default initialization method
        self.num_agents = 20
        self.width = 150  # 修改默认宽度
        self.height = 150  # 修改默认高度
        
        # Store both labels and entries
        self.rank_labels = {}
        self.rank_entries = {}
        
        # Store button references
        self.rank_btn_frame = None
        self.add_btn = None
        self.remove_btn = None
        
        # Store environment reference
        self.env = None
        
        # Bind key press event
        self.root.bind('<space>', self.on_space_press)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Function Mode Selection Section
        ttk.Label(self.main_frame, text="Function Mode", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        self.function_mode_var = tk.StringVar(value="flood-evol-search")
        modes = [
            ("Flex Search", "flex-search"),
            ("Flood Search", "flood-search"),
            ("Flood Evol Search", "flood-evol-search")
        ]
        
        for i, (text, value) in enumerate(modes):
            ttk.Radiobutton(self.main_frame, text=text, value=value, variable=self.function_mode_var).grid(
                row=i+1, column=0, columnspan=2, pady=5, sticky=tk.W
            )
        
        # Rank Configuration Section
        ttk.Label(self.main_frame, text="Rank Configuration", font=('Arial', 12, 'bold')).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Add real-time update checkbox
        self.realtime_update_var = tk.BooleanVar(value=False)
        self.just_enabled_realtime = False  # New flag to track if real-time update was just enabled
        realtime_check = ttk.Checkbutton(self.main_frame, text="Enable Real-time Update", 
                                       variable=self.realtime_update_var,
                                       command=self.on_realtime_update_change)  # Add callback
        realtime_check.grid(row=4, column=2, padx=5, pady=5)
        
        # Create rank config entries
        for i, rank in enumerate(sorted(self.rank_config.keys())):
            label = ttk.Label(self.main_frame, text=f"Rank {rank}:")
            label.grid(row=i+5, column=0, padx=5, pady=5)
            self.rank_labels[rank] = label
            
            entry = ttk.Entry(self.main_frame, width=10)
            entry.insert(0, str(self.rank_config[rank]))
            entry.grid(row=i+5, column=1, padx=5, pady=5)
            self.rank_entries[rank] = entry
        
        # Add/Remove Rank buttons
        rank_btn_frame = ttk.Frame(self.main_frame)
        rank_btn_frame.grid(row=len(self.rank_config)+5, column=0, columnspan=2, pady=10)
        
        ttk.Button(rank_btn_frame, text="Add Rank", command=self.add_rank).pack(side=tk.LEFT, padx=5)
        ttk.Button(rank_btn_frame, text="Remove Rank", command=self.remove_rank).pack(side=tk.LEFT, padx=5)
        
        # Initialization Method Section
        ttk.Label(self.main_frame, text="Initialization Method", font=('Arial', 12, 'bold')).grid(row=len(self.rank_config)+6, column=0, columnspan=2, pady=10)
        
        self.init_var = tk.StringVar(value="grid")
        methods = [
            ("Random Placement", "random"),
            ("Around Enrtance Placement", "grid")
        ]
        
        for i, (text, value) in enumerate(methods):
            ttk.Radiobutton(self.main_frame, text=text, value=value, variable=self.init_var).grid(
                row=len(self.rank_config)+7+i, column=0, columnspan=2, pady=5, sticky=tk.W
            )
        
        # Environment Parameters
        ttk.Label(self.main_frame, text="Environment Parameters", font=('Arial', 12, 'bold')).grid(
            row=len(self.rank_config)+10, column=0, columnspan=2, pady=10
        )
        
        # Number of agents
        ttk.Label(self.main_frame, text="Number of Agents:").grid(row=len(self.rank_config)+11, column=0, padx=5, pady=5)
        self.num_agents_entry = ttk.Entry(self.main_frame, width=10)
        self.num_agents_entry.insert(0, str(self.num_agents))
        self.num_agents_entry.grid(row=len(self.rank_config)+11, column=1, padx=5, pady=5)
        
        # Width
        ttk.Label(self.main_frame, text="Width:").grid(row=len(self.rank_config)+12, column=0, padx=5, pady=5)
        self.width_entry = ttk.Entry(self.main_frame, width=10)
        self.width_entry.insert(0, str(self.width))
        self.width_entry.grid(row=len(self.rank_config)+12, column=1, padx=5, pady=5)
        
        # Height
        ttk.Label(self.main_frame, text="Height:").grid(row=len(self.rank_config)+13, column=0, padx=5, pady=5)
        self.height_entry = ttk.Entry(self.main_frame, width=10)
        self.height_entry.insert(0, str(self.height))
        self.height_entry.grid(row=len(self.rank_config)+13, column=1, padx=5, pady=5)
        
        # Add obstacle option
        self.obstacle_var = tk.BooleanVar(value=False)
        obstacle_check = ttk.Checkbutton(self.main_frame, text="Add Obstacles", variable=self.obstacle_var)
        obstacle_check.grid(row=len(self.rank_config)+14, column=0, columnspan=2, pady=5)
        
        # Add target point options
        target_frame = ttk.LabelFrame(self.main_frame, text="Target Point Options")
        target_frame.grid(row=len(self.rank_config)+15, column=0, columnspan=2, pady=5, sticky='ew')
        
        self.target_var = tk.BooleanVar(value=False)
        target_check = ttk.Checkbutton(target_frame, text="Add Target Point", variable=self.target_var)
        target_check.grid(row=0, column=0, columnspan=2, pady=5)
        
        self.target_pos_var = tk.StringVar(value="right")
        ttk.Radiobutton(target_frame, text="Right Position (120, 60)", value="right", 
                       variable=self.target_pos_var).grid(row=1, column=0, pady=2)
        ttk.Radiobutton(target_frame, text="Top Position (75, 20)", value="top", 
                       variable=self.target_pos_var).grid(row=1, column=1, pady=2)
        
        # Add fork point options (模仿target)
        self.fork_var = tk.BooleanVar(value=True)
        fork_check = ttk.Checkbutton(target_frame, text="Add Fork Point", variable=self.fork_var)
        fork_check.grid(row=2, column=0, columnspan=2, pady=5)
        
        self.fork_pos_var = tk.StringVar(value="middle")
        ttk.Radiobutton(target_frame, text="Right Position (120, 80)", value="right", 
                       variable=self.fork_pos_var).grid(row=3, column=0, pady=2)
        ttk.Radiobutton(target_frame, text="Middle Position (width//2, height//2)", value="middle", 
                       variable=self.fork_pos_var).grid(row=3, column=1, pady=2)
        
        # Initialize and Play/Pause buttons
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.grid(row=len(self.rank_config)+16, column=0, columnspan=2, pady=10)
        
        self.init_btn = ttk.Button(btn_frame, text="Initialize", command=self.initialize_simulation)
        self.init_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(btn_frame, text="Play", command=self.toggle_simulation)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.play_btn.state(['disabled'])  # Initially disabled
        
    def add_rank(self):
        new_rank = max(self.rank_entries.keys()) + 1
        label = ttk.Label(self.main_frame, text=f"Rank {new_rank}:")
        label.grid(row=len(self.rank_entries)+5, column=0, padx=5, pady=5)
        self.rank_labels[new_rank] = label
        
        entry = ttk.Entry(self.main_frame, width=10)
        entry.insert(0, "1")
        entry.grid(row=len(self.rank_entries)+5, column=1, padx=5, pady=5)
        self.rank_entries[new_rank] = entry
        
        # Update all widget positions
        self.update_widgets()
        
    def remove_rank(self):
        if len(self.rank_entries) > 1:
            last_rank = max(self.rank_entries.keys())
            # Destroy both label and entry
            self.rank_labels[last_rank].destroy()
            self.rank_entries[last_rank].destroy()
            del self.rank_labels[last_rank]
            del self.rank_entries[last_rank]
            
            # Update all widget positions
            self.update_widgets()
            
    def update_widgets(self):
        # Update all widget positions after adding/removing ranks
        for i, rank in enumerate(sorted(self.rank_entries.keys())):
            self.rank_labels[rank].grid(row=i+5, column=0, padx=5, pady=5)
            self.rank_entries[rank].grid(row=i+5, column=1, padx=5, pady=5)
        
        # Update positions of all subsequent widgets
        base_row = len(self.rank_entries) + 5
        
        # Update Initialization Method section
        init_label = ttk.Label(self.main_frame, text="Initialization Method", font=('Arial', 12, 'bold'))
        init_label.grid(row=base_row+1, column=0, columnspan=2, pady=10)
        
        # Update real-time update checkbox
        realtime_check = ttk.Checkbutton(self.main_frame, text="Enable Real-time Update", variable=self.realtime_update_var)
        realtime_check.grid(row=base_row+1, column=2, padx=5, pady=5)
        
        # Update initialization method radio buttons
        for i, (text, value) in enumerate([
            ("Random Placement", "random"),
            ("Around Enrtance Placement", "grid")
        ]):
            radio = ttk.Radiobutton(self.main_frame, text=text, value=value, variable=self.init_var)
            radio.grid(row=base_row+2+i, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        # Update Environment Parameters section
        env_label = ttk.Label(self.main_frame, text="Environment Parameters", font=('Arial', 12, 'bold'))
        env_label.grid(row=base_row+5, column=0, columnspan=2, pady=10)
        
        # Update environment parameter entries
        for i, (label_text, entry) in enumerate([
            ("Number of Agents:", self.num_agents_entry),
            ("Width:", self.width_entry),
            ("Height:", self.height_entry)
        ]):
            label = ttk.Label(self.main_frame, text=label_text)
            label.grid(row=base_row+6+i, column=0, padx=5, pady=5)
            entry.grid(row=base_row+6+i, column=1, padx=5, pady=5)
        
        # Update obstacle checkbox
        obstacle_check = ttk.Checkbutton(self.main_frame, text="Add Obstacles", variable=self.obstacle_var)
        obstacle_check.grid(row=base_row+9, column=0, columnspan=2, pady=5)
        
        # Update target point options frame
        target_frame = ttk.LabelFrame(self.main_frame, text="Target Point Options")
        target_frame.grid(row=base_row+10, column=0, columnspan=2, pady=5, sticky='ew')
        
        target_check = ttk.Checkbutton(target_frame, text="Add Target Point", variable=self.target_var)
        target_check.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Radiobutton(target_frame, text="Right Position (120, 60)", value="right", 
                       variable=self.target_pos_var).grid(row=1, column=0, pady=2)
        ttk.Radiobutton(target_frame, text="Top Position (75, 20)", value="top", 
                       variable=self.target_pos_var).grid(row=1, column=1, pady=2)
        
        # Update fork point options frame
        fork_frame = ttk.LabelFrame(self.main_frame, text="Fork Point Options")
        fork_frame.grid(row=base_row+11, column=0, columnspan=2, pady=5, sticky='ew')

        fork_check = ttk.Checkbutton(fork_frame, text="Add Fork Point", variable=self.fork_var)
        fork_check.grid(row=0, column=0, columnspan=2, pady=5)

        ttk.Radiobutton(fork_frame, text="Right Position (120, 80)", value="right", 
                       variable=self.fork_pos_var).grid(row=1, column=0, pady=2)
        ttk.Radiobutton(fork_frame, text="Middle Position (width//2, height//2)", value="middle", 
                       variable=self.fork_pos_var).grid(row=1, column=1, pady=2)
        
        # Update button frame
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.grid(row=base_row+12, column=0, columnspan=2, pady=20)
        
        self.init_btn = ttk.Button(btn_frame, text="Initialize", command=self.initialize_simulation)
        self.init_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(btn_frame, text="Play", command=self.toggle_simulation)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.play_btn.state(['disabled'])  # Initially disabled
        
    def get_config(self):
        """Get configuration from UI inputs"""
        try:
            config = {
                'num_agents': int(self.num_agents_entry.get()),
                'width': int(self.width_entry.get()),
                'height': int(self.height_entry.get()),
                'init_method': self.init_var.get(),
                'function_mode': self.function_mode_var.get(),
                'add_obstacles': self.obstacle_var.get(),
                'add_target': self.target_var.get(),
                'target_position': self.target_pos_var.get(),  # Add target position
                'add_fork': self.fork_var.get(),
                'fork_position': self.fork_pos_var.get(),
                'rank_config': {}
            }
            
            # Get rank configurations
            for rank, entry in self.rank_entries.items():
                config['rank_config'][str(rank)] = int(entry.get())
                
            return config
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
            return None
        
    def initialize_simulation(self):
        """Initialize the simulation environment"""
        config = self.get_config()
        if config:
            print("Saving configuration:", config)
            # Save configuration to file
            with open('swarm_config.json', 'w') as f:
                json.dump(config, f, indent=4)
            
            # 设置matplotlib后端
            import matplotlib
            matplotlib.use('TkAgg')
            plt.rcParams['figure.dpi'] = 150
            plt.rcParams['figure.figsize'] = [15, 15]
            plt.rcParams['font.size'] = 10
            plt.rcParams['lines.linewidth'] = 1.5
            
            # Create environment with config
            self.env = Environment(
                num_agents=config['num_agents'],
                width=config['width'],
                height=config['height'],
                config=config  # Pass the full config
            )
            
            # Initialize agents with full config
            self.env.initialize_agents(config['num_agents'], config)
            
            # Enable play button
            self.play_btn.state(['!disabled'])
            
            # Show initial state
            self.env.step()  # Record initial state
            self.env.render_grid_animation()
            plt.show(block=False)  # Show the plot without blocking
    
    def toggle_simulation(self):
        """Toggle simulation between running and paused states"""
        if not self.env:
            return
            
        if self.env.is_running:
            self.env.stop_simulation()
            self.play_btn.configure(text="Play")
        else:
            self.env.start_simulation()
            self.play_btn.configure(text="Pause")
            self.run_simulation()
    
    def update_rank_config(self):
        """Update rank configuration in real-time"""
        if not self.env:
            return
            
        try:
            new_config = {}
            for rank, entry in self.rank_entries.items():
                new_config[str(rank)] = int(entry.get())
            #print("Updating rank config:", new_config)  # Debug print
            self.env.update_rank_config(new_config)
            print("Current environment rank_config:", self.env.rank_config)  # Debug print
            
            # Force update all agents
            for agent in self.env.agents:
                agent.environment = self.env
                # print(f"Updated agent {agent.id} with new rank config")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
            return

    def on_realtime_update_change(self):
        """Handle real-time update checkbox state change"""
        if self.realtime_update_var.get():
            self.just_enabled_realtime = True
            print("Real-time update enabled, will perform flood search rank update")

    def run_simulation(self):
        """Run simulation steps until stopped"""
        if self.env and self.env.is_running:
            # Check and update rank config if real-time update is enabled
            if self.realtime_update_var.get():
                # If real-time update was just enabled, perform 5 steps of flood search rank updates
                if self.just_enabled_realtime:
                    print("Performing 5 steps of flood search rank updates")
                    # Perform 5 steps of rank updates for all non-entrance agents
                    for step in range(5):
                        for agent in self.env.agents:
                            if not agent.is_entrance:
                                agent._update_flood_search()
                        print(f"Completed step {step + 1} of rank updates")
                    
                    # Reset the flag
                    self.just_enabled_realtime = False
                    print("Rank updates completed")
                
                # Update rank configuration
                self.update_rank_config()
            
            self.env.step()
            self.env.render_grid_animation()
            self.root.after(50, self.run_simulation)  # Changed from 20ms to 50ms for better performance
    
    def run(self):
        self.root.mainloop()

    def on_space_press(self, event):
        """Handle space key press to update rank configuration"""
        if self.env:
            print("Space pressed - Updating rank configuration")
            self.update_rank_config()

if __name__ == "__main__":
    app = SwarmConfigUI()
    app.run() 