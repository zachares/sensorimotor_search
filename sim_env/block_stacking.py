from random import randrange as rand
import pygame, sys
import numpy as np
import yaml
import time
import sys
import h5py

#### maybe add a graph representation where the edges are the line segements between points

with open("/scr-ssd/sensorimotor_search/sim_env/game_params.yml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

cell_size = cfg['game_params']['cell_size']
cols = cfg['game_params']['cols']
rows = cfg['game_params']['rows']
delay = cfg['game_params']['delay']
maxfps = cfg['game_params']['maxfps']
num_goals = cfg['game_params']['num_goals']

class Block(object):

	def __init__(self, pose = np.array([-1, -1, -1]), color = -1):

		if pose[0] == -1:
			self.position = np.array([np.random.randint(cols-1), 0, 0])
		else: 
			self.position = pose

		self.color = self.init_color(color)
		self.shape = self.init_shape()
		self.edge_list = []

		self.rotate(self.position[2])

	def init_color(self, color):

		if color == -1:
			N = np.random.choice([0,1,2,3])
		else:
			N = color
		### Cyberpunk Color Scheme ####

		if N == 0:
			return np.array([0, 255, 159])
		elif N ==  1:
			return np.array([0, 184,255])
		elif N == 2:
			return np.array([189,0,255])
		else:
			return np.array([255,0,239])

	def init_shape(self):
		temp_int = np.random.randint(3)
		temp_int = 1

		if temp_int == 0:
			shape = np.array([[0,0]])
		elif temp_int ==1:
			shape = np.array([[0,0],[0,1]])
		else:
			shape = np.array([[0,0],[0,1],[1,0],[1,1]])
		return shape

	def rotate(self, ori):

		ori = ori % 4
		self.position[2] = ori

		if ori == 0:
			target = np.array([0,1])

		elif ori == 1:
			target = np.array([1, 0])

		elif ori == 2:
			target = np.array([0, -1])

		else:
			target = np.array([-1, 0])

		while self.shape[1,:][0] != target[0] or self.shape[1,:][1] != target[1]:
			self.shape = np.matmul(self.shape, np.array([[0, -1],[1, 0]])) 

	def translate(self, position):

		self.position[:-1] = position 

class env_BP(object): #block placing simulation

	def __init__(self, new_block_bool, num_goals = 2):
		self.num_goals = num_goals
		self.board_reset()

	def board_reset(self):

		self.board = np.zeros((rows, cols))
		self.board_image = np.zeros((3, rows, cols))
		self.block_list = []
		self.bl_idx = -1
		self.reward = -2
		self.construct_goal_structures()

	def new_block(self, pose = np.array([-1, -1, -1]), color = 0):
		# print("New Block Made")
		self.block_list.append(Block(pose = pose, color = color))
		# block = self.block_list[self.bl_idx]

		# block_positions = block.position[:-1] + block.shape
		# print(block_positions)
		self.fill_board()
		self.bl_idx += 1
		self.x_pos = self.block_list[self.bl_idx].position[0]

	def construct_goal_structures(self):

		##### determine goal boundaries based on number of goals
		##### determine hole position for each goal
		##### place blocks to fill goal boundaries
		##### record position of hole for reward function
		#### spawn control block

		#### choosing color of goal block
		self.goal_color = np.random.choice(list(range(self.num_goals)))

		##### calculating intervals for each color structure to contain hole
		goal_interval_size = np.floor(cols / self.num_goals)
		self.goal_intervals = [0]

		for idx in range(self.num_goals):
			self.goal_intervals.append(goal_interval_size * (idx + 1))

		if self.goal_intervals[-1] < cols:
			self.goal_intervals[-1] = cols

		self.goal_bound_array = np.zeros((len(self.goal_intervals) - 1, 2))

		for idx in range(len(self.goal_intervals) - 1):
			self.goal_bound_array[idx, 0] = self.goal_intervals[idx]
			self.goal_bound_array[idx, 1] = self.goal_intervals[idx + 1]

		self.hole_pos_list = []
		
		for idx in range(len(self.goal_intervals) - 1):

			self.hole_pos_list.append(self.goal_intervals[idx] + (self.goal_intervals[idx + 1] - self.goal_intervals[idx]) // 2)

		for idx in range(cols):

			for idx_list in range(len(self.goal_intervals) - 1):

				if idx >= self.goal_intervals[idx_list] and idx < self.goal_intervals[idx_list + 1]:

					if idx == self.hole_pos_list[idx_list]:
						self.new_block(pose = np.array([idx, rows - 2,  0]), color = idx_list)

					else:
						self.new_block(pose = np.array([idx, rows - 2, 0]), color = idx_list)

						self.new_block(pose = np.array([idx, rows - 4, 0]), color = idx_list)

		self.new_block(pose = np.array([-1, -1, -1]), color = self.goal_color)

	def calc_reward(self):

		block = self.block_list[self.bl_idx]

		block_positions = block.position[:-1] + block.shape

		self.reward = -2

		if block_positions[0, 0] == self.hole_pos_list[self.goal_color] and (block_positions[0,1] == rows -3 or block_positions[0,1] == rows - 4):
			self.reward += 10

		if block_positions[1, 0] == self.hole_pos_list[self.goal_color] and (block_positions[1,1] == rows -3 or block_positions[1,1] == rows - 4):
			self.reward += 10

		return self.reward

	def check_hole_collision(self):

		block = self.block_list[self.bl_idx]

		block_positions = block.position[:-1] + block.shape

		for col_pos in self.hole_pos_list:

			if col_pos == self.hole_pos_list[self.goal_color]:
				continue

			if block_positions[0, 0] == col_pos and (block_positions[0,1] == rows -3 or block_positions[0,1] == rows - 4):
				return True

			elif block_positions[1, 0] == col_pos and (block_positions[1,1] == rows -3 or block_positions[1,1] == rows - 4):
				return True


		return False

	def check_boundary_collision(self):

		block = self.block_list[self.bl_idx]

		block_positions = block.position[:-1] + block.shape

		bottom_check = np.sum(np.where(block_positions[:, 1] >= rows, 1, 0))
		left_check = np.sum(np.where(block_positions[:,0] < 0, 1, 0))
		right_check = np.sum(np.where(block_positions[:,0] >= cols, 1, 0))
		top_check = np.sum(np.where(block_positions[:,1] < 0, 1, 0))

		if left_check + right_check + top_check + bottom_check > 0:
			# print("Illegal move")
			return True
		else:
			return False

	def check_block_collision(self):

		block = self.block_list[self.bl_idx]

		block_positions = block.position[:-1] + block.shape

		if np.sum(self.board[block_positions[:,1], block_positions[:,0]]) > 0:
			return True
		else:
			return False

	def fill_board(self):

		if len(self.block_list) != 1:

			for idx, block in enumerate(self.block_list[:-1]):

				block_positions = block.position[:-1] + block.shape
				# print(block_positions)
				self.board[block_positions[:,1], block_positions[:,0]] = idx + 1
				self.board_image[:, block_positions[:,1], block_positions[:,0]] = np.repeat(np.expand_dims(block.color, axis = 1), 2, axis = 1)

	def rotate_block(self, direction):

		ori = (self.block_list[self.bl_idx].position[2] + direction) % 4

		self.block_list[self.bl_idx].rotate(ori)

		ori = (self.block_list[self.bl_idx].position[2] - direction) % 4
		
		if self.check_boundary_collision() or self.check_block_collision() or self.check_hole_collision():
			self.block_list[self.bl_idx].rotate(ori)

		self.x_pos = self.block_list[self.bl_idx].position[0]

		self.calc_reward()

	def translate_block(self, delta):

		position = self.block_list[self.bl_idx].position[:-1] + delta

		self.block_list[self.bl_idx].translate(position)

		position = self.block_list[self.bl_idx].position[:-1] - delta

		if self.check_boundary_collision() or self.check_block_collision() or self.check_hole_collision():
			self.block_list[self.bl_idx].translate(position)

		self.x_pos = self.block_list[self.bl_idx].position[0]

		self.calc_reward()

	def step(self, action_num):

		if action_num == 0:
			# print("LEFT")
			self.translate_block(np.array([-1, 0]))	
			return self.board_image						
		elif action_num == 1:
			# print("RIGHT")
			self.translate_block(np.array([1, 0]))	
			return self.board_image						
		elif action_num == 2:
			# print("DOWN")
			self.translate_block(np.array([0, 1]))
			return self.board_image						
		elif action_num == 3:
			# print("UP")
			self.translate_block(np.array([0, -1]))
		elif action_num == 4:
			self.rotate_block(1)
			return self.board_image
			# print("Rotate CCW")
		elif action_num == 5:
			self.rotate_block(-1)
			return self.board_image
			# print("Rotate CW")
		else:
			self.board_reset()

		self.fill_board()
		return self.board_image, self.x_pos, self.reward

class env_BP_w_display(object): #display
	
	def __init__(self, num_goals = 2):

		pygame.init()
		pygame.key.set_repeat(250,25)		

		pygame.event.set_blocked(pygame.MOUSEMOTION) # We do not need
		                                             # mouse movement
		                                             # events, so we
		                                             # block them.
		self.maxfps = maxfps
		self.delay = delay

		self.width = cell_size * cols
		self.height = cell_size * rows
		self.screen = pygame.display.set_mode((self.width, self.height))
		self.board = env_BP(num_goals)
		self.goal_bound_array = self.board.goal_bound_array

	def print_board(self):

		block_list = self.board.block_list
		for block in block_list:
			block_positions = cell_size * (block.position[:-1] + block.shape)

			for idx_module in range(block_positions.shape[0]):

				block_Rect = pygame.Rect(block_positions[idx_module,0], block_positions[idx_module,1], cell_size, cell_size)
				pygame.draw.rect(self.screen, block.color, block_Rect, 0)		
				pygame.draw.rect(self.screen, np.array([255, 255, 255]), block_Rect, 1)

	def update_display(self):

		pygame.display.update()
		
	def quit(self):
		
		sys.exit()
	
	def board_reset(self):

		self.board.board_reset()

	def run(self):
		key_actions = {
			'ESCAPE':	self.quit,
			'LEFT':		self.board.step,
			'RIGHT':	self.board.step,
			'DOWN':		self.board.step,
			'UP':		self.board.step,
			's':		self.board.step, # left
			'd':		self.board.step, # right
			'SPACE':	self.board.step,
		}

		pygame.time.set_timer(pygame.USEREVENT+1, self.delay)
		dont_burn_my_cpu = pygame.time.Clock()

		### Keyboard control instructions
		print("  ")
		print("#######################Keyboard Controls#######################")
		print("  ")
		print("	ESC          == quit game")
		print("  ")
		print("	Left Arrow   == Move block left")
		print("  ")
		print("	Right Arrow  == Move block right")
		print("  ")
		print("	Down Arrow   == Move block down")
		print("  ")
		print("	Up Arrow     == Move block up")
		print("  ")
		print("	s            == rotate block counter clockwise")
		print("  ")
		print("	d            == rotate block clockwise")
		print("  ")
		print("	SPACE        == reset board with a keyboard")
		print("	                controlled block")
		print("  ")
		print("###############################################################")
		print("  ")

		while 1:

			self.screen.fill((0,0,0))

			self.print_board()

			self.update_display()
			self.image = pygame.surfarray.pixels3d(self.screen)
			self.ee_pos = self.board.block_list[self.board.bl_idx].position

			for event in pygame.event.get():
				if event.type == pygame.USEREVENT+1:
					continue
				elif event.type == pygame.QUIT:
					self.quit()
				elif event.type == pygame.KEYDOWN:
					for key in key_actions:
						if event.key == eval("pygame.K_"+key):
							if key == "LEFT":
								# print("LEFT")
								key_actions[key](0)							
							elif key == "RIGHT":
								# print("RIGHT")
								key_actions[key](1)							
							elif key == "DOWN":
								# print("DOWN")
								key_actions[key](2)							
							elif key == "UP":
								# print("UP")
								key_actions[key](3)
							elif key == "s":
								key_actions[key](4)
								# print("Rotate CCW")
							elif key == "d":
								key_actions[key](5)
								# print("Rotate CW")
							elif key == "SPACE":
								key_actions[key](6)
							else:
								key_actions[key]()

	def step(self, action_num):

		pygame.time.set_timer(pygame.USEREVENT+1, self.delay)
		dont_burn_my_cpu = pygame.time.Clock()
		
		self.screen.fill((0,0,0))

		return self.board.step(action_num)

if __name__ == '__main__':
	App = env_BP_w_display(num_goals)
	App.run()