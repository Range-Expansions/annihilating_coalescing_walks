__author__ = 'bryan'

import numpy as np

LEFT = 0
RIGHT = 1

class Lattice():

    def __init__(self, lattice_size, num_types=3):
        self.lattice_size = lattice_size
        self.lattice = np.random.randint(0, num_types, lattice_size)
        self.walls = self.get_walls()

    def get_walls(self):
        right_shift = np.roll(self.lattice, 1)
        wall_locations = self.lattice != right_shift
        wall_list = []
        wall_positions = np.where(wall_locations)[0]
        for cur_position in wall_positions:
            wall_list.append(Wall(cur_position))
        wall_list = np.array(wall_list)

        # Sort the wall list
        wall_list = np.sort(wall_list, axis=None)

        # Assign neighbors
        for i in range(wall_list.shape[0]):
            left_wall = wall_list[np.mod(i - 1, wall_list.shape[0])]
            right_wall = wall_list[np.mod(i + 1, wall_list.shape[0])]
            wall_list[i].wall_neighbors = np.array([left_wall, right_wall])

        # Indicate what type of wall the wall is
        for i in range(wall_list.shape[0]):
            cur_wall = wall_list[i]
            cur_position = wall_list[i].position

            left_index = np.mod(cur_position - 1, self.lattice_size)
            right_index = cur_position

            left_of_kink = self.lattice[left_index]
            right_of_kink = self.lattice[right_index]
            cur_type = np.array([left_of_kink, right_of_kink])
            cur_wall.wall_type = cur_type

        return wall_list

    def get_lattice_str(self):
        '''Assumes walls are already sorted...as they should be..'''

        # Gives the lattice string SOLELY in terms of wall position
        output_str = ''
        num_walls = self.walls.shape[0]
        # Loop through the walls in terms of position

        if num_walls > 2:
            cur_wall = self.walls[0]
            for i in range(self.lattice_size):
                if cur_wall.position == i:
                    cur_wall = cur_wall.wall_neighbors[1]
                    output_str += '_'
                output_str += str(cur_wall.wall_type[0])
        elif num_walls == 1:
            cur_wall = self.walls[0]
            for i in range(self.lattice_size):
                if i < cur_wall.position:
                    output_str += str(cur_wall.wall_type[0])
                elif i == cur_wall.position:
                    output_str += '_'
                    output_str += str(cur_wall.wall_type[1])
                else:
                    output_str += str(cur_wall.wall_type[1])
        else:
            print 'No walls...I cannot determine lattice information from walls!'''
        return output_str

    def get_lattice_from_walls(self):
        '''Returns the lattice array'''

        output_lattice = np.empty(self.lattice_size)
        num_walls = self.walls.shape[0]
        # Loop through the walls in terms of position

        if num_walls > 2:
            cur_wall = self.walls[0]
            for i in range(self.lattice_size):
                if cur_wall.position == i:
                    cur_wall = cur_wall.wall_neighbors[1]
                output_lattice[i] = cur_wall.wall_type[0]
        elif num_walls == 1:
            cur_wall = self.walls[0]
            for i in range(self.lattice_size):
                if i < cur_wall.position:
                    output_lattice[i] = cur_wall.wall_type[0]
                else:
                    output_lattice[i] = cur_wall.wall_type[1]
        else:
            print 'No walls...I cannot determine lattice information from walls!'''
        return output_lattice

    def collide(self, left_wall, right_wall):
        '''Collides two walls. Make sure the wall that was on the left
        is passed first.'''

        new_wall = None
        type_after_collision_left = left_wall.wall_type[0]
        type_after_collision_right = right_wall.wall_type[1]

        if type_after_collision_left != type_after_collision_right:
            new_position = left_wall.position
            new_type = np.array([type_after_collision_left, type_after_collision_right])
            new_wall = Wall(new_position, wall_type = new_type)

        collided_indices = (self.walls == left_wall)
        first_index = np.where(collided_indices)[0][0]
        if new_wall is not None: # Coalesce
            # Redo neighbors first
            self.walls[first_index] = new_wall
            wall_after = self.walls[np.mod(first_index + 2, self.walls.shape[0])]
            wall_before = self.walls[np.mod(first_index - 1, self.walls.shape[0])]

            wall_before.wall_neighbors[1] = new_wall
            new_wall.wall_neighbors = np.array([wall_before, wall_after])
            wall_after.wall_neighbors[0] = new_wall

            # Delete the undesired wall
            self.walls = np.delete(self.walls, np.mod(first_index + 1, self.walls.shape[0]))
        else: #Annihilate

            # Redo neighbors before annihilation for simplicity
            wall_before_index = self.walls[np.mod(first_index - 1, self.walls.shape[0])]
            wall_two_after_index = self.walls[np.mod(first_index + 2, self.walls.shape[0])]

            wall_before_index.wall_neighbors[1] = wall_two_after_index
            wall_two_after_index.wall_neighbors[0] = wall_before_index

            # Do the actual annihilation
            self.walls = self.walls[~collided_indices]

class Wall():
    def __init__(self, position, wall_neighbors = None, wall_type = None):
        self.position = position
        # Neighbors are neighboring walls! Left neighbor = 0, right neighbor = 1
        self.wall_neighbors = wall_neighbors
        # The wall type indicates the type of kink
        self.wall_type = wall_type

    def __eq__(self, other):
        return self.position == other.position

    def __gt__(self, other):
        return self.position > other.position

    def __lt__(self, other):
        return self.position < other.position

class Neutral_Lattice_Simulation():

    def __init__(self, lattice_size=100, num_types=3, record_every = 1):

        self.lattice_size = lattice_size
        self.num_types = num_types
        self.lattice = Lattice(lattice_size, num_types)

        self.time_array = None
        self.lattice_history = None
        self.record_every = record_every

    def run(self, num_steps):

        self.time_array = np.zeros(num_steps + 1)
        self.lattice_history = -1*np.ones((num_steps, self.lattice_size))

        self.lattice_history[0, :] = self.lattice.get_lattice_from_walls()

        cur_time = 0
        time_remainder = 0
        num_recorded = 1
        for i in range(num_steps):
            # Check to make sure there are at least 2 walls remaining
            if self.lattice.walls.shape[0] > 1:
                index = np.random.randint(0, self.lattice.walls.shape[0])
                current_wall = self.lattice.walls[index]
                # Determine time increment before deletion of walls
                delta_t = 1./self.lattice.walls.shape[0]

                # Draw a random number
                rand_num = np.random.rand()
                if rand_num < .5:
                    jump_direction = RIGHT
                    current_wall.position += 1
                    # No mod here, as we have to do extra stuff if there is a problem.
                    if current_wall.position == self.lattice_size:
                        current_wall.position = 0
                        self.walls = np.concatenate(([current_wall], self.lattice.walls[0:-1]))
                else:
                    jump_direction = LEFT
                    current_wall.position = current_wall.position -1
                    if current_wall.position < 0:
                        current_wall.position = self.lattice_size - 1
                        self.lattice.walls = np.concatenate((self.lattice.walls[1:], [current_wall]))

                new_wall = None
                if jump_direction == LEFT:
                    left_neighbor = current_wall.wall_neighbors[LEFT]
                    if current_wall.position == left_neighbor.position:
                        self.lattice.collide(left_neighbor, current_wall)
                if jump_direction == RIGHT:
                    right_neighbor = current_wall.wall_neighbors[RIGHT]
                    if current_wall.position == right_neighbor.position:
                        self.lattice.collide(current_wall, right_neighbor)

                #### Record information ####
                cur_time += delta_t
                time_remainder += delta_t

                if time_remainder >= 1: # Record this step
                    self.time_array[num_recorded] = cur_time
                    self.lattice_history[num_recorded, :] = self.lattice.get_lattice_from_walls()

                    num_recorded += 1
                    time_remainder -= 1
            else:
                print self.lattice.walls.shape[0] , 'walls remaining, done!'
                # Cut the output appropriately
                self.lattice_history = self.lattice_history[0:num_recorded, :]

                break