__author__ = 'bryan'

import numpy as np

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
            if i == 0:
                left_wall = wall_list[wall_list.shape[0] - 1]
                right_wall = wall_list[i + 1]
            if i == wall_list.shape[0] - 1:
                left_wall = wall_list[i - 1]
                right_wall = wall_list[0]
            else:
                left_wall = wall_list[i - 1]
                right_wall = wall_list[i + 1]

            wall_list[i].wall_neighbors = np.array(left_wall, right_wall)
        # Indicate what type of wall the wall is
        for i in range(wall_list.shape[0]):
            cur_wall = wall_list[i]
            cur_position = wall_list[i].position

            if cur_position == 0:
                left_index = self.lattice_size - 1
                right_index = cur_position
            if cur_position == (self.lattice_size - 1):
                left_index = cur_position - 1
                right_index = 0
            else:
                left_index = cur_position - 1
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
        cur_wall = self.walls[0]
        for i in range(self.lattice_size):
            if cur_wall.position == i:
                cur_wall = cur_wall.wall_neighbors[1]
                output_str += '_'
            output_str += str(cur_wall.wall_type[1])
        return output_str

class Wall():
    def __init__(self, position, wall_neighbors = None, wall_type = None):
        self.position = position
        # Neighbors are neighboring walls!
        self.wall_neighbors = wall_neighbors
        # The wall type indicates the type of kink
        self.wall_type = None

    def __eq__(self, other):
        return self.position == other.position

    def __gt__(self, other):
        return self.position > other.position

    def __lt__(self, other):
        return self.position < other.position

    def collide(self, other):
        '''Collides two walls, returns none or a new wall as appropriate.'''
        # It's not clear which is on the left & which is on the right
        if self.wall_neighbors[1] == other.neighbors[0]:
            # self is on the left, other is on the right
            if self.wall_neighbors[0] == other.neighbors[1]:
                return None
            else:
                new_domain = np.array([self.wall_neighbors[0], other.neighbors[1]])
                new_wall = Wall(self.position, new_domain)
                return new_wall
        elif other.neighbors[1] == self.wall_neighbors[0]:
            #self is on the right, other is on the left
            if other.neighbors[0] == self.wall_neighbors[1]:
                return None
            else:
                new_domain = np.array([other.neighbors[0], self.wall_neighbors[1]])
                new_wall = Wall(self.position, new_domain)
                return new_wall