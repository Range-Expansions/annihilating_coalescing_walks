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
        for i in range(self.lattice_size):
            if wall_locations[i]:
                if i == 0:
                    left_neighbor = self.lattice[self.lattice_size - 1]
                    right_neighbor = self.lattice[i]
                if i == self.lattice_size - 1:
                    left_neighbor = self.lattice[i - 1]
                    right_neighbor = self.lattice[0]
                else:
                    left_neighbor = self.lattice[i - 1]
                    right_neighbor = self.lattice[i]
                neighbors = np.array([left_neighbor, right_neighbor])
                wall_list.append(Wall(i, neighbors))
        return np.array(wall_list)

    def get_lattice_str(self):
        '''Luckily, this is in 1d. So not too hard.'''
        # Gives the lattice string SOLELY in terms of wall position
        output_str = ''
        # Organize walls in terms of position
        positions = [z.position for z in self.walls]
        sorted_spaces = np.argsort(positions)
        sorted_walls = self.walls[sorted_spaces]

        num_walls = sorted_walls.shape[0]

        # Loop through the walls in terms of position
        cur_wall = sorted_walls[0]
        wall_count = 1
        for i in range(self.lattice_size):
            if cur_wall.position == i:
                cur_wall = sorted_walls[np.mod(wall_count, num_walls)]
                wall_count += 1
                output_str += '_'
            output_str += str(cur_wall.neighbors[0])
        return output_str

class Wall():
    def __init__(self, position, neighbors):
        self.position = position
        self.neighbors = neighbors

    def __eq__(self, other):
        return self.position == other.position

    def __gt__(self, other):
        return self.position > other.position

    def __lt__(self, other):
        return self.position < other.position

    def collide(self, other):
        '''Collides two walls, returns none or a new wall as appropriate.'''
        # It's not clear which is on the left & which is on the right
        if self.neighbors[1] == other.neighbors[0]:
            # self is on the left, other is on the right
            if self.neighbors[0] == other.neighbors[1]:
                return None
            else:
                new_domain = np.array([self.neighbors[0], other.neighbors[1]])
                new_wall = Wall(self.position, new_domain)
                return new_wall
        elif other.neighbors[1] == self.neighbors[0]:
            #self is on the right, other is on the left
            if other.neighbors[0] == self.neighbors[1]:
                return None
            else:
                new_domain = np.array([other.neighbors[0], self.neighbors[1]])
                new_wall = Wall(self.position, new_domain)
                return new_wall