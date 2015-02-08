__author__ = 'bryan'

import numpy as np

LEFT = 0
RIGHT = 1

ANNIHILATE = 0
COALESCE = 1

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

        if num_walls > 1:
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

        if num_walls > 1:
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
            return COALESCE
        else: #Annihilate
            print np.where(collided_indices)[0]
            # Redo neighbors before annihilation for simplicity
            wall_before_index = self.walls[np.mod(first_index - 1, self.walls.shape[0])]
            wall_two_after_index = self.walls[np.mod(first_index + 2, self.walls.shape[0])]

            wall_before_index.wall_neighbors[1] = wall_two_after_index
            wall_two_after_index.wall_neighbors[0] = wall_before_index

            # Do the actual annihilation
            self.walls = self.walls[~collided_indices]
            return ANNIHILATE

class Selection_Lattice(Lattice):

    def __init__(self, delta_prob_dict, lattice_size, num_types=3):
        Lattice.__init__(self, lattice_size, num_types=num_types)
        self.delta_prob_dict = delta_prob_dict
        # Initialize the walls correctly; make them selection walls!
        for i in range(self.walls.shape[0]):
            self.walls[i] = Selection_Wall.get_selection_wall_from_neutral(self.walls[i], self.delta_prob_dict)

    def collide(self, left_wall, right_wall):
        '''Collides two walls. Make sure the wall that was on the left
        is passed first.'''

        new_wall = None
        type_after_collision_left = left_wall.wall_type[0]
        type_after_collision_right = right_wall.wall_type[1]

        if type_after_collision_left != type_after_collision_right:
            new_position = left_wall.position
            new_type = np.array([type_after_collision_left, type_after_collision_right])
            # This is the change here
            new_wall = Selection_Wall(new_position, wall_type = new_type, delta_prob_dict=self.delta_prob_dict)

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
            return COALESCE
        else: #Annihilate
            if np.where(collided_indices)[0].shape[0] < 2:
                print 'Somehow, annihilations are occuring at a single wall...wtf'
                positions = [z.position for z in self.walls]
                print positions

            # Redo neighbors before annihilation for simplicity
            wall_before_index = self.walls[np.mod(first_index - 1, self.walls.shape[0])]
            wall_two_after_index = self.walls[np.mod(first_index + 2, self.walls.shape[0])]

            wall_before_index.wall_neighbors[1] = wall_two_after_index
            wall_two_after_index.wall_neighbors[0] = wall_before_index

            # Do the actual annihilation
            self.walls = self.walls[~collided_indices]
            return ANNIHILATE

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

    def get_jump_direction(self):
        # This is the neutral case
        random_num = np.random.rand()
        if random_num < 0.5:
            return RIGHT
        else:
            return LEFT

class Selection_Wall(Wall):
    '''One must define which wall gets a selective advantage. Let's say the biggest.'''

    def __init__(self, position, wall_neighbors=None, wall_type = None, delta_prob_dict = None):
        '''delta_prob_dict dictates what change in probability you get based on your neighbor.'''
        Wall.__init__(self, position, wall_neighbors, wall_type)
        self.delta_prob_dict = delta_prob_dict

    def get_jump_direction(self):
        # Get the change in probability of jumping left and right.
        right_prob_change = self.delta_prob_dict[self.wall_type[0], self.wall_type[1]]

        random_num = np.random.rand()
        if random_num < 0.5 + right_prob_change:
            return RIGHT
        else:
            return LEFT

    @staticmethod
    def get_selection_wall_from_neutral(neutral_wall, delta_prob_dict):
        return Selection_Wall(neutral_wall.position, wall_neighbors = neutral_wall.wall_neighbors,
                              wall_type = neutral_wall.wall_type, delta_prob_dict = delta_prob_dict)

class Lattice_Simulation():

    def __init__(self, lattice_size=100, num_types=3, record_every = 1,
                 record_lattice=True):

        self.lattice_size = lattice_size
        self.num_types = num_types
        self.record_every = record_every

        self.lattice = Lattice(lattice_size, num_types)

        self.time_array = None
        self.lattice_history = None
        self.coalescence_array = None
        self.annihilation_array = None
        self.num_walls_array = None

        self.record_lattice = record_lattice

    def run(self, max_time):

        num_record_steps = max_time / self.record_every + 1

        self.time_array = np.zeros(num_record_steps)

        if self.record_lattice:
            self.lattice_history = -1*np.ones((num_record_steps, self.lattice_size))
            self.lattice_history[0, :] = self.lattice.get_lattice_from_walls()

        self.coalescence_array = -1*np.ones(num_record_steps)
        self.annihilation_array = -1*np.ones(num_record_steps)
        self.num_walls_array = -1*np.ones(num_record_steps)

        self.coalescence_array[0] = 0
        self.annihilation_array[0] = 0
        self.num_walls_array[0] = self.lattice.walls.shape[0]

        cur_time = 0
        time_remainder = 0
        num_recorded = 1

        step_count = 0
        annihilation_count_per_time = 0
        coalescence_count_per_time = 0

        while (self.lattice.walls.shape[0] > 1) and (cur_time <= max_time):

            #### Debug
            print 'Before jump'
            print [z.position for z in self.lattice.walls]

            index = np.random.randint(0, self.lattice.walls.shape[0])
            current_wall = self.lattice.walls[index]
            # Determine time increment before deletion of walls
            delta_t = 1./self.lattice.walls.shape[0]

            #### Choose a jump direction ####
            jump_direction = current_wall.get_jump_direction()
            if jump_direction == RIGHT:
                current_wall.position += 1
                # No mod here, as we have to do extra stuff if there is a problem.
                if current_wall.position == self.lattice_size:
                    current_wall.position = 0
                    self.lattice.walls = np.roll(self.lattice.walls, 1)
            else:
                current_wall.position -= 1
                if current_wall.position < 0:
                    current_wall.position = self.lattice_size - 1
                    self.lattice.walls = np.roll(self.lattice.walls, -1)

            #### Debug ####
            print 'After jump'
            print [z.position for z in self.lattice.walls]

            #### Deal with collisions ####
            new_wall = None
            collision_type = None
            if jump_direction == LEFT:
                left_neighbor = current_wall.wall_neighbors[LEFT]
                print current_wall.position, left_neighbor.position
                if current_wall.position == left_neighbor.position:
                    print 'Collision!'
                    collision_type = self.lattice.collide(left_neighbor, current_wall)
            if jump_direction == RIGHT:
                right_neighbor = current_wall.wall_neighbors[RIGHT]
                print current_wall.position, right_neighbor.position
                if current_wall.position == right_neighbor.position:
                    print 'Collision!'
                    collision_type = self.lattice.collide(current_wall, right_neighbor)

            if collision_type is not None:
                if collision_type == ANNIHILATE:
                    annihilation_count_per_time += 1
                else:
                    coalescence_count_per_time += 1

            #### Debug #####
            print 'After collision'
            print [z.position for z in self.lattice.walls]
            print
            print

            #### Record information ####
            cur_time += delta_t
            time_remainder += delta_t

            if time_remainder >= self.record_every: # Record this step
                # Deal with keeping track of time
                self.time_array[num_recorded] = cur_time
                time_remainder -= self.record_every

                # Create lattice
                if self.record_lattice:
                    self.lattice_history[num_recorded, :] = self.lattice.get_lattice_from_walls()

                # Count annihilations & coalescences
                self.annihilation_array[num_recorded] = annihilation_count_per_time
                self.coalescence_array[num_recorded] = coalescence_count_per_time
                annihilation_count_per_time = 0
                coalescence_count_per_time = 0

                # Count the number of walls
                self.num_walls_array[num_recorded] = self.lattice.walls.shape[0]

                # Increment the number recorded

                num_recorded += 1

            step_count += 1

        if num_recorded == num_record_steps:
            print 'Used up available amount of time.'
        elif self.lattice.walls.shape[0] < 2:
            print 'There are less than two walls remaining.'

        print self.lattice.walls.shape[0] , 'walls remaining, done!'
        # Cut the output appropriately
        self.time_array = self.time_array[0:num_recorded]
        if self.record_lattice:
            self.lattice_history = self.lattice_history[0:num_recorded, :]
        self.annihilation_array = self.annihilation_array[0:num_recorded]
        self.coalescence_array = self.coalescence_array[0:num_recorded]
        self.num_walls_array = self.num_walls_array[0:num_recorded]

class Selection_Lattice_Simulation(Lattice_Simulation):

    def __init__(self, delta_prob_dict, lattice_size=100, num_types=3, record_every = 1,
                 record_lattice=True):
        Lattice_Simulation.__init__(self, lattice_size = lattice_size,
                                    num_types = num_types, record_every = record_every,
                                    record_lattice = record_lattice)
        self.delta_prob_dict = delta_prob_dict
        self.lattice = Selection_Lattice(delta_prob_dict, lattice_size, num_types=num_types)