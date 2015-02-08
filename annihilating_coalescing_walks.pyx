#cython: profile=True
#cython: boundscheck=True
#cython: initializedcheck=True
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

__author__ = 'bryan'

cimport cython
import numpy as np
cimport numpy as np
from cython_gsl cimport *
from cpython cimport bool

cdef unsigned int LEFT = 0
cdef unsigned int RIGHT = 1

cdef unsigned int ANNIHILATE = 0
cdef unsigned int COALESCE = 1

cdef class Wall:

    cdef:
        public long position
        public  Wall[:] wall_neighbors
        public long[:] wall_type

    def __init__(Wall self, long position, Wall[:] wall_neighbors = None, long[:] wall_type = None):
        self.position = position
        # Neighbors are neighboring walls! Left neighbor = 0, right neighbor = 1
        self.wall_neighbors = wall_neighbors
        # The wall type indicates the type of kink
        self.wall_type = wall_type

    def __richcmp__(Wall self, Wall other, int operation):
        if operation == 0: # Less than
            return self.position < other.position
        elif operation == 1: # less then or equal to
            return self.position <= other.position
        elif operation == 2: # equal to
            return self.position == other.position
        elif operation == 3: # not equal to
            return self.position != other.position
        elif operation == 4: # greater than
            return self.position > other.position
        elif operation == 5: #greater than or equal to
            return self.position >= other.position

    cdef unsigned int get_jump_direction(Wall self, gsl_rng *r):
        # Feed in gsl_rng *r as the random generator
        cdef double random_num = gsl_rng_uniform(r)
        if random_num < 0.5:
            return RIGHT
        else:
            return LEFT

cdef class Selection_Wall(Wall):
    '''One must define which wall gets a selective advantage. Let's say the biggest.'''

    cdef:
        dict delta_prob_dict

    def __init__(Selection_Wall self, long position,
                 Selection_Wall[:] wall_neighbors=None, long[:] wall_type = None,
                 dict delta_prob_dict = None):
        '''delta_prob_dict dictates what change in probability you get based on your neighbor.'''
        Wall.__init__(self, position, wall_neighbors = wall_neighbors, wall_type = wall_type)
        self.delta_prob_dict = delta_prob_dict

    cdef unsigned int get_jump_direction(Selection_Wall self, gsl_rng *r):
        # Get the change in probability of jumping left and right.
        cdef double right_prob_change = self.delta_prob_dict[self.wall_type[0], self.wall_type[1]]

        cdef double random_num = gsl_rng_uniform(r)
        if random_num < 0.5 + right_prob_change:
            return RIGHT
        else:
            return LEFT

cdef class Lattice:

    cdef:
        public long lattice_size
        public long[:] lattice
        public Wall[:] walls

    def __init__(Lattice self, long lattice_size, long num_types=3):
        self.lattice_size = lattice_size
        self.lattice = np.random.randint(0, num_types, lattice_size)
        self.walls = self.get_walls()

    cdef Wall[:] get_walls(Lattice self):
        right_shift = np.roll(self.lattice, 1)
        wall_locations = self.lattice != right_shift
        wall_list = []
        wall_positions = np.where(wall_locations)[0]
        for cur_position in wall_positions:
            wall_list.append(self.get_new_wall(cur_position))
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

    cpdef str get_lattice_str(Lattice self):
        '''Assumes walls are already sorted...as they should be..'''

        # Gives the lattice string SOLELY in terms of wall position
        cdef str output_str = ''
        cdef int num_walls = self.walls.shape[0]
        # Loop through the walls in terms of position

        cdef Wall cur_wall
        cdef int i

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

    cpdef long[:] get_lattice_from_walls(Lattice self):
        '''Returns the lattice array'''

        cdef long[:] output_lattice = -1*np.ones(self.lattice_size, dtype=np.long)
        cdef int num_walls = self.walls.shape[0]
        # Loop through the walls in terms of position

        cdef Wall cur_wall
        cdef int i

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

    cdef collide(Lattice self, Wall left_wall, Wall right_wall):
        '''Collides two walls. Make sure the wall that was on the left
        is passed first.'''

        cdef Wall new_wall = None
        cdef long type_after_collision_left = left_wall.wall_type[0]
        cdef long type_after_collision_right = right_wall.wall_type[1]

        cdef long new_position
        cdef long[:] new_type

        if type_after_collision_left != type_after_collision_right:
            new_position = left_wall.position
            new_type = np.array([type_after_collision_left, type_after_collision_right])
            new_wall = self.get_new_wall(new_position, wall_type = new_type)

        cdef bool[:] collided_indices = (self.walls == left_wall)
        cdef int[:] collision_indices = np.where(collided_indices)[0]
        if collision_indices.shape[0] < 2:
            print 'Something has gone very wrong, a wall appears to be colliding with itself.'
        cdef int first_index = collision_indices[0]

        cdef Wall wall_after, wall_before, wall_before_index, wall_two_after_index
        cdef int before_index, two_after_index
        cdef int[:] to_delete

        if new_wall is not None: # Coalesce
            # Redo neighbors first
            self.walls[first_index] = new_wall
            wall_after = self.walls[(first_index + 2) % self.walls.shape[0]]
            wall_before = self.walls[(first_index - 1) % self.walls.shape[0]]

            wall_before.wall_neighbors[1] = new_wall
            new_wall.wall_neighbors = np.array([wall_before, wall_after])
            wall_after.wall_neighbors[0] = new_wall

            # Delete the undesired wall
            self.walls = np.delete(self.walls, (first_index + 1) % self.walls.shape[0])
            return COALESCE
        else: #Annihilate
            # Redo neighbors before annihilation for simplicity

            before_index = (first_index -1) % self.walls.shape[0]
            two_after_index = (first_index +2) % self.walls.shape[0]

            wall_before_index = self.walls[before_index % self.walls.shape[0]]
            wall_two_after_index = self.walls[two_after_index % self.walls.shape[0]]

            wall_before_index.wall_neighbors[1] = wall_two_after_index
            wall_two_after_index.wall_neighbors[0] = wall_before_index

            # Do the actual annihilation
            to_delete = np.array([before_index, two_after_index])
            self.walls = np.delete(self.walls, to_delete)
            return ANNIHILATE

    cdef get_new_wall(self, new_position, wall_type=None, wall_neighbors=None):
        '''Creates a new wall appropriate for the lattice. Necessary for subclassing.'''
        return Wall(new_position, wall_type=wall_type, wall_neighbors=wall_neighbors)

cdef class Selection_Lattice(Lattice):

    def __init__(self, delta_prob_dict, lattice_size, num_types=3):
        self.delta_prob_dict = delta_prob_dict
        # If this is not done first, very bad things will happen. This needs to be defined
        # in order for walls to be created correctly.
        Lattice.__init__(self, lattice_size, num_types=num_types)

    def get_new_wall(self, new_position, wall_type=None, wall_neighbors = None):
        '''What is returned when a new wall is created via coalescence.'''
        return Selection_Wall(new_position, wall_type=wall_type, delta_prob_dict=self.delta_prob_dict)

class Lattice_Simulation():

    def __init__(self, lattice_size=100, num_types=3, record_every = 1,
                 record_lattice=True, debug=False):

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
        self.debug=debug

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

            #### Debug ####
            if self.debug:
                print 'Before jump'
                print [z.position for z in self.lattice.walls]

            index = np.random.randint(0, self.lattice.walls.shape[0])
            current_wall = self.lattice.walls[index]
            if self.debug:
                print 'Current wall position:' , current_wall.position
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
            if self.debug:
                print 'After jump'
                print [z.position for z in self.lattice.walls]

            #### Deal with collisions ####
            new_wall = None
            collision_type = None

            if jump_direction == LEFT:
                left_neighbor = current_wall.wall_neighbors[LEFT]
                if current_wall.position == left_neighbor.position:
                    if self.debug:
                        print 'Collision!'
                    collision_type = self.lattice.collide(left_neighbor, current_wall)
            if jump_direction == RIGHT:
                right_neighbor = current_wall.wall_neighbors[RIGHT]
                if current_wall.position == right_neighbor.position:
                    if self.debug:
                        print 'Collision!'
                    collision_type = self.lattice.collide(current_wall, right_neighbor)

            if collision_type is not None:
                if collision_type == ANNIHILATE:
                    annihilation_count_per_time += 1
                else:
                    coalescence_count_per_time += 1

            #### Debug #####
            if self.debug:
                print 'After collision'
                print [z.position for z in self.lattice.walls]

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

            if self.debug: #This takes a lot of time but helps to pinpoint problems.
                for i in range(self.lattice.walls.shape[0]):
                    current_wall = self.lattice.walls[i]
                    left_neighbor_position = current_wall.wall_neighbors[0].position
                    right_neighbor_position = current_wall.wall_neighbors[1].position

                    actual_left_position = self.lattice.walls[(i-1) % self.lattice.walls.shape[0]].position
                    actual_right_position = self.lattice.walls[(i+1) % self.lattice.walls.shape[0]].position

                    if (left_neighbor_position != actual_left_position) or (right_neighbor_position != actual_right_position):
                        print 'Neighbors are messed up. Neighbor positions and actual positions do not line up.'
                        print 'Problem position is ', current_wall.position
                        print 'It thinks neighbors are' , left_neighbor_position , right_neighbor_position
                        print 'Its neigbors actually are', actual_left_position, actual_right_position
                print
                print

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
                 record_lattice=True, debug=False):
        Lattice_Simulation.__init__(self, lattice_size = lattice_size,
                                    num_types = num_types, record_every = record_every,
                                    record_lattice = record_lattice, debug=debug)
        self.delta_prob_dict = delta_prob_dict
        self.lattice = Selection_Lattice(delta_prob_dict, lattice_size, num_types=num_types)