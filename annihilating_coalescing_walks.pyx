#cython: profile=False
#cython: boundscheck=False
#cython: initializedcheck=False
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

cdef long c_pos_mod(long num1, long num2) nogil:
    if num1 < 0:
        return num1 + num2
    else:
        return num1 % num2

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

    def __cmp__(Wall self, Wall other):
        if self.position < other.position:
            return -1
        elif self.position == other.position:
            return 0
        else:
            return 1

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
        public dict delta_prob_dict

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
        bool debug

    def __init__(Lattice self, long lattice_size, long num_types=3, bool debug=False):
        self.lattice_size = lattice_size
        self.lattice = np.random.randint(0, num_types, lattice_size)
        self.walls = self.get_walls()
        self.debug = debug

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

        cdef long left_index
        cdef long right_index

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
        cdef long num_walls = self.walls.shape[0]
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

    cdef unsigned int collide(Lattice self, Wall left_wall, Wall right_wall, long left_wall_index):
        '''Collides two walls. Make sure the wall that was on the left
        is passed first. The left wall current_wall_index gives the current current_wall_index of the left wall.'''

        cdef Wall new_wall = None
        cdef long type_after_collision_left = left_wall.wall_type[0]
        cdef long type_after_collision_right = right_wall.wall_type[1]

        cdef long new_position
        cdef long[:] new_type

        if type_after_collision_left != type_after_collision_right:
            new_position = left_wall.position
            new_type = np.array([type_after_collision_left, type_after_collision_right])
            new_wall = self.get_new_wall(new_position, wall_type = new_type)

        cdef Wall wall_after, wall_before
        cdef long before_index, after_index
        cdef long[:] to_delete

        if self.walls[c_pos_mod(left_wall_index + 1, self.walls.shape[0])].position != self.walls[left_wall_index].position:
            print 'Something is screwed up...walls are colliding with themselves?'

        if new_wall is not None: # Coalesce
            if self.debug:
                print 'Coalesce!'
                print 'Left wall current_wall_index is' , left_wall_index
            # Redo neighbors first
            self.walls[left_wall_index] = new_wall

            before_index = c_pos_mod(left_wall_index - 1, self.walls.shape[0])
            after_index = c_pos_mod(left_wall_index + 2, self.walls.shape[0])

            wall_after = self.walls[after_index]
            wall_before = self.walls[before_index]

            wall_before.wall_neighbors[1] = new_wall
            new_wall.wall_neighbors = np.array([wall_before, wall_after])
            wall_after.wall_neighbors[0] = new_wall

            # Delete the undesired wall
            self.walls = np.delete(self.walls, c_pos_mod(left_wall_index + 1, self.walls.shape[0]))
            return COALESCE
        else: #Annihilate
            if self.debug:
                print 'Annihilate!'
                print 'Left wall current_wall_index is' , left_wall_index
            # Redo neighbors before annihilation for simplicity

            before_index = c_pos_mod(left_wall_index - 1, self.walls.shape[0])
            after_index = c_pos_mod(left_wall_index +2, self.walls.shape[0])

            wall_before = self.walls[c_pos_mod(before_index, self.walls.shape[0])]
            wall_after = self.walls[c_pos_mod(after_index, self.walls.shape[0])]

            wall_before.wall_neighbors[1] = wall_after
            wall_after.wall_neighbors[0] = wall_before

            # Do the actual annihilation
            to_delete = np.array([left_wall_index, c_pos_mod(left_wall_index + 1, self.walls.shape[0])])
            self.walls = np.delete(self.walls, to_delete)
            return ANNIHILATE

    cdef get_new_wall(self, new_position, wall_type=None, wall_neighbors=None):
        '''Creates a new wall appropriate for the lattice. Necessary for subclassing.'''
        return Wall(new_position, wall_type=wall_type, wall_neighbors=wall_neighbors)

cdef class Selection_Lattice(Lattice):
    cdef:
        public dict delta_prob_dict

    def __init__(self, delta_prob_dict, lattice_size, num_types=3, bool debug = False):
        self.delta_prob_dict = delta_prob_dict
        # If this is not done first, very bad things will happen. This needs to be defined
        # in order for walls to be created correctly.
        Lattice.__init__(self, lattice_size, num_types=num_types, debug=debug)

    cdef get_new_wall(self, new_position, wall_type=None, wall_neighbors = None):
        '''What is returned when a new wall is created via coalescence.'''
        return Selection_Wall(new_position, wall_type=wall_type, delta_prob_dict=self.delta_prob_dict)

cdef class Lattice_Simulation:

    cdef:
        public long lattice_size
        public long num_types
        public double record_every
        public bool record_lattice
        public bool debug

        public Lattice lattice
        public double[:] time_array
        public long[:, :] lattice_history

        public long[:] annihilation_array
        public long[:] coalescence_array
        public long[:] num_walls_array

        public unsigned long int seed

    def __init__(Lattice_Simulation self, long lattice_size=100, long num_types=3, double record_every = 1,
                 bool record_lattice=True, bool debug=False, unsigned long int seed = 0):

        self.lattice_size = lattice_size
        self.num_types = num_types
        self.record_every = record_every
        self.record_lattice = record_lattice
        self.debug=debug
        self.seed = seed

        self.lattice = Lattice(lattice_size, num_types, debug=self.debug)

        self.time_array = None
        self.lattice_history = None
        self.coalescence_array = None
        self.annihilation_array = None
        self.num_walls_array = None

    def run(Lattice_Simulation self, double max_time):
        '''This should only be run once! Weird things will happen otherwise as the seed will be weird.'''
        # Initialize the random number generator
        np.random.seed(self.seed)
        cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(r, self.seed)


        cdef long num_record_steps = long(max_time / self.record_every) + 1

        self.time_array = np.zeros(num_record_steps, dtype=np.double)

        if self.record_lattice:
            self.lattice_history = -1*np.ones((num_record_steps, self.lattice_size), dtype=np.long)
            if self.record_lattice:
                self.lattice_history[0, :] = self.lattice.get_lattice_from_walls()

        self.coalescence_array = -1*np.ones(num_record_steps, dtype=np.long)
        self.annihilation_array = -1*np.ones(num_record_steps, dtype=np.long)
        self.num_walls_array = -1*np.ones(num_record_steps, dtype=np.long)

        self.coalescence_array[0] = 0
        self.annihilation_array[0] = 0
        self.num_walls_array[0] = self.lattice.walls.shape[0]

        cdef double cur_time = 0
        cdef double time_remainder = 0
        cdef int num_recorded = 1

        cdef int step_count = 0
        cdef int annihilation_count_per_time = 0
        cdef int coalescence_count_per_time = 0

        cdef:
            unsigned int current_wall_index
            Wall current_wall
            double delta_t
            unsigned int jump_direction
            unsigned int collision_type
            Wall left_neighbor, right_neighbor
            cdef long left_wall_index

        while (self.lattice.walls.shape[0] > 1) and (cur_time <= max_time):

            #### Debug ####
            if self.debug:
                print 'Before jump'
                print [z.position for z in self.lattice.walls]

            current_wall_index = gsl_rng_uniform_int(r, self.lattice.walls.shape[0])
            current_wall = self.lattice.walls[current_wall_index]
            if self.debug:
                print 'Current wall position:' , current_wall.position
            # Determine time increment before deletion of walls
            delta_t = 1./self.lattice.walls.shape[0]

            #### Choose a jump direction ####
            jump_direction = current_wall.get_jump_direction(r)
            if jump_direction == RIGHT:
                current_wall.position += 1
                # No mod here, as we have to do extra stuff if there is a problem.
                if current_wall.position == self.lattice_size:
                    current_wall.position = 0
                    self.lattice.walls = np.roll(self.lattice.walls, 1)
                    current_wall_index = 0
            else:
                current_wall.position -= 1
                if current_wall.position < 0:
                    current_wall.position = self.lattice_size - 1
                    self.lattice.walls = np.roll(self.lattice.walls, -1)
                    current_wall_index = self.lattice.walls.shape[0] - 1

            #### Debug ####
            if self.debug:
                print 'After jump'
                print [z.position for z in self.lattice.walls]

            #### Deal with collisions ####

            if jump_direction == LEFT:
                left_neighbor = current_wall.wall_neighbors[LEFT]
                if current_wall.position == left_neighbor.position:
                    if self.debug:
                        print 'Jump Left Collision!'
                    left_wall_index = c_pos_mod(current_wall_index - 1, self.lattice.walls.shape[0])
                    collision_type = self.lattice.collide(left_neighbor, current_wall, left_wall_index)
            if jump_direction == RIGHT:
                right_neighbor = current_wall.wall_neighbors[RIGHT]
                if current_wall.position == right_neighbor.position:
                    if self.debug:
                        print 'Jump Right Collision!'
                    collision_type = self.lattice.collide(current_wall, right_neighbor, current_wall_index)

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

                    actual_left_position = self.lattice.walls[c_pos_mod(i-1, self.lattice.walls.shape[0])].position
                    actual_right_position = self.lattice.walls[c_pos_mod(i+1, self.lattice.walls.shape[0])].position

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

        # DONE! Deallocate as necessary.
        gsl_rng_free(r)

cdef class Selection_Lattice_Simulation(Lattice_Simulation):

    cdef:
        public dict delta_prob_dict

    def __init__(Selection_Lattice_Simulation self, dict delta_prob_dict, long lattice_size=100, long num_types=3,
                 double record_every = 1, bool record_lattice=True, bool debug=False):
        Lattice_Simulation.__init__(self, lattice_size = lattice_size,
                                    num_types = num_types, record_every = record_every,
                                    record_lattice = record_lattice, debug=debug)
        self.delta_prob_dict = delta_prob_dict
        self.lattice = Selection_Lattice(delta_prob_dict, lattice_size, num_types=num_types, debug=self.debug)