#cython: profile=False
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
cdef unsigned int NO_COLLISIONS = 2

cdef long c_pos_mod(long num1, long num2) nogil:
    if num1 < 0:
        return num1 + num2
    else:
        return num1 % num2

cdef class Wall:

    cdef:
        public double position
        public  Wall[:] wall_neighbors
        public long[:] wall_type

    def __init__(Wall self, double position, Wall[:] wall_neighbors = None, long[:] wall_type = None):
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
    '''One must define which wall gets a selective advantage.'''

    cdef:
        public dict delta_prob_dict

    def __init__(Selection_Wall self, double position,
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
        public long[:] lattice_ic
        public long[:] lattice
        public Wall[:] walls
        bool debug

    def __init__(Lattice self, long lattice_size, long num_types=3, bool debug=False, long[:] lattice=None):
        self.lattice_size = lattice_size
        if lattice is None:
            self.lattice_ic = np.random.randint(0, num_types, lattice_size)
            self.lattice = self.lattice_ic.copy()
        else:
            self.lattice_ic = lattice
            self.lattice = self.lattice_ic.copy()
        self.walls = self.get_walls()
        self.debug = debug

    cdef reset(self):
        self.lattice = self.lattice_ic.copy()
        self.walls = self.get_walls()

    cdef Wall[:] get_walls(Lattice self):
        """Only to be used when initializing. If used again, terrible, terrible things will happen."""
        right_shift = np.roll(self.lattice, 1)
        wall_locations = self.lattice != right_shift
        wall_list = []
        wall_positions = np.where(wall_locations)[0]
        for cur_position in wall_positions:
            wall_list.append(self.get_new_wall(float(cur_position)))
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
            cur_position = int(np.round(wall_list[i].position))

            left_index = np.mod(cur_position - 1, self.lattice_size)
            right_index = cur_position

            left_of_kink = self.lattice[left_index]
            right_of_kink = self.lattice[right_index]
            cur_type = np.array([left_of_kink, right_of_kink])
            cur_wall.wall_type = cur_type

        return wall_list

    cpdef long[:] get_lattice_from_walls(Lattice self, double[:] output_bins_space):
        '''Returns the lattice array'''

        cdef long[:] output_lattice = -1*np.ones(output_bins_space.shape[0], dtype=np.long)

        cdef long num_walls = self.walls.shape[0]
        # Loop through the walls in terms of position

        cdef int i
        cdef Wall cur_wall

        cdef int num_walls_passed = 0
        if num_walls > 1:
            cur_wall = self.walls[0]
            for i in range(output_bins_space.shape[0]):
                if (cur_wall.position <= output_bins_space[i]) and (num_walls_passed != self.walls.shape[0]):
                    cur_wall = cur_wall.wall_neighbors[RIGHT]
                    num_walls_passed += 1
                output_lattice[i] = cur_wall.wall_type[LEFT]
        elif num_walls == 1:
            cur_wall = self.walls[0]
            for i in range(output_bins_space.shape[0]):
                if (output_bins_space[i] < cur_wall.position) and (num_walls_passed != self.walls.shape[0]):
                    output_lattice[i] = cur_wall.wall_type[0]
                    num_walls_passed += 1
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

        cdef double new_position
        cdef long[:] new_type

        if type_after_collision_left != type_after_collision_right:
            new_position = left_wall.position
            new_type = np.array([type_after_collision_left, type_after_collision_right])
            new_wall = self.get_new_wall(new_position, wall_type = new_type)

        cdef Wall wall_after, wall_before
        cdef long before_index, after_index
        cdef long[:] to_delete

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

    cdef get_new_wall(self, double new_position, wall_type=None, wall_neighbors=None):
        '''Creates a new wall appropriate for the lattice. Necessary for subclassing.'''
        return Wall(new_position, wall_type=wall_type, wall_neighbors=wall_neighbors)

cdef class Selection_Lattice(Lattice):
    cdef:
        public dict delta_prob_dict

    def __init__(self, delta_prob_dict, **kwargs):
        self.delta_prob_dict = delta_prob_dict
        # If this is not done first, very bad things will happen. This needs to be defined
        # in order for walls to be created correctly.
        Lattice.__init__(self, **kwargs)

    cdef get_new_wall(self, double new_position, wall_type=None, wall_neighbors = None):
        '''What is returned when a new wall is created via coalescence.'''
        return Selection_Wall(new_position, wall_type=wall_type, delta_prob_dict=self.delta_prob_dict)

cdef class Inflation_Lattice_Simulation:

    cdef:
        public double record_every
        public bool record_lattice
        public bool debug
        public bool verbose
        public bool record_coal_annih_type

        public Lattice lattice
        public double[:] time_array
        public long[:, :] lattice_history
        public list wall_position_history
        public bool record_wall_position
        public double[:] record_time_array

        public double[:] annihilation_array
        public double[:] coalescence_array
        public long[:] num_walls_array

        public unsigned long int seed

        public double radius
        public double velocity
        public double lattice_spacing_output
        public double[:] output_bins_space

    def __init__(Inflation_Lattice_Simulation self, double record_every = 1, bool record_lattice=True, bool debug=False,
                 unsigned long int seed = 0, record_time_array = None, bool verbose=True,
                 record_coal_annih_type = False, double radius=1.0, double velocity=0.01, double lattice_spacing_output=0.5,
                 bool record_wall_position=False,
                 **kwargs):
        '''The idea here is the kwargs initializes the lattice.'''
        self.record_every = record_every
        self.record_lattice = record_lattice
        self.debug=debug
        self.record_time_array = record_time_array
        self.verbose = verbose
        self.record_coal_annih_type = record_coal_annih_type

        # Make sure the python seed is set before initializing the lattice...
        self.seed = seed
        np.random.seed(self.seed)
        self.lattice = self.initialize_lattice(**kwargs)

        self.time_array = None # Assumes the first time is always zero!
        self.lattice_history = None
        self.wall_position_history = None
        self.coalescence_array = None
        self.annihilation_array = None
        self.num_walls_array = None

        self.radius = radius
        self.velocity = velocity

        self.lattice_spacing_output = lattice_spacing_output
        self.output_bins_space = None

        self.record_wall_position = record_wall_position


    def initialize_lattice(Inflation_Lattice_Simulation self, **kwargs):
        '''Necessary for subclassing.'''
        return Lattice(debug=self.debug, **kwargs)

    def reset(Inflation_Lattice_Simulation self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        self.lattice.reset()

        self.time_array = None
        self.lattice_history = None
        self.coalescence_array = None
        self.annihilation_array = None
        self.num_walls_array = None


    def run(Inflation_Lattice_Simulation self, double max_time):
        '''This should only be run once! Weird things will happen otherwise as the seed will be weird.'''
        # Initialize the random number generator

        cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(r, self.seed)

        cdef long num_record_steps
        if self.record_time_array is None:  # If the user doesn't send in a time array, record at record_every
             num_record_steps = long(max_time / self.record_every) + 1
             self.time_array = np.zeros(num_record_steps, dtype=np.double)
        else: # Record at specified times
            num_record_steps = self.record_time_array.shape[0]
            self.time_array = self.record_time_array.copy()

        self.output_bins_space = np.arange(0, self.lattice.lattice_size + self.lattice_spacing_output, self.lattice_spacing_output)
        if self.record_lattice:
            self.lattice_history = -1*np.ones((num_record_steps, self.output_bins_space.shape[0]), dtype=np.long)
            if self.record_lattice:
                self.lattice_history[0, :] = self.lattice.get_lattice_from_walls(self.output_bins_space)
        if self.record_wall_position:
            self.wall_position_history = []
            self.wall_position_history.append([z.position for z in self.lattice.walls])

        self.coalescence_array = -1*np.ones(num_record_steps + 1, dtype=np.double)
        self.annihilation_array = -1*np.ones(num_record_steps + 1, dtype=np.double)
        self.num_walls_array = -1*np.ones(num_record_steps + 1, dtype=np.long)

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
            unsigned int collision_type = NO_COLLISIONS
            Wall left_neighbor, right_neighbor
            long left_wall_index
            double distance_moved

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
            if self.debug:
                if jump_direction is LEFT:
                    print 'Jump left!'
                if jump_direction is RIGHT:
                    print 'Jump right!'
            if jump_direction == RIGHT:
                right_neighbor = current_wall.wall_neighbors[RIGHT]
                adjusted_right_neighbor_position = right_neighbor.position
                if adjusted_right_neighbor_position < current_wall.position: # There is a wrap around:
                    adjusted_right_neighbor_position += self.lattice.lattice_size
                distance_between_walls = adjusted_right_neighbor_position - current_wall.position

                distance_moved = 1./self.radius

                if distance_between_walls <= distance_moved: #Collision!'
                    if self.debug:
                        print 'Jump Right Collision!'
                    current_wall.position = right_neighbor.position
                    collision_type = self.lattice.collide(current_wall, right_neighbor, current_wall_index)
                else: # Deal with wrapping
                    current_wall.position += distance_moved
                    if current_wall.position > self.lattice.lattice_size:
                        current_wall.position -= self.lattice.lattice_size
                        self.lattice.walls = np.roll(self.lattice.walls, 1)
                        current_wall_index = 0

            if jump_direction == LEFT: # Jump left
                left_neighbor = current_wall.wall_neighbors[LEFT]
                adjusted_left_neighbor_position = left_neighbor.position
                if adjusted_left_neighbor_position > current_wall.position: # There is a wrap around:
                    adjusted_left_neighbor_position -= self.lattice.lattice_size
                distance_between_walls = current_wall.position - adjusted_left_neighbor_position

                distance_moved = 1./self.radius

                if distance_between_walls <= distance_moved: #Collision!'
                    if self.debug:
                        print 'Jump Left Collision!'
                    left_wall_index = c_pos_mod(current_wall_index - 1, self.lattice.walls.shape[0])
                    # Left neighbor position is where the collision should happen.
                    collision_type = self.lattice.collide(left_neighbor, current_wall, left_wall_index)
                else: # Deal with wrapping
                    current_wall.position -= distance_moved
                    if current_wall.position < 0:
                        current_wall.position += self.lattice.lattice_size
                        self.lattice.walls = np.roll(self.lattice.walls, -1)
                        current_wall_index = self.lattice.walls.shape[0] - 1

            #### Count collisions ####

            if collision_type is not NO_COLLISIONS:
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

            if self.record_time_array is None: # Record at record every
                if time_remainder >= self.record_every: # Record this step
                    # Deal with keeping track of time
                    self.time_array[num_recorded] = cur_time

                    # Create lattice
                    if self.record_lattice:
                        self.lattice_history[num_recorded, :] = self.lattice.get_lattice_from_walls(self.output_bins_space)

                    # Count annihilations & coalescences
                    self.annihilation_array[num_recorded] = annihilation_count_per_time/time_remainder
                    self.coalescence_array[num_recorded] = coalescence_count_per_time/time_remainder
                    annihilation_count_per_time = 0
                    coalescence_count_per_time = 0

                    # Count the number of walls
                    self.num_walls_array[num_recorded] = self.lattice.walls.shape[0]

                    # Increment the number recorded
                    time_remainder = 0
                    num_recorded += 1

                    if self.record_wall_position:
                        if self.lattice.walls.shape[0] != 0:
                            self.wall_position_history.append([z.position for z in self.lattice.walls])

            else: # Record at given times
                if cur_time >= self.record_time_array[num_recorded - 1]: # Need the minus 1, as the zero is always recorded

                    # Create lattice
                    if self.record_lattice:
                        self.lattice_history[num_recorded, :] = self.lattice.get_lattice_from_walls(self.output_bins_space)

                    # Count annihilations & coalescences
                    self.annihilation_array[num_recorded] = annihilation_count_per_time/time_remainder
                    self.coalescence_array[num_recorded] = coalescence_count_per_time/time_remainder
                    annihilation_count_per_time = 0
                    coalescence_count_per_time = 0

                    # Count the number of walls
                    self.num_walls_array[num_recorded] = self.lattice.walls.shape[0]

                    # Increment the number recorded
                    num_recorded += 1

                    time_remainder = 0 # time-remainder acts as the delta t in this case.

                    # Record wall positions

                    if self.record_wall_position:
                        if self.lattice.walls.shape[0] != 0:
                            self.wall_position_history.append([z.position for z in self.lattice.walls])

            # Refresh for the next time interval
            step_count += 1
            collision_type = NO_COLLISIONS

            #### Debug if necessary ####
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

            if self.debug:
                cur_positions = [wall.position for wall in self.lattice.walls]
                if sorted(cur_positions) != cur_positions:
                    print 'The walls are not ordered correctly. Something terrible has happened.'
                    print cur_positions
                print
                print

            #### Inflate! #####
            self.radius +=  delta_t * self.velocity

        #### Simulation is done; finish up. ####

        if self.verbose:
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

cdef class Selection_Inflation_Lattice_Simulation(Inflation_Lattice_Simulation):

    cdef:
        public dict delta_prob_dict

    def __init__(Selection_Inflation_Lattice_Simulation self, dict delta_prob_dict, **kwargs):

        self.delta_prob_dict = delta_prob_dict
        Inflation_Lattice_Simulation.__init__(self, **kwargs)

    def initialize_lattice(Selection_Inflation_Lattice_Simulation self, **kwargs):
        '''Necessary for subclassing.'''
        return Selection_Lattice(self.delta_prob_dict, debug=self.debug, **kwargs)