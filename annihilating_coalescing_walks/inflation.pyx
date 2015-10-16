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
cdef unsigned int NO_COLLISIONS = 2

import weakref

cdef long c_pos_mod(long num1, long num2) nogil:
    if num1 < 0:
        return num1 + num2
    else:
        return num1 % num2

cdef class Wall(object):

    cdef:
        public double position
        public  object[:] wall_neighbors
        public long[:] wall_type
        object __weakref__

    def __init__(Wall self, double position, object[:] wall_neighbors = None, long[:] wall_type = None):
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
                 object[:] wall_neighbors=None, long[:] wall_type = None,
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

cdef double ANGULAR_SIZE = 2*np.pi

cdef double a = 10.**-3. # E. coli size, a constant

cdef class Lattice(object):

    cdef:
        public long lattice_size
        public long[:] lattice_ic
        public long[:] lattice
        public Wall[:] walls
        public int num_types
        public bool debug
        public bool use_random_float_IC
        public bool use_specified_or_bio_IC
        public bool use_default_IC

    def __init__(Lattice self, long lattice_size, long num_types=3, bool debug=False,
                long[:] lattice=None, bool use_default_IC=True, bool use_random_float_IC=False,
                 bool use_specified_or_bio_IC=False):

        self.lattice_size = lattice_size
        self.num_types=num_types
        self.use_default_IC = use_default_IC
        self.use_random_float_IC = use_random_float_IC
        self.use_specified_or_bio_IC = use_specified_or_bio_IC
        self.debug = debug

        if self.use_random_float_IC or self.use_specified_or_bio_IC:
            self.use_default_IC = False
        elif self.use_default_IC:
            self.use_random_float_IC = True # Use the float IC as default
        else:
            print 'I have no idea what IC you want me to use! Quitting...'

        if self.use_specified_or_bio_IC:
            if lattice is None:
                L = self.lattice_size # This means something different in the float-based case
                self.lattice_ic = np.random.randint(0, num_types, L)
                self.lattice = self.lattice_ic.copy()
            else:
                self.lattice_ic = lattice
                self.lattice = self.lattice_ic.copy()
            self.walls = self.get_on_lattice_walls()
        elif self.use_random_float_IC:
            self.walls = self.get_off_lattice_walls()

    cdef Wall[:] get_on_lattice_walls(Lattice self):
        """Only to be used when initializing. If used again, terrible, terrible things will happen."""
        right_shift = np.roll(self.lattice, 1)
        wall_locations = self.lattice != right_shift
        wall_list = []
        wall_positions = np.where(wall_locations)[0]
        # Convert wall positions to angle!
        wall_positions = ANGULAR_SIZE * wall_positions/float(self.lattice_size)
        for cur_position in wall_positions:
            wall_list.append(self.get_new_wall(float(cur_position)))
        wall_list = np.array(wall_list)

        # Sort the wall list
        wall_list = np.sort(wall_list, axis=None)

        # Assign neighbors
        for i in range(wall_list.shape[0]):
            left_wall = wall_list[np.mod(i - 1, wall_list.shape[0])]
            right_wall = wall_list[np.mod(i + 1, wall_list.shape[0])]
            wall_list[i].wall_neighbors = np.array([weakref.proxy(left_wall), weakref.proxy(right_wall)])

        cdef long left_index
        cdef long right_index

        # Indicate what type of wall the wall is by looking back in the lattice
        cdef double to_lattice_factor = float(self.lattice_size)/ANGULAR_SIZE
        for i in range(wall_list.shape[0]):
            cur_wall = wall_list[i]
            cur_position = int(np.round(to_lattice_factor * wall_list[i].position))

            left_index = np.mod(cur_position - 1, self.lattice_size)
            right_index = cur_position

            left_of_kink = self.lattice[left_index]
            right_of_kink = self.lattice[right_index]
            cur_type = np.array([left_of_kink, right_of_kink])
            cur_wall.wall_type = cur_type

        return wall_list

    cdef Wall[:] get_off_lattice_walls(Lattice self): #TODO: If you input an IC, alternative code should be run!
        """Only to be used when initializing. If used again, terrible, terrible things will happen. This is a hacky
        way to take the IC off lattice in the inflation case, as if a spacing of 1 is used between points
        (the on lattice IC) strange things will happen. Kind of hackey but that's ok. Also assumes
        equal fractions of all colors, but we could fix that in the future."""

        num_walls = self.lattice_size
        wall_positions = np.random.rand(num_walls)*ANGULAR_SIZE
        # Create walls
        wall_list = []

        # Sort the positions
        wall_positions = np.sort(wall_positions)

        previous_wall_type = None
        new_wall_type = None
        count = 0
        for cur_position in wall_positions:
            if count == 0:
                previous_wall_type = np.random.randint(self.num_types)
                new_wall_type = np.random.randint(self.num_types)
                while new_wall_type == previous_wall_type: # The wall has to be different...
                    new_wall_type = np.random.randint(self.num_types)
                wall_type = np.array([previous_wall_type, new_wall_type], dtype=np.long)
                new_wall = self.get_new_wall(cur_position, wall_type=wall_type)
                wall_list.append(new_wall)
                previous_wall_type = new_wall_type
            elif count == num_walls - 1:
                new_wall_type = np.random.randint(self.num_types)
                while (new_wall_type == previous_wall_type) or (new_wall_type == wall_list[0].wall_type[RIGHT]): # The wall has to be different...
                    new_wall_type = np.random.randint(self.num_types)
                # Set the wall type of the first wall to the right of this one.
                wall_list[0].wall_type[LEFT] = new_wall_type
                wall_type = np.array([previous_wall_type, new_wall_type], dtype=np.long)
                new_wall = self.get_new_wall(cur_position, wall_type=wall_type)
                wall_list.append(new_wall)
            else:
                new_wall_type = np.random.randint(self.num_types)
                while new_wall_type == previous_wall_type: # The wall has to be different...
                    new_wall_type = np.random.randint(self.num_types)
                wall_type = np.array([previous_wall_type, new_wall_type], dtype=np.long)
                new_wall = self.get_new_wall(cur_position, wall_type=wall_type)
                wall_list.append(new_wall)
                previous_wall_type = new_wall_type
            count += 1

        wall_list = np.array(wall_list)

        # Assign neighbors
        for i in range(wall_list.shape[0]):
            left_wall = wall_list[np.mod(i - 1, wall_list.shape[0])]
            right_wall = wall_list[np.mod(i + 1, wall_list.shape[0])]
            wall_list[i].wall_neighbors = np.array([weakref.proxy(left_wall), weakref.proxy(right_wall)])

        return wall_list

    cpdef long[:] get_lattice_from_walls(Lattice self, double[:] output_bins_space):
        '''Returns the lattice array'''

        cdef long[:] output_lattice = -1*np.ones(output_bins_space.shape[0], dtype=np.long)

        cdef long num_walls = self.walls.shape[0]
        # Loop through the walls in terms of position

        cdef int i
        cdef object cur_wall

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

    cdef unsigned int collide(Lattice self, object left_wall, object right_wall, long left_wall_index):
        '''Collides two walls. Make sure the wall that was on the left
        is passed first. The left wall current_wall_index gives the current current_wall_index of the left wall.'''

        # Do a quick check to make sure nothing pathological has happened...
        if left_wall.wall_type[1] != right_wall.wall_type[0]:
            print 'SOMETHING TERRIBLE HAS HAPPENED...INCORRECT WALL COLLISION'

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
        cdef int cur_index
        cdef Wall cur_wall
        cdef bool[:] mask

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

            wall_before.wall_neighbors[1] = weakref.proxy(new_wall)
            new_wall.wall_neighbors = np.array([weakref.proxy(wall_before), weakref.proxy(wall_after)])
            wall_after.wall_neighbors[0] =weakref.proxy(new_wall)

            # Delete the undesired wall #TODO: Is this the problem with memory leakage?
            to_delete = np.array([c_pos_mod(left_wall_index + 1, self.walls.shape[0])])

            self.walls = np.delete(self.walls, to_delete)

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

            wall_before.wall_neighbors[1] = weakref.proxy(wall_after)
            wall_after.wall_neighbors[0] = weakref.proxy(wall_before)

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
        return Selection_Wall(new_position, wall_type=wall_type, delta_prob_dict=self.delta_prob_dict, wall_neighbors=wall_neighbors)

cdef class Inflation_Lattice_Simulation(object):

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
        public list wall_type_history
        public list collision_type_history
        public bool record_wall_position
        public bool record_collision_types
        public double[:] record_time_array

        public double[:] annihilation_array
        public double[:] coalescence_array
        public long[:] num_walls_array

        public unsigned long int seed

        public double initial_radius
        public double radius
        public double velocity
        public double jump_length
        public double lattice_spacing_output
        public double[:] output_bins_space
        public bool superdiffusive

        public double finish_time
        public int final_num_walls
        public bool mini_debug

    def __init__(Inflation_Lattice_Simulation self, double record_every = 1, bool record_lattice=True, bool debug=False,
                 unsigned long int seed = 0, record_time_array = None, bool verbose=True,
                 double radius=1.0, double velocity=0.01,
                 double lattice_spacing_output=ANGULAR_SIZE/180., bool record_wall_position=False,
                 double jump_length=0.001, bool superdiffusive=False, record_collision_types=False,
                 **kwargs):
        '''The idea here is the kwargs initializes the lattice.'''

        self.mini_debug = True

        self.record_every = record_every
        self.record_lattice = record_lattice
        self.debug=debug
        self.record_time_array = record_time_array
        self.verbose = verbose

        # Make sure the python seed is set before initializing the lattice...
        self.seed = seed
        np.random.seed(self.seed)

        self.time_array = None # Assumes the first time is always zero!
        self.lattice_history = None
        self.wall_position_history = None
        self.wall_type_history = None
        self.collision_type_history = None
        self.coalescence_array = None
        self.annihilation_array = None
        self.num_walls_array = None

        self.velocity = velocity
        self.jump_length = jump_length # Should be a hard constant, equal to 0.001mm, which are the units of the simulation!

        self.radius = radius
        if 'use_specified_or_bio_IC' in kwargs:
            if kwargs['use_specified_or_bio_IC']:
                print 'Replacing radius with expected one for Biological IC.' # Based on e.coli size and jump length
                self.radius = float(self.jump_length*kwargs['lattice_size'])/(2*np.pi)
                print 'Radius: ' , self.radius
        self.initial_radius = self.radius # Do this after the correction or *bad* things will happen

        self.superdiffusive = superdiffusive

        self.lattice_spacing_output = lattice_spacing_output
        self.output_bins_space = None

        self.record_wall_position = record_wall_position
        self.record_collision_types = record_collision_types

        self.finish_time = -1
        self.final_num_walls = -1

        self.lattice = self.initialize_lattice(**kwargs) # Initialize the lattice after the seed is set!

    def initialize_lattice(Inflation_Lattice_Simulation self, **kwargs):
        """Necessary for subclassing."""
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
            num_record_steps = self.record_time_array.shape[0] + 1 # Since we always record 0
            self.time_array = self.record_time_array.copy()
            # Actually append to the time array...say that we are recording at time 0.
            self.time_array = np.insert(self.time_array, 0, 0.0)

        if self.record_lattice:
            self.output_bins_space = np.arange(0, ANGULAR_SIZE + self.lattice_spacing_output, self.lattice_spacing_output)
            self.lattice_history = -1*np.ones((num_record_steps, self.output_bins_space.shape[0]), dtype=np.long)
            self.lattice_history[0, :] = self.lattice.get_lattice_from_walls(self.output_bins_space)
        if self.record_wall_position:
            self.wall_position_history = []
            self.wall_position_history.append([z.position for z in self.lattice.walls])
            self.wall_type_history = []
            self.wall_type_history.append([z.wall_type for z in self.lattice.walls])
        if self.record_collision_types:
            self.collision_type_history = [[]]

        self.coalescence_array = -1*np.ones(num_record_steps, dtype=np.double)
        self.annihilation_array = -1*np.ones(num_record_steps, dtype=np.double)
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
            unsigned int collision_type = NO_COLLISIONS
            object left_neighbor, right_neighbor
            long left_wall_index
            double distance_moved
            double random_num
            double p_of_x

            bool wrap_around_event = False

        while (self.lattice.walls.shape[0] > 1) and (cur_time <= max_time):
            #### Debug ####
            if self.debug:
                print 'Before jump'
                print [z.position for z in self.lattice.walls]
                # Also print integer positions
                print [int(np.round(self.lattice.lattice_size*z.position/ANGULAR_SIZE)) for z in self.lattice.walls]

            current_wall_index = gsl_rng_uniform_int(r, self.lattice.walls.shape[0])

            current_wall = self.lattice.walls[current_wall_index]
            if self.debug:
                print 'Current wall position:' , current_wall.position
            # Determine time increment before deletion of walls
            delta_t = 1./self.lattice.walls.shape[0]

            #### Determine how far you move ####
            if not self.superdiffusive:
                distance_moved = self.jump_length/self.radius
            else:
                # Draw from the superdiffusive distribution and move
                random_num = gsl_rng_uniform(r)
                p_of_x = (1-random_num)**(-2./3.)
                distance_moved = self.jump_length * p_of_x / self.radius

            #### Choose a jump direction ####
            wrap_around_event = False

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
                    adjusted_right_neighbor_position += ANGULAR_SIZE
                    wrap_around_event = True
                distance_between_walls = adjusted_right_neighbor_position - current_wall.position

                if distance_between_walls <= distance_moved: #Collision!'
                    if self.debug:
                        print 'Jump Right Collision!'

                    if wrap_around_event:
                        self.lattice.walls = np.roll(self.lattice.walls, 1)
                        current_wall_index = 0

                    current_wall.position = right_neighbor.position

                    if self.record_collision_types:
                        self.collision_type_history[num_recorded - 1].append([current_wall.wall_type, right_neighbor.wall_type])

                    collision_type = self.lattice.collide(current_wall, right_neighbor, current_wall_index)
                else: # Deal with wrapping
                    current_wall.position += distance_moved
                    if current_wall.position > ANGULAR_SIZE:
                        current_wall.position -= ANGULAR_SIZE
                        self.lattice.walls = np.roll(self.lattice.walls, 1)
                        current_wall_index = 0

            if jump_direction == LEFT: # Jump left
                left_neighbor = current_wall.wall_neighbors[LEFT]
                adjusted_left_neighbor_position = left_neighbor.position
                if adjusted_left_neighbor_position > current_wall.position: # There is a wrap around:
                    adjusted_left_neighbor_position -= ANGULAR_SIZE
                    wrap_around_event = True
                distance_between_walls = current_wall.position - adjusted_left_neighbor_position

                if distance_between_walls <= distance_moved: #Collision!'
                    if self.debug:
                        print 'Jump Left Collision!'
                    if wrap_around_event:
                        self.lattice.walls = np.roll(self.lattice.walls, -1)
                        current_wall_index = self.lattice.walls.shape[0] - 1

                    left_wall_index = c_pos_mod(current_wall_index - 1, self.lattice.walls.shape[0])
                    # Left neighbor position is where the collision should happen.
                    # TODO: Check if this is a wraparound collision!

                    if self.record_collision_types:
                        self.collision_type_history[num_recorded - 1].append([left_neighbor.wall_type, current_wall.wall_type])

                    collision_type = self.lattice.collide(left_neighbor, current_wall, left_wall_index)
                else: # Deal with wrapping
                    current_wall.position -= distance_moved
                    if current_wall.position < 0:
                        current_wall.position += ANGULAR_SIZE
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

                    # Add a new index to count annihilations and coalescences
                    if self.record_collision_types:
                        self.collision_type_history.append([])

                    # Increment the number recorded
                    time_remainder = 0
                    num_recorded += 1

                    if self.record_wall_position:
                        if self.lattice.walls.shape[0] != 0:
                            self.wall_position_history.append([z.position for z in self.lattice.walls])
                            self.wall_type_history.append([z.wall_type for z in self.lattice.walls])

            else: # Record at given times
                if num_recorded - 1 < self.record_time_array.shape[0]:
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

                        # Record collision types

                        if self.record_collision_types:
                            self.collision_type_history.append([])

                        # Record wall positions

                        if self.record_wall_position:
                            if self.lattice.walls.shape[0] != 0:
                                self.wall_position_history.append([z.position for z in self.lattice.walls])
                                self.wall_type_history.append([z.wall_type for z in self.lattice.walls])

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

            # Used to be a debug statement here...
            cur_positions = [wall.position for wall in self.lattice.walls]
            if sorted(cur_positions) != cur_positions:
                print 'The walls are not ordered correctly. Something terrible has happened.'
                print cur_positions

            #### Inflate! #####
            self.radius = self.initial_radius + self.velocity*cur_time

        #### Simulation is done; finish up. ####

        if self.verbose:
            if num_recorded == num_record_steps:
                print 'Used up available amount of time.'
            elif self.lattice.walls.shape[0] < 2:
                print 'There are less than two walls remaining.'

            print self.lattice.walls.shape[0] , 'walls remaining, done!'

        # Record the finish time
        self.finish_time = cur_time
        self.final_num_walls = self.lattice.walls.shape[0]
        # Cut the output appropriately

        self.time_array = self.time_array[0:num_recorded]
        if self.record_lattice:
            self.lattice_history = self.lattice_history[0:num_recorded, :]
        self.annihilation_array = self.annihilation_array[0:num_recorded]
        self.coalescence_array = self.coalescence_array[0:num_recorded]
        self.num_walls_array = self.num_walls_array[0:num_recorded]
        if self.record_collision_types:
            self.collision_type_history = self.collision_type_history[0:num_recorded]

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