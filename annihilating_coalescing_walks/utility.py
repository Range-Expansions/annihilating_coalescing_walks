import numpy as np
import pandas as pd
import annihilating_coalescing_walks.inflation as acwi
import numpy as np
import statsmodels.api as sm
import gc

def get_log_record_times(max_order, number_per_interval=100, include_first_two_orders=True):
    if number_per_interval > 200:
        print '200 point is the most that can be done per octave without' \
              'looking at a scale less than O(1)'
        return None
    else:
        if max_order == 1:
            return np.arange(1, 11)
        elif max_order == 2:
            return np.arange(1, 101)
        else:
            first_two_orders = None
            if include_first_two_orders:
                first_two_orders = np.arange(1, 100)
            num_octaves = max_order - 2
            rest_of_orders = np.logspace(2, max_order, number_per_interval*num_octaves)
            if include_first_two_orders:
                return np.hstack((first_two_orders, rest_of_orders))
            else:
                return rest_of_orders

def get_simulation_df(sim):
    # Let's make a pandas array of the results. It will be too annoying otherwise.
    wall_array = np.asarray(sim.num_walls_array)
    annih_array = np.asarray(sim.annihilation_array)
    coal_array = np.asarray(sim.coalescence_array)
    # Just make the time go from 0 to the max value to avoid problems.
    time_array = np.asarray(sim.time_array)
    df = pd.DataFrame(data={'wall_count': wall_array, 'annih': annih_array, 'coal':coal_array, 'time':time_array})

    return df

######## Averaging domain sizes from simulation ###############

def get_domain_size_df(completed_sim):
    angular_distances = []
    domain_types = []

    walls = np.asarray(completed_sim.wall_position_history)
    wall_types = np.asarray(completed_sim.wall_type_history)
    count = 0
    for cur_walls, cur_types in zip(walls, wall_types):
        cur_walls = np.array(cur_walls)
        cur_types = np.array(cur_types)
        deltaThetas = cur_walls[1:] - cur_walls[0:-1]
        deltaThetas = np.append(deltaThetas, cur_walls[0] - cur_walls[-1] + 2*np.pi)
        angular_distances.append(deltaThetas)
        cur_domain_types = [cur_types[z][0] for z in range(1, len(cur_types))]
        cur_domain_types = np.append(cur_domain_types, cur_types[0][0])
        domain_types.append(cur_domain_types)

        if np.any(angular_distances[count] < 0):
            print 'Walls must not have been ordered correctly...'
            print count

    time_array = np.asarray(completed_sim.time_array)

    # Now wrap the data in a DF
    df_list = []
    count = 0
    for cur_ang, cur_type, time_point in zip(angular_distances, domain_types, time_array):
        df_list.append(pd.DataFrame({'angular_distance': cur_ang,
                                     'time':time_point,
                                     'time_index' : count,
                                     'type': cur_type}))
        count += 1

    df_combined = pd.concat(df_list)

    return df_combined


def get_domain_size_ecdf(domain_size_df, type='all', num_ecdf_points=360):

    x = np.linspace(0, 2*np.pi, num_ecdf_points)
    # filter out the desired type
    if type != 'all':
        domain_size_df = domain_size_df.loc[domain_size_df['type'] == type, :]

    # groupby time index
    gb = domain_size_df.groupby('time_index')

    angular_ecdf_df_list = []
    for name, data in gb:
        ecdf = sm.distributions.ECDF(data['angular_distance'].values)
        cdf = ecdf(x)

        df = pd.DataFrame({'angle':x, 'domain_ecdf':cdf, 'time_index':name, 'time':data['time'].iloc[0]})
        angular_ecdf_df_list.append(df)

    combined_df = pd.concat(angular_ecdf_df_list)

    return x, combined_df

def get_total_fracs(sim):
    '''Given the domain size df, returns total fractions. Assumes seed, time_index, and type are all columns.'''

    domains = get_domain_size_df(sim)

    gb = domains.groupby(['time_index', 'type'])
    total_size_df = gb.agg(np.sum)
    total_size_df['frac'] = total_size_df['angular_distance']/(2*np.pi)

    # Deal with fractions that have vanished...bleh
    num_times = total_size_df.reset_index()['time_index'].unique()
    type_list = range(sim.lattice.num_types)

    new_idx = pd.MultiIndex.from_product([num_times,  type_list],
                                    names=['time_index', 'type'])
    replaced_total_df = total_size_df.reindex(index=new_idx)
    replaced_total_df['frac'].fillna(0, inplace=True)
    replaced_total_df['angular_distance'].fillna(0, inplace=True)

    # Drop columns that are misleading
    desired_cols = ['angular_distance', 'frac']
    to_drop = filter(lambda x: x not in desired_cols, replaced_total_df.columns.values)

    replaced_total_df.drop(to_drop, axis=1, inplace=True)

    return replaced_total_df


def get_collision_type_df(sim):

    collision_type_history = sim.collision_type_history

    time_array = []
    collisions = []

    for time_index, time_data in enumerate(collision_type_history):
        for collision_type in time_data:
            time_array.append(time_index)
            collision_type = np.array(collision_type)
            flattened_type = collision_type.flatten()
            collisions.append(flattened_type)

    time_array = np.array(time_array)
    collisions = np.array(collisions)

    collision_df = pd.DataFrame({'time_index':time_array,
                                 'i':collisions[:, 0],
                                 'j':collisions[:, 1],
                                 'k':collisions[:, 2],
                                 'l':collisions[:, 3]
                                })
    collision_df.set_index('time_index', inplace=True)

    # Set the times for simplicity
    times = np.asarray(sim.time_array)
    time_df = pd.DataFrame({'time':times, 'time_index':np.arange(0, times.shape[0])})
    time_df.set_index('time_index', inplace=True)

    collision_df = collision_df.join(time_df)

    return collision_df

def get_collision_type_count_df(sim):
    collisions = get_collision_type_df(sim)
    gb = collisions.reset_index().groupby(['i', 'j', 'k', 'l', 'time_index'])
    count_type_list = []

    for name, cur_data in gb:
        i, j, k, l = name[0], name[1], name[2], name[3]
        cur_time_index = name[4]
        cur_time = cur_data['time'].iloc[0]
        count_type_list.append([i, j, k, l, cur_data.shape[0], cur_time_index, cur_time])

    collision_type_summary = pd.DataFrame(data=count_type_list,
                                          columns=['i', 'j', 'k', 'l', 'num_events', 'time_index', 'time'])
    return collision_type_summary


############### Matching with Experiment #########################

# These are the parameters when the random walk approximation begins to hold
INITIAL_RADIUS = 3.50
VELOCITY = 1.19
JUMP_LENGTH = .4
#LATTICE_SIZE = lambda q: 22/(1.-1./float(q))
SUPERDIFFUSIVE_JUMP_LENGTH = 0.1

def get_sim_experimental_match(num_colors, lattice_size, s=0.0, record_lattice=False, lattice_spacing_output=2*np.pi/500.,
                               max_power=1, record_every=None, verbose=False, superdiffusive=False):

    debug=False

    num_types = num_colors
    seed = np.random.randint(0, 2**32)
    record_wall_position = False

    #record_every=10.

    record_time_array = None
    if record_every is None:
        record_time_array = get_log_record_times(max_power).astype(np.double)

    # 1 will have the selective advantage, like our experiments
    delta_prob_dict = {}

    ##########################################################
    ##### These parameters should be checked & fine tuned! ###
    ##########################################################
    radius = INITIAL_RADIUS
    velocity= VELOCITY
    if not superdiffusive:
        jump_length= JUMP_LENGTH
    else:
        jump_length = SUPERDIFFUSIVE_JUMP_LENGTH

    # Selective disadvantage for one of them...strain 1, like in the experiments
    for i in range(num_types):
        for j in range(num_types):
            if i != j:
                if i == 1:
                    delta_prob_dict[i, j] = -s
                elif j == 1:
                    delta_prob_dict[i, j] = s
                else:
                    delta_prob_dict[i, j] = 0

    if record_every is None:
        sim = acwi.Selection_Inflation_Lattice_Simulation(
            delta_prob_dict,
            lattice_size = lattice_size,
            num_types = num_types,
            seed=seed,
            record_lattice=record_lattice,
            record_time_array=record_time_array,
            velocity=velocity,
            radius=radius,
            jump_length=jump_length,
            record_wall_position=record_wall_position,
            lattice_spacing_output=lattice_spacing_output,
            debug=debug,
            verbose=verbose, superdiffusive=superdiffusive)
    else:
        sim = acwi.Selection_Inflation_Lattice_Simulation(
            delta_prob_dict,
            lattice_size = lattice_size,
            num_types = num_types,
            seed=seed,
            record_lattice=record_lattice,
            record_every=record_every,
            velocity=velocity,
            radius=radius,
            jump_length=jump_length,
            record_wall_position=record_wall_position,
            lattice_spacing_output=lattice_spacing_output,
            debug=debug,
            verbose=verbose, superdiffusive=superdiffusive)

    return sim