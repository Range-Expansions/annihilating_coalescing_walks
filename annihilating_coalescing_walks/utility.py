import numpy as np
import pandas as pd
import annihilating_coalescing_walks.inflation as acwi

def get_log_record_times(max_order, number_per_interval=100):
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
            first_two_orders = np.arange(1, 101)
            num_octaves = max_order - 2
            rest_of_orders = np.logspace(2, max_order, number_per_interval*num_octaves)
            return np.hstack((first_two_orders, rest_of_orders))

def get_simulation_df(sim, max_time_power = 8, run_sim=True):
    if run_sim:
        sim.run(10**max_time_power)

    # Let's make a pandas array of the results. It will be too annoying otherwise.
    wall_array = np.asarray(sim.num_walls_array)
    annih_array = np.asarray(sim.annihilation_array)
    coal_array = np.asarray(sim.coalescence_array)
    # Just make the time go from 0 to the max value to avoid problems.
    time_array = np.asarray(sim.time_array)

    df = pd.DataFrame(data={'wall_count': wall_array, 'annih': annih_array, 'coal':coal_array, 'time':time_array})

    return df

def average_simulations(sim, num_simulations = 100, **kwargs):
    '''Creates many simulations based off of sim and returns the combined dataframes.'''

    df_list = []
    for i in range(num_simulations):
        new_df = get_simulation_df(sim, **kwargs)
        new_df['sim_num'] = i
        df_list.append(new_df)
        new_seed = np.random.randint(0, 2**32 - 1)
        sim.reset(new_seed)
    return df_list

INITIAL_RADIUS = 1.93
VELOCITY = 1.19
JUMP_LENGTH = 790

def get_sim_experimental_match(num_colors, s=0.0, record_lattice=False, max_power=2, record_every=None):

    debug=False

    lattice_size = 10**4
    num_types = num_colors
    seed = np.random.randint(0, 2**32)
    record_wall_position = False
    lattice_spacing_output = 1.0

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
    jump_length= JUMP_LENGTH


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
            debug=debug)
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
            debug=debug)

    return sim