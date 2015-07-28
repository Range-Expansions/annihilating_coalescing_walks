import numpy as np
import pandas as pd
import annihilating_coalescing_walks as acw

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