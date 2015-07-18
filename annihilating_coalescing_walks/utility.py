import numpy as np
import pandas as pd
import annihilating_coalescing_walks as acw

def get_log_record_times(min_power, max_power, num_to_record = 300):
    return np.logspace(min_power, max_power, num_to_record)

def get_simulation_df(sim, max_time_power = 8):

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