# Code for matching with experimental correlation functions. A utility, but important.

import numpy as np
import pandas as pd
import utility as acwu
import inflation as acwi


# Measured values
Dw_experiment = 0.100
Ro_experiment = 3.5
# Where I recorded the two-point correlation functions
L_experiment_values = np.array([0.5, 1.5, 2.5, 4.5, 6.5])

a = 0.001 # Size of a "lattice site" in mm (units of the simulation)
v = a # Distance that the lattice expands per generation

class Exp_Corr_Matcher(object):
    def __init__(self, Ls_experiment, Ls_sim, num_colors=3):
        # Experimental values
        self.Ls_experiment = Ls_experiment
        self.L_div_Ls_experiment = L_experiment_values / Ls_experiment

        # kappa is matched between experiment & theory
        self.kappa = np.sqrt(Ro_experiment / Ls_experiment)

        # Simulation values
        self.Ls_sim = Ls_sim
        self.s_sim = np.sqrt(a/(2*Ls_sim))
        self.No_sim = np.pi*(self.kappa/self.s_sim)**2
        self.No_sim = int(round(self.No_sim))
        print 'No_sim is:', self.No_sim

        self.num_colors = num_colors

        print 'Initializing simulation...'
        self.sim = self.get_sim()
        print 'Done!'

    def get_sim(self, **kwargs):
        # Convert into simulation units
        num_types = self.num_colors
        seed = np.random.randint(0, 2 ** 32)

        # Match L/Ls between experiment and simulation
        L_values = self.L_div_Ls_experiment * self.Ls_sim

        record_time_array = L_values / v
        # record_time_array = record_time_array[1:]

        # 1 will have the selective disadvantage, like our experiments
        delta_prob_dict = {}

        ##########################################################
        ##### These parameters should be checked & fine tuned! ###
        ##########################################################

        # Selective disadvantage for one of them...strain 1, like in the experiments
        for i in range(num_types):
            for j in range(num_types):
                if i != j:
                    if i == 1:
                        delta_prob_dict[i, j] = -self.s_sim
                    elif j == 1:
                        delta_prob_dict[i, j] = self.s_sim
                    else:
                        delta_prob_dict[i, j] = 0

        sim = acwi.Selection_Inflation_Lattice_Simulation(
            delta_prob_dict,
            lattice_size=self.No_sim,
            num_types=num_types,
            seed=seed,
            record_time_array=record_time_array,
            velocity=v,
            use_specified_or_bio_IC=True,
            record_lattice=True,
            lattice_spacing_output=2 * np.pi / (360 * 8.),
            **kwargs)

        return sim