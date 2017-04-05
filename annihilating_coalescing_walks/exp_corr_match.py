# Code for matching with experimental correlation functions. A utility, but important.

import numpy as np
import pandas as pd
import utility as acwu
import inflation as acwi


#### Measured values ####
Dw_experiment = 0.100
Ro_experiment = 3.5
# Where I recorded the two-point correlation functions
#L_experiment_values = np.array([0.5, 1.5, 2.5, 4.5, 5.5, 6.5])
L_experiment_values = np.array([0.5, 1.5, 2.5, 4.5, 6.5])

#### Simulation values ####
a = 0.001 # Size of a "lattice site" in mm (units of the simulation)
v = a # Distance that the lattice expands per generation

Dw_sim = a/2. # Simulation diffusion constant per length expanded

class Exp_Corr_Matcher(object):
    def __init__(self, Ls_experiment, Ls_sim, num_colors=3, unequal_ic=None, delta_prob_dict=None,
                 L_to_record=None, record_wall_position=False):
        """Ls_experiment is used to set kappa, Ls_sim is used to adjust the simulation
        s value to match kappa."""

        self.delta_prob_dict = delta_prob_dict

        # Experimental values
        self.Ls_experiment = Ls_experiment
        self.L_to_record = L_to_record
        if self.L_to_record is None:
            self.L_to_record  = L_experiment_values
        self.L_div_Ls_experiment = self.L_to_record / Ls_experiment

        self.record_wall_position = record_wall_position

        # kappa is matched between experiment & theory
        self.kappa = np.sqrt(Ro_experiment / Ls_experiment)
        print 'kappa:', self.kappa

        # Simulation values
        self.Ls_sim = Ls_sim
        self.s_sim = np.sqrt(a/(2*Ls_sim))
        self.No_sim = np.pi*(self.kappa/self.s_sim)**2
        self.No_sim = int(round(self.No_sim))

        print 'No_sim is:', self.No_sim

        self.num_colors = num_colors

        # Calculate what the the characteristic angular correlation length is, theta_c
        self.Ro_sim = (self.No_sim*a)/(2*np.pi)
        self.theta_c = np.sqrt((8*Dw_sim)/self.Ro_sim)

        print 'Initializing simulation...'
        self.unequal_ic = unequal_ic
        if self.unequal_ic is not None:
            strains = np.arange(self.num_colors)
            initial_lattice = np.random.choice(strains, size=self.No_sim, replace=True, p=self.unequal_ic)
            self.sim = self.get_sim(lattice_ic=initial_lattice, record_wall_position = self.record_wall_position)
        else:
            self.sim = self.get_sim(record_wall_position = self.record_wall_position)
        print 'Done!'

    def get_sim(self, lattice_ic = None, **kwargs):
        # Convert into simulation units
        num_types = self.num_colors
        seed = np.random.randint(0, 2 ** 32)

        # Match L/Ls between experiment and simulation
        L_values = self.L_div_Ls_experiment * self.Ls_sim

        record_time_array = L_values / v
        # record_time_array = record_time_array[1:]

        # 1 will have the selective disadvantage, like our experiments
        if self.delta_prob_dict is None: # Populate
            ##########################################################
            ##### These parameters should be checked & fine tuned! ###
            ##########################################################

            self.delta_prob_dict = {}

            # Selective disadvantage for one of them...strain 1, like in the experiments
            for i in range(num_types):
                for j in range(num_types):
                    if i != j:
                        if i == 1:
                            self.delta_prob_dict[i, j] = -self.s_sim
                        elif j == 1:
                            self.delta_prob_dict[i, j] = self.s_sim
                        else:
                            self.delta_prob_dict[i, j] = 0

        if lattice_ic is None:
            sim = acwi.Selection_Inflation_Lattice_Simulation(
                self.delta_prob_dict,
                lattice_size=self.No_sim,
                num_types=num_types,
                seed=seed,
                record_time_array=record_time_array,
                velocity=v,
                use_specified_or_bio_IC=True,
                record_lattice=True,
                lattice_spacing_output=2 * np.pi / (360 * 8.),
                **kwargs)
        else:
            sim = acwi.Selection_Inflation_Lattice_Simulation(
                self.delta_prob_dict,
                lattice_size=self.No_sim,
                num_types=num_types,
                seed=seed,
                record_time_array=record_time_array,
                velocity=v,
                use_specified_or_bio_IC=True,
                record_lattice=True,
                lattice=lattice_ic,
                lattice_spacing_output=2 * np.pi / (360 * 8.),
                **kwargs)

        return sim

class q4_Exp_Corr_Matcher(object):
    def __init__(self, Ls_experiment, Ls_sim, num_colors=3, delta_prob_dict=None):
        """Ls_experiment is used to set kappa, Ls_sim is used to adjust the simulation
        s value to match kappa."""

        self.delta_prob_dict = delta_prob_dict

        # Experimental values
        self.Ls_experiment = Ls_experiment
        self.L_div_Ls_experiment = L_experiment_values / Ls_experiment

        # kappa is matched between experiment & theory
        self.kappa = np.sqrt(Ro_experiment / Ls_experiment)
        print 'kappa:', self.kappa

        # Simulation values
        self.Ls_sim = Ls_sim
        self.s_sim = np.sqrt(a/(2*Ls_sim))
        self.No_sim = np.pi*(self.kappa/self.s_sim)**2
        self.No_sim = int(round(self.No_sim))

        print 'No_sim is:', self.No_sim

        self.num_colors = num_colors

        # Calculate what the the characteristic angular correlation length is, theta_c
        self.Ro_sim = (self.No_sim*a)/(2*np.pi)
        self.theta_c = np.sqrt((8*Dw_sim)/self.Ro_sim)

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

        if self.delta_prob_dict is None:

            self.delta_prob_dict = {}

            ##########################################################
            ##### These parameters should be checked & fine tuned! ###
            ##########################################################

            # Selective disadvantage for one of them...strain 1, like in the experiments
            for i in range(num_types):
                for j in range(num_types):
                    if i != j:
                        if i == 1:
                            self.delta_prob_dict[i, j] = -self.s_sim
                        elif j == 1:
                            self.delta_prob_dict[i, j] = self.s_sim
                        else:
                            self.delta_prob_dict[i, j] = 0

        sim = acwi.Selection_Inflation_Lattice_Simulation(
            self.delta_prob_dict,
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

from scipy.special import erfc, erf

def get_Fij_prediction(i, j, L, num_theta_bins=1000, q=3., rescale_by_theta_c=False, Fio = None, Fjo = None):
    """Get the neutral prediction for Fij"""
    if Fio is None:
        Fio = 1./q
    if Fjo is None:
        Fjo = 1./q

    theta_bins = np.linspace(-2* np.pi, 2 * np.pi, num_theta_bins)
    if not rescale_by_theta_c:
        arg = (Ro_experiment / (8 * Dw_experiment))
    else:
        arg = 1.0
    arg_root = np.sqrt(arg * (1 + Ro_experiment/ L))
    if i == j:  # Fii
        result = Fio * (1 - (1 - Fio) * erf(np.abs(theta_bins) * arg_root))
    else: # Fij
        result = Fio * Fjo * erf(np.abs(theta_bins) * arg_root)
    return theta_bins, result


def get_H_prediction(L, num_theta_bins=1000, q=3., rescale_by_theta_c=False):
    Ho = 1. -  1/q

    theta_bins = np.linspace(-2* np.pi, 2 * np.pi, num_theta_bins)
    if not rescale_by_theta_c:
        arg = (Ro_experiment / (8 * Dw_experiment))
    else:
        arg = 1.0
    arg_root = np.sqrt(arg * (1 + Ro_experiment/ L))

    result = Ho * erf(np.abs(theta_bins) * arg_root)
    return theta_bins, result

def get_Fii_sum_prediction(L, num_theta_bins=1000, q=3., rescale_by_theta_c=False):
    """Get the neutral prediction for Fij"""
    Fio = 1. / q
    Fjo = 1. / q

    theta_bins = np.linspace(-2* np.pi, 2 * np.pi, num_theta_bins)
    if not rescale_by_theta_c:
        arg = (Ro_experiment / (8 * Dw_experiment))
    else:
        arg = 1.0
    arg_root = np.sqrt(arg * (1 + Ro_experiment/ L))
    result = q*Fio * (1 - (1 - Fio) * erf(np.abs(theta_bins) * arg_root))

    return theta_bins, result