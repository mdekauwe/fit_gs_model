#!/usr/bin/env python

"""
Fit Belinda Medlyn's stomatal conductance model g0 and g1 paramaters, where
g0 (mol m−2 s−1) and g1 represent the residual stomatal conductance when A
reaches zero, and the slope of the sensitivity of gs to A.

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (30.06.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from lmfit import minimize, Parameters

class FitMedlyn(object):
    """
    Fit Belinda's stomatal conductance model using the lmfit package

    Reference:
    ---------
    * Medlyn et al. (2011) Global Change Biology (2011) 17, 2134-2144.

    """

    def __init__(self, param_fits=["g0", "g1"], obs="OBS", vpd="VPD",
                 assim="Photo", co2="CO2S"):

        self.param_fits = param_fits
        self.obs = obs
        self.vpd = vpd
        self.assim = assim
        self.co2 = co2

    def setup_model_params(self, fit_g0=False):
        # Which params to fit? By default g0 will be zero """

        params = Parameters()
        if fit_g0:
            params.add('g0', value=0.0, vary=True)
        else:
            params.add('g0', value=0.0, vary=False)
        params.add('g1', value=2.0, min=0.0)

        return params

    def minimise_params(self, params, df):

        obs = df[self.obs]

        try:
            result = minimize(self.residual, params, args=(df, obs))
            success = True
        except:
            result = -999
            success = False

        return (result, success)

    def gs_model(self, vpd, gpp, co2, g0, g1):
        # Eqn q in corrigendum Medlyn et al. (2012) Global Change Biology 18.

        return g0 + 1.6 * (1.0 + (g1 / np.sqrt(vpd))) * (gpp / co2)

    #def residual(self, params, df, obs):
    def residual(self, params, df, obs):
        g0 = params['g0'].value
        g1 = params['g1'].value

        model = self.gs_model(df[self.vpd], df[self.assim], df[self.co2],
                              g0, g1)

        return (obs - model)

    def print_fit_to_screen(self, d):
        for key, val in d.items():
            print( '%s = %.4f ' % (key, val) )

    def get_fit_stats(self, result, df):

        d = {}
        d['g1'] = result.params['g1'].value
        d['g1_se'] = result.params['g1'].stderr
        d['g0'] = result.params['g0'].value
        d['g0_se'] = result.params['g0'].stderr
        pred = self.gs_model(df[self.vpd], df[self.assim], df[self.co2],
                             d['g0'], d['g1'])
        obs = df[self.obs]
        d['rsq'] = (pearsonr(obs, pred)[0])**2
        d['num_pts'] = len(obs)
        d['rmse_val'] = self.rmse(obs, pred)

        return (d)

    def rmse(self, x, y):
        return np.sqrt(np.mean((x - y)**2))

if __name__ == "__main__":

    fname = "data/PI_ISierrae_Ags.csv"
    df = pd.read_csv(fname)

    # Set the col names from your file.
    F = FitMedlyn(obs="OBS", vpd="VPD", assim="Photo", co2="CO2S")
    params = F.setup_model_params(fit_g0=True)
    (result, success) = F.minimise_params(params, df)
    if success:
        (d) = F.get_fit_stats(result, df)
        F.print_fit_to_screen(d)
