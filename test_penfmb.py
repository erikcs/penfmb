import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from pandas_datareader.data import DataReader

import penfmb as pn

# French Data library
kwds = {'data_source': 'famafrench', 'start': '1972-1', 'end': '2013-12'}
MONTHLY = 0
ff = DataReader("F-F_Research_Data_Factors", **kwds)[MONTHLY]
twentyfive = DataReader("25_Portfolios_5x5", **kwds)[MONTHLY]
mom = DataReader("F-F_Momentum_Factor", **kwds)[MONTHLY]

twentyfive = twentyfive.subtract(ff['RF'], axis=0)
carhart = pd.concat([ff.drop('RF', axis=1), mom], axis=1)

class TestMain(unittest.TestCase):

    def test_fmb(self):
        _, b , _ = pn._fmb(twentyfive, ff['Mkt-RF'])
        assert_almost_equal(b.params,
                 np.array([ 1.4741268, -0.6969725]), decimal=4)

        _, b, _ = pn._fmb(twentyfive, ff.drop('RF', axis=1))
        assert_almost_equal(b.params,
                np.array([ 1.3495192, -0.7947864,  0.1414998,  0.4257349]),
                decimal=4)

    def test_PenFMB(self):
        penfmb = pn.PenFMB(nboot=1).fit(twentyfive, carhart)
        assert_almost_equal(penfmb.coefs_['coef'],
        np.array([ 1.29852433, -0.74661834,  0.14244865,  0.42918938,  0.]),
        decimal=4)
