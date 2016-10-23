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
                 np.array([ 1.47420462, -0.69708656]))

        _, b, _ = pn._fmb(twentyfive, ff.drop('RF', axis=1))
        assert_almost_equal(b.params,
                np.array([ 1.34944109, -0.79471028,  0.14141737,  0.42571358]))

    def test_PenFMB(self):
        penfmb = pn.PenFMB(nboot=1).fit(twentyfive, carhart)
        assert_almost_equal(penfmb.coefs_['coef'],
        np.array([ 1.29839995, -0.7464947 ,  0.14236288,  0.42916711,  0.]))
