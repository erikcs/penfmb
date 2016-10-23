Simple Python implementation of the Penalized Fama-MacBeth estimator from *S. Bryzgalova 'Spurious Factors in Linear Asset Pricing Models' (2015)*

###### Installation
`$ pip install git+https://github.com/nuffe/penfmb.git`

###### Example
Apply the estimator to the Carhart 4 factor model (`carhart`) with the ubiquitous
25 Fama-French portfolios (`twentyfive`)

```
>>> from penfmb import PenFMB
>>> est = PenFMB(nboot=1000).fit(twentyfive, carhart)
>>> est.coefs_
            coef  shrinkage rate
const   1.298400           0.000
Mkt-RF -0.746495           0.000
SMB     0.142363           0.001
HML     0.429167           0.000
Mom     0.000000           0.999
```
