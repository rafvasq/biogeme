import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.draws as draws
import biogeme.models as models

pandas = pd.read_table("tripdata.dat")
database = db.Database("tripdata",pandas)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
#print(database.data.describe())

from headers import *

# Removing some observations can be done directly using pandas.
#remove = (((database.data.PURPOSE != 1) & (database.data.PURPOSE != 3)) | (database.data.CHOICE == 0))
#database.data.drop(database.data[remove].index,inplace=True)

# Here we use the "biogeme" way for backward compatibility
#exclude = (( PURPOSE != 1 ) * (  PURPOSE   !=  3  ) +  ( CHOICE == 0 )) > 0
#database.remove(exclude)


ASC_AUTO_DRIVE = Beta('ASC_AUTO_DRIVE',0,None,None,0)
ASC_AUTO_PASS = Beta('ASC_AUTO_PASS',0,None,None,0)
ASC_METRO = Beta('ASC_METRO',0,None,None,0)
ASC_TRAIN = Beta('ASC_TRAIN',0,None,None,0)
ASC_WALK = Beta('ASC_WALK',0,None,None,0)

B_IVTT = Beta('B_IVTT',0,None,None,0)
B_IVTT_S = Beta('B_IVTT_S',0,None,None,0)

MU_1 = Beta('MU_1',0.5,1,10,0)
MU_2 = Beta('MU_2',0.5,1,10,0)

#B_TIME = Beta('B_TIME',0,None,None,0)
#B_TIME_S = Beta('B_TIME_S',0,None,None,0)
#B_COST = Beta('B_COST',0,None,None,0)
#B_COST_S = Beta('B_COST_S',0,None,None,0)

# Define a random parameter, log normally distributed, designed to be used
# for Monte-Carlo simulation
# = (B_IVTT + B_IVTT_S * bioDraws('B_IVTT_RND','NORMAL'))
#MU_1_RND = MU_1 * bioDraws('MU_1_RND', 'NORMAL')
#MU_2_RND = MU_2 * bioDraws('MU_2_RND', 'NORMAL')
#B_TIME_RND = -exp(B_TIME + B_TIME_S * bioDraws('B_TIME_RND','NORMAL'))
#B_COST_RND = -exp(B_COST + B_COST_S * bioDraws('B_TIME_RND','NORMAL'))

# Utility functions

#If the person has a GA (season ticket) her incremental cost is actually 0 
#rather than the cost value gathered from the
# network data. 
#SM_COST =  SM_CO   * (  GA   ==  0  ) 
#TRAIN_COST =  TRAIN_CO   * (  GA   ==  0  )

# For numerical reasons, it is good practice to scale the data to
# that the values of the parameters are around 1.0. 
# A previous estimation with the unscaled data has generated
# parameters around -0.01 for both cost and time. Therefore, time and
# cost are multipled my 0.01.

#TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED',\
#                                 TRAIN_TT / 100.0,database)
#TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED',\
#                                   TRAIN_COST / 100,database)
#SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0,database)
#SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100,database)
#CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100,database)
#CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', CAR_CO / 100,database)

#V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
#V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
#V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

V1 = ASC_AUTO_DRIVE + \
     B_IVTT * ivtt1
V2 = ASC_AUTO_PASS + \
     B_IVTT * ivtt2
V3 = ASC_METRO + \
     B_IVTT * ivtt3
V4 = ASC_TRAIN + \
     B_IVTT * ivtt4
V5 = ASC_WALK + \
     B_IVTT * ivtt5

# Associate utility functions with the numbering of alternatives
V = {1: V1,
     2: V2,
     3: V3,
     4: V4,
     5: V5}

# Associate the availability conditions with the alternatives

av = {1: avail1,
      2: avail2,
      3: avail3,
      4: avail4,
      5: avail5}

#Definition of nests:
# 1: nests parameter
# 2: list of alternatives
auto = MU_1 , [1,2]
train = MU_2 , [3,4]
walk = 1.0 , [5]
nests = auto,train,walk

# The choice model is a logit, with availability conditions
#prob = models.logit(V,av,choice)
#logprob = log(MonteCarlo(prob))

#biogeme = bio.BIOGEME(database,logprob,numberOfDraws=1000)

#biogeme.modelName = '17lognormalMixture'
#results = biogeme.estimate()
#print(results)

logprob = models.lognested(V,av,nests,choice)
biogeme  = bio.BIOGEME(database,logprob)
biogeme.modelName = "09nested"
results = biogeme.estimate()
print("Results=",results)