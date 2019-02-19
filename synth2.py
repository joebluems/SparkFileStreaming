import random
import time
import string
import datetime
import numpy as np
from random import randrange
import random

### set sampling parameters ###
N = 1000    ## number of planes
V = 17  ## length of ID number

print("id,target,feature1,feature2,feature3,feature4,feature5")

### create sample ###
for i in range(0,N):
  x = 0
  if random.random()<0.1: x=1
  id = ''.join(random.choice(string.digits + string.ascii_uppercase) for _ in range(V))
  print(
      id+','+
      str(x)+','+
      str(np.random.normal(0,1))+','+
      str(np.random.normal(0,1))+','+
      str(np.random.normal(0,1))+','+
      str(np.random.normal(0,1))+','+
      str(np.random.normal(0,1))
  )

