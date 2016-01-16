


import pandas as pd
import numpy as np
from scipy import stats
import itertools
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt



class sequenceFactor:
    """This class holds a sequence and its starting and ending slices."""
    def __init__(self, seq, inherited_support=0, ind_tuple=(np.NaN,np.NaN) ):
        """Provide the sequence itself as well as an optional slice of a dataframe."""
        self.sequence = seq
        self.ind_tuple = ind_tuple
        self.norm_seq = (self.sequence-self.sequence.min()) / (self.sequence.max()-self.sequence.min())
        self.support = inherited_support #at this point is an array containing the supprot x-ticks
        self.vcount = self.sequence.value_counts()
        self.norm_support = []
        try:
            self.support_vector = np.zeros(self.support.shape) 
        except:
            ValueError('support not provided')
        
        for index in range(len(self.support)):
            #if that value exists in the value_count, fill it out in the support vector
            if( self.support[index] in self.vcount.keys()):
                #the value is present. Fill it out.
                self.support_vector[index] = self.vcount[self.support[index]]
            else:
                continue



    def __repr__():
        return self.sequence
    
    def provide_full(self):
        return dict([('values',self.sequence),('limits', self.ind_tuple)])

    def norm_vector(self):
        return self.norm_seq

    def unitize(self):
        self.norm_support = np.divide(self.support_vector,np.sum(self.support_vector))





class PrimeSequence:
    """This class is created by supplying a sequence. It contains methods to factor the sequence."""
    def __init__(self, sequence, param_dict = False):
        self.mother = sequence #the main sequence that will be deconstructed
        self.param_dict = param_dict #a dictionary of parameters for overriding defaults
        self.rand_obj = np.random.RandomState()
        self.factors = []
        self.trainset = []
        self.support = np.array(sequence.value_counts().sort_index(ascending=True).keys()) #this will be the grid for the histogram

    def __repr__(self):
        return """Empty."""
        pass


    def _define_limits(self, limit_range = 3000 ,train=True):
        """Used to generate the limits for creating contiguous subsequence selection."""
        if(train==False):
            #does this if a test set needs to be generated instead
            upper_lim = len(self.mother) - 10001 #the maximum size for a lower bound of the array
            pivot= self.rand_obj.randint(0, high=upper_lim) #this will be the value that will be used as the start of the index
            bookend = self.rand_obj.randint(1000, high=10000) #the size range for a member of the test set should be [1000,10000]
            return {'high':pivot+bookend, 'low':pivot}
        upper_lim = len(self.mother)-limit_range-1 #this will be the maximum upper limit
        #limit range prevents unauthorized accesses or out of bound ones
        pivot = self.rand_obj.randint(0,high=upper_lim) #this will be the starting point for the index
        bookend = pivot+self.rand_obj.randint(600,high = limit_range) #this will be the ending point
        return dict([('low', pivot) ,   ('high', pivot+500)    ])

        


    def _block_bootstrap(self,sample_length=5000,low_range=300, high_range = 3000):
        """Generates a list of samples through block bootstrapping with replacement.



        The defaults were chosen to make sense for heart rate data. """

        for x in range(sample_length):
            #this produces sample_length samples. The indices it needs are computed by define_limits
            limits = self._define_limits()
            seq = self.mother.iloc[limits['low']:limits['high']] #needs to be encapsulated in a sequencefactor object
            if(len(seq.dropna()) == 0):
                print 'Nan\n'
                continue
            seq = sequenceFactor(seq,inherited_support=self.support,ind_tuple = limits)
            self.factors.append(seq) #adds it to the list of sequences already there.

    def _factor_sequence(self):
        """Removes all distributions similar to a pivot distribution at a time."""
        pivot = self.factors.pop(0)
        self.factors = [x for x in self.factors if stats.ks_2samp(x.norm_vector(),pivot.norm_vector())[1] <0.8 ]
        #the above line only keeps samples for which we can reject the null hypothesis
        self.factors.append(pivot)

    def _factor_watchdog(self):
        fail_count = 0
        while(fail_count<30):
            prev_len = len(self.factors)
            self._factor_sequence()
            if(len(self.factors)==prev_len):
                fail_count = fail_count + 1

    def _generate_test(self):
        if(type(self.param_dict)==type(False)):
            #if param dict hasnt been filled out and is a bool, resort to defaults. 
            return 

        for x in range(self.param_dict['test_size']):
            limits = self._define_limits(train=False) #set it to use define limits with train=False so i will get test set size elements
            seq = self.mother.iloc[limits['low']:limits['high']]
            seq = sequenceFactor(seq,inherited_support = self.support ,ind_tuple=limits)
            self.trainset.append(seq) 
    
    
    def score(self):
        match_count = 0
        if(type(self.param_dict)==type(False)):
            #if param dict hasn't been filled out and is a bool, resort to defaults
            for x in range(len(self.trainset)):
                for factors in self.factors:
                    if(stats.ks_2samp(self.trainset[x].provide_full()['values'], factors.provide_full()['values'])[1] > 0.01):
                        #we accept the null hypothesis with 80% confidence
                        match_count = match_count + 1 #if a match is found, count it 
        return match_count 







def main():
    csvname = "TEST_DATA_FILE_HERE" 
    df = pd.read_csv(csvname)
    param_dict = {'test_size':10000, 'test_min':1000, 'test_max':10000 , 'train_size':300, 'train_length':500}
    df = df['hr']
    df.dropna(inplace=True)
    heart_df = lower_dim(df)
    test_prime = PrimeSequence(heart_df,param_dict = param_dict)
    test_prime._block_bootstrap()
    test_prime._generate_test()
    for x in test_prime.factors:
        x.unitize()
    for x in test_prime.trainset:
        x.unitize()


    return test_prime
    

def workout_factor(csvdata):
    csvname = csvdata 
    df = pd.read_csv(csvname)
    param_dict = {'test_size':10000, 'test_min':1000, 'test_max':10000 , 'train_size':300, 'train_length':100}
    df = df['hr']
    df.dropna(inplace=True)
    df = df-df.mean()
    test_prime = PrimeSequence(df,param_dict = param_dict)
    print test_prime._define_limits()
    test_prime._block_bootstrap()
    test_prime._factor_watchdog()
    return test_prime



def factor_seq(mlist, min_confidence,probabilistic=True):
    """Factors an arbitrary list of lists."""
    pivot = mlist.pop(0)
    if probabilistic==True :
        mlist = [x for x in mlist if stats.ks_2samp(x,pivot)[1]<min_confidence]
    else:
        mlist = [x for x in mlist if np.linalg.norm(pivot-x)>min_confidence]
    mlist.append(pivot)
    return mlist


def divide_sequence(factors,product, distance):
    """Uses found factors to divvy up a sequence."""
    match_dict = {str(x):[] for x in range(len(factors))}
    for x in range(len(factors)):
        pivot = factors[x]
        for y in product:
            if(np.linalg.norm(pivot-y.norm_support)<distance):
                match_dict[str(x)].append(y)
    return match_dict


def factor_watchdog(mlist,min_confidence, probabilistic=True):
    """Min confidence is how sure you want to be that two distributions are equal."""
    fail_count = 0
    while(fail_count<30):
        prev_len = len(mlist)
        mlist = factor_seq(mlist,min_confidence, probabilistic)
        if(len(mlist)==prev_len):
            fail_count = fail_count + 1
    return mlist


def factor_by_distance(mlist, minimum_distance,get_metric=lambda x: x):
    pivot = mlist.pop(0)
    mlist = [x for x in mlist if np.linalg.norm(get_metric(x)-get_metric(pivot)) > minimum_distance] #only includes vectors in a list that are far enough away
    mlist.append(pivot) #sticks pivot at the end
    return mlist



def lower_dim(sequence):
    """Lowers dimensionality of a pandas series."""


    sequence = sequence - (sequence % 10) #lops off the ones place
    return sequence 


if __name__ == '__main__':
    main()



