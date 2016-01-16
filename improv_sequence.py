import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors.kde import KernelDensity
import itertools
import sys

def give_random_sequence(sample_seq, sample_length=2, low_range = 300, high_range= 3000):
        """Generates a list of random sequences for sampling. """
	ret_sequence = []
	random_obj = np.random.RandomState()
	for x in range(sample_length):
	#generates a number of samples
		sample = random_obj.choice(sample_seq, size = random_obj.randint(low_range, high = high_range)) 
		ret_sequence.append(sample)
	return ret_sequence
    

try:
    csvname = sys.argv[1]
except:
    csvname = "TEST DATA" 


df = pd.read_csv(csvname)


heartframes = df['hr'].dropna() #i will only be using heart rates for a proof of concept


#I need to get a long list of sequences

seqlist = give_random_sequence(heartframes, sample_length = 200)

#i am now going to select another random list of sequences, but this one will be longer and have larger subsequences


complist = give_random_sequence(heartframes, sample_length = 3000, low_range = 1000, high_range = 10000)


#I then need to compare each sequence in seqlist to the sequences in complist. I need to develop some sort of classification method so I can spot sequences which are the same.

#i will write a function that sort of factors out sequences which are similar


def split_sequence(sequencelist):
    
    pivot = sequencelist.pop(0) #pops off the first element
    filtered_list = [x for x in sequencelist if stats.ks_2samp(pivot,x)[1] < 0.90]
    filtered_list.append(pivot)
    return filtered_list


fail_count = 0  #i just use this loop to terminate when I have more than 30 failures
while(fail_count < 30):
    prev_len = len(seqlist) #holds the previous length
    seqlist = split_sequence(seqlist) #splits
    if (len(seqlist) == prev_len):
        fail_count = fail_count + 1


try:
    plot_flag = sys.argv[2]
except:
    plot_flag = False

if(plot_flag == '-p'):
    plt.figure()
    for x in seqlist:
        pd.Series(x).plot(kind='kde')
    plt.title('There are: %s sequences' %(str(len(seqlist))))
    plt.show(block=True)

print '%s sequences found\n' %(str(len(seqlist)))


#I am now going to attempt to see how much of the variance in the compare sequence I can explain 

explained_sequences = []
complen = len(complist) #this is the length of the original list

for test_seq in seqlist:
    for x in range(len(complist)):
        #iterating over the index to ensure I can pop
        if(x>len(complist)-1):
            break;
        if((stats.ks_2samp(test_seq,complist[x])[1])>0.8):
            tmp_seq = complist.pop(x) #if the sequence matches an already found sequence
            explained_sequences.append(tmp_seq)
        


print len(complist)/float(complen) #this number is the percentage of unmatched sequences left
print len(explained_sequences)/float(complen)
