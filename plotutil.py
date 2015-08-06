"""A group of utility functions for validating data and graphing them, as well as automating pulling
	from created dictionaries."""
# This library contains utility functions for visualizing the results of clustering algorithms
# from scikit learn. It relies on matplotlib, seaborn, and pylab. This exists because the natural
# input to most machine learning algorithms is an array of vectors. The resulting predictions along
# with their tags can be represented succintly as a tuple containing a vector and the label given
# to it. However, most graphing functions require a list of the coordinates in each dimensions;
# this necessiates splitting the list of tuples vertically for passing to the graphing function.

import pandas as pd
import numpy as np
import matplotlib as plt
import pylab as pyl
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import bisect
import datetime
import warnings
from sklearn import cluster
from hmmlearn import hmm




def tuple_check(ntuple):
    """Takes in a tuple, returns true only if every member of the tuple is a number."""
    filtered_tuple = np.isnan(ntuple)
    if all(item==False for item in filtered_tuple):
        return True
    else: 
        return False


def pull_from_tag(tag_to_pull,whichpair,list_to_pull_from):
	"""Returns all items with tag_to_pull from iterable list_to_pull_from using whichpair to 
		determine which element to take out"""
	if whichpair == 1:
		return [x for x,y in list_to_pull_from if y == tag_to_pull] #decides whether to return first element or second
	else:
		return [y for x,y in list_to_pull_from if x == tag_to_pull]


def tuple_list_creator(list_to_generate_from):
    """Takes in a list of lists of tuples, and then slices them vertically to return a lists of lists of x-
    	dimensions the same as that of the tuple represented as a vector."""
    list_to_return = []
    for x in list_to_generate_from:
        list_to_return.append(zip(*x)) #this is the part doing the slicing
    return list_to_return

colormap = ['#66FF66','#008000','#000066','#8080FF','#660000','#FF4D4D','#990099','#FF33FF','#808000','#FFFF4D','#B26B00','#FFAD33','#476B6B','#A3C2C2','#6B2400','#D6AD99','#FFFFFF','#000000']  
#colormap is a list that provides HTML color codes for makePlot to use. It can represent up to
#eighteen different data sets.

def makePlot_3d(coordinate_list):
    """Creates a 3d plot of objects with multiple tags from coordinate list.
    	coordinate_list is a list of tuples of lists, where each tuple element is a set of
    	coordinates for that particular list. Ex: [([x,x,x,x],[y,y,y,y],[z,z,z,z]),...]"""
    plotObjectBox = pyl.figure() #creates a figure
    plotObjectBox_ax = plotObjectBox.add_subplot(111, projection='3d') #adds a subplot
    togetherlist = zip(coordinate_list,colormap[:len(coordinate_list)-1]) #creates a tuple list
    for x,y in togetherlist: #associates each set of coordinates with an html color tag
        plotObjectBox_ax.scatter(x[0], x[1],x[2],c=y)

def index(a, x):
    """Locate the leftmost value exactly equal to x, arg a is list, x=key

    Returns item if found, returns False if item not found,"""
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return False


def timeTrack_recordnew(datetimeseries):
    """Takes in a datetimeseries, returns list of skips [(skiplength, index)...]"""
    breaklist = []
    mylen = range(0,len(datetimeseries)-1)
    for x in mylen:
        if datetimeseries[x+1] != datetimeseries[x]+timedelta(seconds=1):
            nextstep = x+1
            breaklist.append([datetimeseries[nextstep]-datetimeseries[x],x])
        else:
            continue
    return breaklist


def access_DFrow(indextopull,dataFrameToPullFrom):
    """access_DFrow(indextopull,dataFrameToPullFrom)-> return row"""
    listToReturn =[] #list to return
    for x in dataFrameToPullFrom.keys():
        TEMP_chainvar = dataFrameToPullFrom[x]
        listToReturn.append(TEMP_chainvar[indextopull])
    return listToReturn

def PullDate(date,framename):
    timeseries = pd.to_datetime(framename['time'])
    startdate = timeseries[0]
    return index(timeseries, startdate.replace(day=date,hour=0,second=0,minute=0))
    


def sliceDF(tupleIndex, frameInUse):
    """Creates a dataframe bookended by a tuple"""
    myframe = pd.DataFrame()
    for x in frameInUse.keys():
        myframe[x]=frameInUse[x][tupleIndex[0]:tupleIndex[1]:1]
    return myframe
    
def SliceMaker(framename,colname):
    zippedDateSlices = [] #will hold the tuples of start and end indices
    fullDateIndexList = [] #will hold the list of day indexes
    for x in range(1,32):
        fullDateIndexList.append(PullDate(x,framename))
    for x in range(len(fullDateIndexList)):
        if x==len(fullDateIndexList)-1:
            break
        elif fullDateIndexList[x]==False :
            continue
        else:
            mytuple = (fullDateIndexList[x],fullDateIndexList[x+1])
            zippedDateSlices.append(mytuple)
    listofDayFrames = []
    for x in zippedDateSlices:
        listofDayFrames.append(sliceDF(x,framename))
    return listofDayFrames


def makeKDE(series,clusnum):
    """"Series is a series and clusnum is the number of clusters.


    Returns a (dataframe,kmeans object)"""
    stouse = np.array(series.dropna())
    artouse = np.resize(stouse,(len(stouse),1))
    kmetouse = cluster.MiniBatchKMeans(n_clusters = clusnum)
    kmetouse.fit(artouse)
    predtouse = kmetouse.predict(artouse)
    frametoret = pd.DataFrame()
    ziplist = zip(predtouse,stouse)
    for x in range(clusnum):
        frametoret[str(x)] = pd.Series([z for y,z in ziplist if y ==x])
    return frametoret,kmetouse



def HMMmaker(kclus,DFlist,statenum,s_name):
	"""Takes in a kmeans object and a list of dataframes containing days."""
	detlist = []
	warnings.filterwarnings("ignore", category=DeprecationWarning) 
	for x in DFlist:
		benchHMM=hmm.GaussianHMM(n_components=statenum)
		x['pred'+s_name] = kclus.predict(np.resize(x[s_name],(len(x[s_name]),1)))
		benchHMM.fit([np.reshape(x['pred'+s_name],(len(x),1))])
		print np.linalg.det(benchHMM.transmat_)
		detlist.append(np.linalg.det(benchHMM.transmat_))
	return detlist



def proper_convert(nanDaylist):
	trashlist = []
	for x in nanDaylist:
		trashlist.append(x.dropna(subset=['hr','accel_magnitude','skin_temp']))
	validatedList = []
	for x in trashlist:
		if len(x)==0 :
			print 'Dropped'
		else:
			validatedList.append(x)
	print 'Total dropped:'+str(len(trashlist)-len(validatedList))
	return validatedList