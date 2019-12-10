# import needed libraries
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
import logging
logging.basicConfig(filename='../output/log_file.log',level=logging.DEBUG)


def Kmeansclustering(X, n_clusters):
	""" K-means Clustering of bike stations based on location.
	Parameters
	    df: pandas dataframe
		    Bike stations data containing location
	    n_clusters: int
		    Number of clusters.
	Returns
		clusters labels
	"""		
	# Kmeans clustering   
	kmeans = KMeans(n_clusters=n_clusters,init='k-means++', random_state=0)

	# fit and predict kmeans based on the location of bike stations
	clusters = kmeans.fit_predict(X)
	
	return clusters

def plot_elbowandsilouhette(X,kmin,kmax):
	""" plot elbow and silouhette to help to find optimum n_clusters.
	Parameters
		df: pandas dataframe
		    Bike stations data containing location
	    kmin: int
		    minimum clusters number
	    kmax: int
		    maximum clusters number
	
	"""


	# create the within-cluster sums of squares (WCSS) to choose the number of clusters
	wcss = []
	# create the silouhette average score to choose the number of clusters
	silhouette_avg = []

	for i in range(kmin,kmax):
		kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
		kmeans.fit(X)
		wcss.append(kmeans.inertia_)
		if i==1:
			silhouette_avg.append(0)
		else:
			silhouette_avg.append(silhouette_score(X, kmeans.labels_))

	fig, (ax1, ax2) = plt.subplots(1, 2)
	
	# plot Elbow graph
	ax1.plot(range(kmin,kmax), wcss)
	ax1.set_title('Elbow graph')
	ax1.set(xlabel='Cluster number', ylabel='WCSS')

	
    # plot silouhette graph
	ax2.plot(range(kmin,kmax), silhouette_avg)
	ax2.set_title('silouhette graph')
	ax1.set(xlabel='Cluster number', ylabel='silhouette_avg')
	
	plt.show()



def plot_result(df, clusters, cmap, result,colors):
	""" Plot result of bike stations clustering.
	Parameters
	    df: pandas dataframe
		    Bike stations data containing location
	clusters: list of int
		labels of the bike stations clusters
        cmap : str
        map of the city (manually imported from openstreetmap.org) -> can be automatised later
        if = 'skip' plot without map in background
	result: str
		path to save the plot
        if = 'skip' no save
	colors: str list
		used colors for clusters
	"""


    # get the bounding box of the map
	BBox = ((df.longitude.min(), df.longitude.max(), df.latitude.min(), df.latitude.max()))
	
	if cmap != 'skip':
        # read the map.png
		mappng = plt.imread(cmap)
	
        #plot bike stations with corresponding clusters
		fig, ax = plt.subplots(figsize = (10,10))
		ax.scatter(df.longitude, df.latitude, zorder=1, alpha= 1, c=[colors[l] for l in clusters], s=10)
		ax.set_title('Bike Stations Clusters')
		ax.set_xlim(BBox[0],BBox[1])
		ax.set_ylim(BBox[2],BBox[3])
		ax.imshow(mappng, zorder=0, extent = BBox, aspect= 'equal')
		plt.show()
	else:
		fig = plt.figure(figsize = (10,10))
		plt.scatter(df.longitude, df.latitude, c=[colors[l] for l in clusters])
		plt.show()

	#save figure
	if result != 'skip':
		fig.savefig(result + 'bike_stations_clusters', dpi=fig.dpi)
	


if __name__ == "__main__":
	#description
	parser = argparse.ArgumentParser(description='A script for clustering bike stations based on location. \
	 user can train K-means Clustering specifing the number of clusters \
	 script can plot elbow and silouhette graph to help user to choose the number of clusters.')
	#arguments
	parser.add_argument("-op", action='store_true', default=False, dest='boolean_switch', help='Set a switch to true')
	parser.add_argument("-k", "--n_clusters", type=int, default=6, help='number of clusters.')
	parser.add_argument("-kn", "--kmin", type=int, default=1, help='number of clusters min.')
	parser.add_argument("-kx", "--kmax", type=int, default=11, help='number of clusters max.')
	parser.add_argument("-m", "--cmap", type=str, default='skip', help="path for city map: ")
	parser.add_argument('-p', '--path', type=str, default='skip', help="path where to save result.")
	parser.add_argument('-d', '--file_path', type=str, help='path of the json data file.', required=True)

	#retreiving arguments
	args = vars(parser.parse_args())
	n_clusters = args["n_clusters"]
	kmin = args["kmin"]
	kmax = args["kmax"]
	cmap = args["cmap"]
	result = args['path']
	input_file = args['file_path']
	optimize = args['boolean_switch']


	#read data from jsob file
	df = pd.read_json(input_file)
	logging.info('json file loaded')

	# get location from dataframe
	X = df.loc[:, ['latitude','longitude']]

	#costumize colors
	colors = sns.color_palette('Dark2_r', n_clusters)

	if optimize:
		#plot elbow and silouhette to help choosing the number of clusters
		plot_elbowandsilouhette(X, kmin, kmax)
		logging.info('plot finished')
	else:
		#perform clustering 
		clusters = Kmeansclustering(X, n_clusters)
		logging.info('clustering finished')
		#plot the clusters
		plot_result(df, clusters, cmap, result,colors)

		#export clustered data to json file
		if result != "skip":
			df['clusters'] = clusters
			df.to_json(result + "clustered_bike_stations.json")
			logging.info('Results saved to json')