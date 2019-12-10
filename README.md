# bike-stations

This project deals with Bike stations clustering.

This script performs K-means clustering for bike stations based on location.

It can be run in 2 modes:

1. performs Kmeans clustering choosing the clusters number, default=6.

2. help to choose the optimum k plotting the elbow and silouhette graphs for k between k_min and k_max. 

### Launch the program ###

The script is launched through the command lines:

1. Kmeans clustering mode
```
python bikestations.py -d ../data/Brisbane_CityBike.json
```
2. Plot elbow and silouhette graph mode:
```
python bikestations.py -op -d ../data/Brisbane_CityBike.json
```
The command line arguments are:
```
  -h, --help            show this help message and exit
  -op                   Set a switch to true
  -k N_CLUSTERS, --n_clusters N_CLUSTERS
                        number of clusters.
  -kn KMIN, --kmin KMIN
                        number of clusters min.
  -kx KMAX, --kmax KMAX
                        number of clusters max.
  -m CMAP, --cmap CMAP  path for city map:
  -p PATH, --path PATH  path where to save result.
  -d FILE_PATH, --file_path FILE_PATH
                        path of the json data file.
```