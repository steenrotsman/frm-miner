from os.path import join

from utils import parse, plot

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import numpy as np
import tensorflow as tf

from motifminer import Miner
from motifminer.preprocessing import standardize

def main():
    args = parse()
    data = get_data()

    mm = Miner(data, args.min_sup, args.w, args.a, args.l, args.o, args.k, args.m)
    motifs = mm.mine_motifs()

    if args.plot:
        plot(standardize(data), motifs, args.w, args.a)


def get_data():
    gdf = gpd.read_file(join('data', 'telangana', 'mandals_population_2020.geojson'))
    gdf['mean_pop'] = gdf['zonalstat'].apply(lambda d: d['mean'])
    df = pd.read_csv(join('data', 'telangana', 'telangana_fires.csv'), parse_dates=['acq_date'])
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    df = df.set_crs(gdf.crs)

    data = df.sjoin(gdf, how="inner", predicate='intersects')
    distnames = data.Dist_Name.unique()

    rag = []
    for name in distnames:
        rag.append(data.loc[data.Dist_Name == name].brightness.values)
    rag = tf.ragged.constant(rag, dtype=tf.float32)
    rag = standardize(rag)
    return rag


if __name__ == '__main__':
    main()