# import relevant packages
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from eofs.standard import Eof
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

# ** relevant global variables **
# variable titles 
names = ["Sea Surface Temperature",
            "Mean Sea Level Pressure",
            "Meridional Wind Velocity",
            "Zonal Wind Velocity"]

# bi-monthly season labels
bi_monthly_labels = [
    "Dec-Jan", "Jan-Feb", "Feb-Mar", "Mar-Apr", "Apr-May", "May-Jun",
    "Jun-Jul", "Jul-Aug", "Aug-Sep", "Sep-Oct", "Oct-Nov", "Nov-Dec"
]


"""
FUNCTION: initialise data for manipulation and further use in the Eof analysis

OUTPUT:
sst_sea, msl_sea, v10_sea, u10_sea = xarray datasets reduced to sea coordinates only, with spatial dimensions flattened
canvas = numpy array of 0s, the same dimensions as the specified region
sea_rows, sea_cols, land_rows, land_cols = set of indidices representing land and sea locations in canvas
lon, lat = arrays of longitude and latitude
timepoints = array with dates


1. load in the datasets 
2. reduce datasets to the specified region
3. flatten spatial coordinates 
4. reduce all datasets to sea coordinates only
5. locate position of sea coordinates in the entire region, used for plotting loading patterns
6. create blank canvas for plotting results onto
7. save measurement dates for plotting the timeseries
"""
def ReduceToSea():
    print("\n\n Extracting sea points... wait ~1min please... \n\n")         # status update

    # *** step 1 
    # load in the datasets 
    sst = xr.open_dataset('sst_monthly.nc')['sst']
    msl = xr.open_dataset('msl_monthly.nc')['msl']
    v10 = xr.open_dataset('v10_monthly.nc')['v10']
    u10 = xr.open_dataset('u10_monthly.nc')['u10']

    # *** step 2
    # select information at region 30S–30N, 100E–70W
    region = dict(latitude=slice(30, -30), longitude=slice(100, 290)) # 70W == 360-70=290 degrees
    sst_red = sst.sel(**region)
    msl_red = msl.sel(**region)
    v10_red = v10.sel(**region)
    u10_red = u10.sel(**region)

    # *** step 3
    # flatten spatial dimensions, i.e. (lat, lon) -> (space)
    msl_flat = msl_red.stack(space=('latitude', 'longitude'))
    v10_flat = v10_red.stack(space=('latitude', 'longitude'))
    u10_flat = u10_red.stack(space=('latitude', 'longitude'))
    sst_flat = sst_red.stack(space=('latitude', 'longitude'))

    # *** step 4
    # identify sea coordinates (non-NaN points in SST) across all time steps
    valid_points = ~sst_flat.isnull().any(dim='time')

    # reduce ALL datasets to sea-only points
    sst_sea = sst_flat[:, valid_points]
    msl_sea = msl_flat[:, valid_points]
    v10_sea = v10_flat[:, valid_points]
    u10_sea = u10_flat[:, valid_points]


    # *** step 5
    # store coordinates of the whole region (could use any _flat dataset)
    gridpoints = []
    for i in range(len(msl_flat.space.values)):
        coord = msl_flat.space.values[i]
        gridpoints.append(np.array(coord).tolist())

    # store coordinates of only the sea data
    seapoints = []
    for i in range(len(sst_sea.space.values)):
        coord = sst_sea.space.values[i]
        seapoints.append(np.array(coord).tolist())

    # find indices of sea points in the whole grid 
    # NB: not a very efficient way to do this, takes ~45s
    sea_inds = [ np.where( (np.array(gridpoints) == coord).all(axis=1))[0][0] for coord in seapoints ]
    land_inds = np.setdiff1d( np.arange(len(gridpoints)), sea_inds )

    # *** step 6
    # arrays of latitude and longitude 
    lat = msl_red['latitude'].to_numpy()
    lon = msl_red['longitude'].to_numpy()

    # blank canvas of the region, used to place results onto
    canvas = np.zeros( (len(lat), len(lon)) )

    # convert 1D indices into 2D indices
    sea_rows, sea_cols = np.unravel_index(sea_inds, canvas.shape)
    land_rows, land_cols = np.unravel_index(land_inds, canvas.shape)

    # *** step 7
    # save times as x values for the timeseries plot
    times = sst.time.to_numpy()

    return sst_sea, msl_sea, v10_sea, u10_sea, canvas, sea_rows, sea_cols, land_rows, land_cols, lon, lat, times

sst_sea, msl_sea, v10_sea, u10_sea, canvas, sea_rows, sea_cols, land_rows, land_cols, lon, lat, times = ReduceToSea()

"""
FUNCTION: group a dataset into bi-monthly seasons

INPUT: data = xarray dataset
OUTPUT: grouped = xarray dataset grouped into 12 binomnthly seasons

1. take rolling average of data between adjacent months
2. re-assign the average to a new coordinate label, "bimonthly_season"
3. group the data into 12 bimonthly seasons
"""
def groupData(data):

    # *** step 1
    # take the mean between adjacent months
    ds = data.rolling(time=2, center=True).mean().dropna('time')

    # create a bi-monthly index for each time point
    # input: time = individual month as an xarray coordinate
    def assign_bimonthly_group(time):
        
        # get month as an integer 
        month = time.values

        # return relevant bi-monthly index
        if month == 12:  # December -> "Dec-Jan"
            return "Dec-Jan"
        else:
            return bi_monthly_labels[month]

    # *** step 2
    # create new coordinate in the dataset: bimonthly_season
    ds = ds.assign_coords( bimonthly_season=("time", [ assign_bimonthly_group(t) for t in ds["time"].dt.month ]) )

    # *** step 3
    # group the data by the bimonthly season
    grouped = ds.groupby("bimonthly_season")

    return grouped



"""
FUNCTION: organise the datasets into 12 separate ones with data for each bi-monthly season

OUTPUT:
sst_normalised, msl_normalised, v10_normalised, u10_normalised = anomaly matrices for each variable
all_data = list of 12 xarray datasets with the 4 variables concatenated along the spatial dimension

1. group datasets into bi-monthly seasons
2. normalise (i.e. find anomaly matrices) for each variable
3. extract each bi-monthly group of each dataset and store in a list for each month
4. for the 4 variables of each bimonthly group, combine into a single dataset by concatenating along the spatial dimension
"""
def get_Bimonthlies():
    print("Organising into the bi-monthly seasons... \n\n")      # status update

    # *** step 1
    # group each dataset into the bimonthly seasons
    sst_grouped = groupData(sst_sea)
    msl_grouped = groupData(msl_sea)
    v10_grouped = groupData(v10_sea)
    u10_grouped = groupData(u10_sea)

    # *** step 2
    # normalising procedure: subtract the mean and divide by the standard deviation
    def normalise(data):
        mean = data.mean(dim='time', skipna=True)
        std = data.std(dim='time', skipna=True)
        return (data - mean) / std

    sst_normalised = normalise(sst_grouped)
    msl_normalised = normalise(msl_grouped)
    v10_normalised = normalise(v10_grouped)
    u10_normalised = normalise(u10_grouped)
    
    # *** step 3
    # lists to contain data for 4 variables
    DJ = []
    JF = []
    FM = []
    MA = []
    AM = []
    MJ = []
    JJ = []
    JA = []
    AS = []
    SO = []
    ON = []
    ND = []

    # select the particular bi-monthly seasons from each dataset and organising into the lists
    normalised_datasets = [sst_normalised, msl_normalised, v10_normalised, u10_normalised]
    for i in range(4):
        ds = normalised_datasets[i]
        DJ.append(ds.sel(bimonthly_season = 'Dec-Jan'))
        JF.append(ds.sel(bimonthly_season = 'Jan-Feb'))
        FM.append(ds.sel(bimonthly_season = 'Feb-Mar'))
        MA.append(ds.sel(bimonthly_season = 'Mar-Apr'))
        AM.append(ds.sel(bimonthly_season ='Apr-May'))
        MJ.append(ds.sel(bimonthly_season = 'May-Jun'))
        JJ.append(ds.sel(bimonthly_season = 'Jun-Jul'))
        JA.append(ds.sel(bimonthly_season = 'Jul-Aug'))
        AS.append(ds.sel(bimonthly_season = 'Aug-Sep'))
        SO.append(ds.sel(bimonthly_season = 'Sep-Oct'))
        ON.append(ds.sel(bimonthly_season = 'Oct-Nov'))
        ND.append(ds.sel(bimonthly_season = 'Nov-Dec'))

    # *** step 4
    # concatenate along the SPATIAL DIMENSION to use for EOF analysis
    DJ_comb = xr.concat(DJ, dim='space')
    JF_comb = xr.concat(JF, dim='space')
    FM_comb = xr.concat(FM, dim='space')
    MA_comb = xr.concat(MA, dim='space')
    AM_comb = xr.concat(AM, dim='space')
    MJ_comb = xr.concat(MJ, dim='space')
    JJ_comb = xr.concat(JJ, dim='space')
    JA_comb = xr.concat(JA, dim='space')
    AS_comb = xr.concat(AS, dim='space')
    SO_comb = xr.concat(SO, dim='space')
    ON_comb = xr.concat(ON, dim='space')
    ND_comb = xr.concat(ND, dim='space')

    all_data = [DJ_comb, JF_comb, FM_comb, MA_comb, AM_comb, MJ_comb, JJ_comb, JA_comb, AS_comb, SO_comb, ON_comb, ND_comb]
    
    return sst_normalised, msl_normalised, v10_normalised, u10_normalised, all_data

sst_normalised, msl_normalised, v10_normalised, u10_normalised, all_data = get_Bimonthlies()


"""
FUNCTION: perform EOF analysis using package eofs.standard
(Takes < 1.5min for all the seasons)

OUTPUT: 
solver = object returned by Eof package
eofs = spatial patterns
pcs = timeseries data of principal components

1. Use Eof package to find EOF loading pattern and principal components of each dataset
2. Rescale the data
3. Ensure consistent sign convention
"""

def getEofs():
    print("Performing EOF analysis... wait ~1.5mins please... \n\n")        # status update

    # lists to contain the solved EOF matrices and PC1 timeseries for each bi-monthly season
    solvers = []
    eofs = []
    pcs = []
    for i in range(12):
        print(f"Working on the {i}th season... \n")        # print a status update at each iteration

        # *** step 1
        data_matrix = all_data[i].fillna(0).values      # replace null values with 0s
        solver = Eof(data_matrix) 

        # *** step 2
        # Retrieve eigenvalues
        eigenvalue = solver.eigenvalues(neigs=1)  # Only need the first one for PC1
        singular_value = np.sqrt(eigenvalue[0])   # Singular value for PC1

        # extract loading pattern and principal component time series
        eof = solver.eofs(neofs=1)
        PC1 = solver.pcs(npcs=1)

        # rescale according to sqrt(eigenvalue)
        eof_rescaled = eof*singular_value
        PC1_rescaled = PC1/singular_value

        # *** step 3
        # ensuring consistent sign convention throughout the 4x12 result sets
        if eof_rescaled[0, 0] < 0:  
            eof_rescaled = -eof_rescaled 
            PC1_rescaled = -PC1_rescaled   

        # save results
        eofs.append( eof_rescaled )
        pcs.append( PC1_rescaled )
        solvers.append(solver)       

        # show the variance explained
        variance = solver.varianceFraction(neigs=1)[0] * 100
        print(f"The variance of the datasets explained by the first EOF for {bi_monthly_labels[i]}: {variance:.2f}% \n\n")           
    
    return solvers, eofs, pcs


solvers, eofs, pcs = getEofs()


"""
FUNCTION: plot the loading pattern (contour map) of the EOF analysis for December-January

INPUT: 
loadings = eofs output from the solver
month = string to use in the plot title
"""
def plot_DecJan_EOFs(loadings, month):

    # endpoints of the slices for plotting
    slices = [0, len(sst_normalised[1]), 
              len(msl_normalised[1]), 
              len(u10_normalised[1]), 
              len(v10_normalised[1])]
    start = slices[0]       # initialise slice locations
    end = slices[1]

    # force colorbar scale limits
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    # filenames for saving quickly
    filenames = ['sst', 'msl', 'meridional', 'zonal']
    
    # plot each of the loading patterns and save them 
    for i in range(4):

        # prepare figure
        fig, ax = plt.subplots( figsize = (15,8), )

        # slice the block of (sea) data corresponding to this variable
        block = loadings[start:end]

        # place data onto the canvas
        canvas[sea_rows, sea_cols] = block                  # place sea data at sea coordinates
        canvas[land_rows, land_cols] = 0                    # place 0s at land coords
        
        # plot the loading pattern
        T = ax.contourf(lon, lat, canvas, cmap='bwr', norm=norm, levels=20)
        ax.set_xlabel('Longitude', fontsize = 15)
        ax.set_ylabel('Latitude', fontsize=15)
        cbar = plt.colorbar(T, ax=ax, shrink = 0.6,)
    
        # draw black coastlines and colour land white
        m = Basemap(llcrnrlat=lat.min(), urcrnrlat=lat.max(), llcrnrlon=lon.min(), urcrnrlon=lon.max(), ax=ax)
        m.drawcoastlines(ax=ax, linewidth=0.8)
        m.fillcontinents(color='white')

        # set axis ticks
        ax.tick_params(axis='both', which='both', labelsize=12) 
        ax.set_xticks(lon[::50])
        ax.set_yticks(lat[::20])

        plt.title(f"First EOF Loading Pattern of {names[i]} for {month}", fontsize=17)
        plt.tight_layout()
        plt.savefig( f"DecJan_{filenames[i]}_EOF.png" )

        # update the start and end points for the next slice
        start = end
        if i < 3:
            end += slices[i+2] 

# Dec-Jan corresponds to the 0th bimonthly group -> use index 0
plot_DecJan_EOFs(eofs[0][0], bi_monthly_labels[0])



"""
FUNCTION: plot the index time series of the EOF analysis for December-January
"""
def plot_DecJan_TimeSeries():

    # save the year (and its index) of every 50th data point, to plot on the x axis
    indices =[]
    years = []
    for i, year in enumerate(times):        # iterate through measurement dates
        if i%50 == 0:
            indices.append(i)
            years.append(year.astype('datetime64[Y]').astype(int) + 1970)
    

    # plot each bi-monthly season's time series
    fig, ax = plt.subplots(1,1, figsize = (12,4))
    # for i in range(1):
        
    # plot data and gridlines
    ax.plot(pcs[0], color='mediumblue')
    ax.grid(ls='--')

    ax.tick_params(axis='both', which='major', labelsize=12) # change size of y axis ticks
    ax.set_xticks(indices, years, rotation = 45, fontsize=12)
    ax.set_title(f'Index Time Series for {bi_monthly_labels[0]}', fontsize=15)
    ax.set_xlabel('Year', fontsize=15)
    ax.set_ylabel('Amplitude', fontsize=15)

    # make sure plots don't overlap
    plt.tight_layout()      

    plt.savefig("DecJan_TimeSeries.pdf")

plot_DecJan_TimeSeries()



"""
FUNCTION: plot the loading patterns (contour maps) of the EOF analyses for a given bimonthly season

method: slice through the output matrix at indices corresponding to the lengths of each variable's dataset

INPUT: 
loadings = output from the solver
month = string stating which bi-monthly season to plot

OUTPUT: 
fig = figure of 4 subplots (one for each variable)
"""
def plot_all_loadingPatterns(loadings, month):

    # prepare figure
    fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (15,8), )
    ax = ax.ravel()         # allows iteration through 2x2=4 axes

    # endpoints of the slices for plotting
    slices = [0, len(sst_normalised[1]), 
              len(msl_normalised[1]), 
              len(u10_normalised[1]), 
              len(v10_normalised[1])]
    start = slices[0]       # initialise slice locations
    end = slices[1]

    # Create a custom normalization
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    # plot the loading pattern for all 4 variables
    for i in range(4):

        # slice the block of (sea) data corresponding to this variable
        block = loadings[start:end]

        # place data onto the canvas
        canvas[sea_rows, sea_cols] = block                  # place sea data at sea coordinates
        canvas[land_rows, land_cols] = 0                    # place 0s at land coords

        # plot the loading pattern
        T = ax[i].contourf(lon, lat, canvas, levels=20, cmap='bwr', norm=norm)
        cbar = plt.colorbar(T, ax=ax[i], shrink = 0.6,)
        
        ax[i].set_title(names[i], fontsize=13)
        ax[i].set_xlabel('Longitude', fontsize=12)
        ax[i].set_ylabel('Latitude', fontsize=12)

        # draw black coastlines; colour land white
        m = Basemap(llcrnrlat=lat.min(), urcrnrlat=lat.max(), llcrnrlon=lon.min(), urcrnrlon=lon.max(), ax=ax[i])
        m.drawcoastlines(ax=ax[i], linewidth=0.8)
        m.fillcontinents(color='white')

        # set axis ticks
        ax[i].tick_params(axis='both', which='both', labelsize=10) 
        ax[i].set_xticks(lon[::50],)
        ax[i].set_yticks(lat[::20])

        # update the start and end points for the next slice
        start = end
        if i < 3:
            end += slices[i+2] 

    plt.suptitle(f"First EOF Loading Patterns: {month}", fontsize=15)
    plt.tight_layout()

    # return figure for saving in pdf
    return fig


"""
FUNCTION: create & save loading patterns for all 12 bimonthly seasons

NOTE: color scales have been standardised to make comparisons from month to month easier.
The spatial pattern of Dec-Jan is unchanged, but the colours are more intense on the singular plot
produced by plot_DecJan_EOFs for more aesthetic use in the report.
"""
def All_loadings():

    # create pdf
    pp = PdfPages('All_Loadings.pdf')

    # create each month's figure; add to the pdf
    for i, eof in enumerate(eofs):
        figure  = plot_all_loadingPatterns(eof[0], bi_monthly_labels[i])
        pp.savefig(figure)

    # finish with pdf
    pp.close()

All_loadings()


"""
FUNCTION: create & save time series charts for all 12 bimonthly seasons
"""
def PlotAll_TimeSeries():
    
    # save the year (and its index) of every 50th data point, to plot on the x axis
    indices =[]
    years = []
    xvals = times        # make time coordinates iterable
    for i, year in enumerate(xvals):
        if i%50 == 0:
            indices.append(i)
            years.append(year.astype('datetime64[Y]').astype(int) + 1970)
    

    # plot each bi-monthly season's time series
    fig, ax = plt.subplots(12,1, figsize = (12,48))
    for i in range(12):
        
        # plot data and gridlines
        ax[i].plot(pcs[i], color='mediumblue')
        ax[i].grid(ls='--')

        ax[i].tick_params(axis='both', which='major', labelsize=12) # change size of y axis ticks
        ax[i].set_xticks(indices, years, rotation = 45, fontsize=12)

        ax[i].set_title(f'Index Time Series for {bi_monthly_labels[i]}', fontsize=15)
        ax[i].set_xlabel('Year', fontsize=15)
        ax[i].set_ylabel('Amplitude', fontsize=15)

        # make sure plots don't overlap
        plt.tight_layout()      

    plt.savefig("All_TimeSeries.pdf")

PlotAll_TimeSeries()