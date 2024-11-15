# importing relevant packages

import xarray as xr
import matplotlib.pyplot as plt

# loading in the datasets 
msl_ds = xr.open_dataset("msl_monthly.nc")
sst_ds = xr.open_dataset("sst_monthly.nc")
v10_ds = xr.open_dataset("v10_monthly.nc")
u10_ds = xr.open_dataset("u10_monthly.nc")

# print(msl_ds)
# print(sst_ds)
# print(v10_ds)
# print(u10_ds)

# variables holding dimensions, coordinates of each dataset
msl_dims = msl_ds.dims
msl_coords = msl_ds.coords
sst_dims = sst_ds.dims
sst_coords = sst_ds.coords
v10_dims = v10_ds.dims
v10_coords = v10_ds.coords
u10_dims = u10_ds.dims
u10_coords = u10_ds.coords

# showing an example dataset to see what prints
print(msl_dims)
print(type(msl_dims), "\n")
for value, key in msl_dims.items():
    print(value, key, "\n")


print(msl_coords["longitude"], "\n")
print(msl_coords["longitude"][0], "\n")
print("The above information is stored in a ", type(msl_coords["longitude"][0]), "datatype \n")


# printing number of longitude points 
print("Length of longitude array in msl_monthly.nc:")
print(len(msl_coords["longitude"]))

# plotting them to see if the msl["longitude"] variable 
plt.plot(msl_coords["longitude"])
plt.show()