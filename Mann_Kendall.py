import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import geopandas as gpd
import pymannkendall as mk
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dask.diagnostics import ProgressBar
from matplotlib.colors import SymLogNorm
import matplotlib.colors as mcolors


# Função para calcular o teste de Mann-Kendall
def calcular_mann_kendall(series):
    if np.all(np.isnan(series)):
        return None
    result = mk.original_test(series)
    return result

# Carrega o shapefile usando geopandas
shapefile_path = "caminho_para_seu_arquivo/bacia_pindare_srtm.shp"
gdf = gpd.read_file(shapefile_path)

if gdf.crs != 'EPSG:4326':
    gdf = gdf.to_crs(epsg=4326)
ferrovia_path = "caminho_para_seu_arquivo/extensao_efc_shp.shp"
ferrovia = gpd.read_file(ferrovia_path)
if ferrovia.crs is None:
    ferrovia.set_crs(epsg=4326, inplace=True)


# Função para carregar e processar os dados NetCDF
def processar_dados(file_pattern):
    files = glob.glob(file_pattern)
    ds = xr.open_mfdataset(files, combine='by_coords')
    #ds = ds.sel(time=slice('2061', '2100'))
    
    # Recorta para a região de interesse
    bbox = gdf.total_bounds
    lat_min, lon_min, lat_max, lon_max = bbox[1], bbox[0], bbox[3], bbox[2]
    lon_min -= 0.5
    lat_max +=0.2
    lat_mask = (ds.lat >= lat_min) & (ds.lat <= lat_max)
    lon_mask = (ds.lon >= lon_min) & (ds.lon <= lon_max)
    
    lat_mask=lat_mask.compute()
    lon_mask=lon_mask.compute()
    
    ds_clipped = ds.where(lat_mask & lon_mask, drop=True)
    
    # Agrega para dados diários e mensais
    precip = ds_clipped['pr'].resample(time='1D').sum(dim='time') * 10800
    precip_mensal = precip.resample(time='M').sum()
    
    # Converte para numpy
    with ProgressBar():
        precip_mensal_np = precip_mensal.compute().values

    return precip_mensal, precip_mensal_np

# Carrega os dois cenários
file_pattern_rcp45 = caminho_para_seu_arquivo/pr*.nc"
file_pattern_rcp85 = "caminho_para_seu_arquivo/pr*.nc"

precip_mensal_rcp45, precip_mensal_np_rcp45 = processar_dados(file_pattern_rcp45)
precip_mensal_rcp85, precip_mensal_np_rcp85 = processar_dados(file_pattern_rcp85)

# Função para calcular Mann-Kendall e criar DataArray
def calcular_mann_kendall_mapa(precip_mensal_np, lat, lon):
    tau_values = np.empty((precip_mensal_np.shape[1], precip_mensal_np.shape[2]))
    p_values = np.empty((precip_mensal_np.shape[1], precip_mensal_np.shape[2]))

    for i in range(precip_mensal_np.shape[1]):
        for j in range(precip_mensal_np.shape[2]):
            series = precip_mensal_np[:, i, j]
            result = calcular_mann_kendall(series)
            if result:
                tau_values[i, j] = result.Tau
                p_values[i, j] = result.p
            else:
                tau_values[i, j] = np.nan
                p_values[i, j] = np.nan

    tau_da = xr.DataArray(tau_values, dims=['y', 'x'], coords={'lat': (['y', 'x'], lat), 'lon': (['y', 'x'], lon)})
    p_da = xr.DataArray(p_values, dims=['y', 'x'], coords={'lat': (['y', 'x'], lat), 'lon': (['y', 'x'], lon)})
    return tau_da, p_da

# Calcula Mann-Kendall para ambos os cenários
tau_rcp45, p_rcp45 = calcular_mann_kendall_mapa(precip_mensal_np_rcp45, 
                                                  precip_mensal_rcp45.lat.values, 
                                                  precip_mensal_rcp45.lon.values)

tau_rcp85, p_rcp85 = calcular_mann_kendall_mapa(precip_mensal_np_rcp85, 
                                                  precip_mensal_rcp85.lat.values, 
                                                  precip_mensal_rcp85.lon.values)






# Função para plotar os dois cenários lado a lado com as médias como ticks na colorbar
def plot_comparativo_tau(tau_rcp45, tau_rcp85, p_rcp45, p_rcp85, gdf):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Ajustar limites de latitude
    lat_min = tau_rcp45.lat.min()
    lon_min = tau_rcp45.lon.min()
    lat_max = tau_rcp45.lat.max()
    lon_max = tau_rcp45.lon.max()
    axes[0].set_ylim(lat_min, lat_max)  
    axes[1].set_ylim(lat_min, lat_max)  
    axes[0].set_xlim(lon_min, lon_max)  
    axes[1].set_xlim(lon_min, lon_max)  
    
    norm = SymLogNorm(linthresh=0.03, linscale=3, vmin=-0.1, vmax=0.1)
    norm_rcp85 = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
    
    # Plot para RCP45
    im1 = axes[0].pcolormesh(tau_rcp45.lon, tau_rcp45.lat, tau_rcp45, cmap='coolwarm_r', norm=norm, shading='auto')
    axes[0].set_title('MK -RegHad - 2020-2100 - RCP 4.5')
    significance_mask_45 = p_rcp45 < 0.05  # Máscara para p-values significativos
    axes[0].contourf(tau_rcp45.lon, tau_rcp45.lat, significance_mask_45, levels=[0.5, 1], hatches=['/'], colors='none', transform=ccrs.PlateCarree())
    gdf.plot(ax=axes[0], edgecolor='black', facecolor='none', linewidth=1)
    ferrovia.plot(ax=axes[0], color='red', linewidth=1.5)
    # Plot para RCP85
    im2 = axes[1].pcolormesh(tau_rcp85.lon, tau_rcp85.lat, tau_rcp85, cmap='coolwarm_r', norm=norm_rcp85, shading='auto')
    axes[1].set_title('MK - RegHad - 2020-2100 - RCP 8.5')
    significance_mask_85 = p_rcp85 < 0.05  # Máscara para p-values significativos
    axes[1].contourf(tau_rcp85.lon, tau_rcp85.lat, significance_mask_85, levels=[0.5, 1], hatches=['/'], colors='none', transform=ccrs.PlateCarree())
    gdf.plot(ax=axes[1], edgecolor='black', facecolor='none', linewidth=1)
    ferrovia.plot(ax=axes[1], color='red', linewidth=1.5)#, label='Ferrovia')

    # Calcula a média de cada cenário
    media_tau_rcp45 = np.nanmean(tau_rcp45.values)
    media_tau_rcp85 = np.nanmean(tau_rcp85.values)

    # Adiciona a colorbar compartilhada
    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.05, pad=0.01, shrink=0.95)
    cbar.set_label('Tau')

    # Configura os ticks da colorbar e adiciona as médias
    #ticks = cbar.get_ticks()  # Obtém os ticks padrão
    #new_ticks = np.append(ticks, [media_tau_rcp45, media_tau_rcp85])  # Adiciona as médias aos ticks
    #cbar.set_ticks(new_ticks)  # Define os novos ticks
    #cbar.ax.set_yticklabels([f'{tick:.3f}' for tick in new_ticks])  # Atualiza os rótulos

    plt.show()

# Plota os resultados
plot_comparativo_tau(tau_rcp45, tau_rcp85, p_rcp45, p_rcp85, gdf)
