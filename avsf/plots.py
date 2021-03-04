
from pathlib import Path
from datetime import datetime

import netCDF4 as nc 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs

from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import kde
from KDEpy import FFTKDE

import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes

class PlotsAVSF:

    def __init__(self, base_dir):

        self.base_dir = Path(base_dir)

        self.ds = {f"run_{run}": nc.MFDataset(list((self.base_dir/f"T63_short_r{run}/POST/").glob("BOT_*.nc"))) for run in [1,2,3,4,5,6,8,9,10,11,12]}

        self.years = list(np.array([int(str(d)[:4]) for d in self.ds["run_1"]["time"][:]]).reshape(-1,12)[:,0])
    
    def compare(self, lat, lon, run_data, year_data, years, extent = "global", cmap = plt.cm.Reds, figtitle = None, bartitle = None, **kwargs):
        run_data = np.concatenate([run_data[:,:,lon > 180],run_data[:,:,lon <= 180]], axis = -1)
        year_data = np.concatenate([year_data[:,lon > 180],year_data[:,lon <= 180]], axis = -1)
        lon = np.concatenate([lon[lon > 180] - 360, lon[lon <= 180]], axis = -1)
        
        if extent == "europe":
            year_data = year_data[np.logical_and(lat>=34., lat<=59.),:][:,np.logical_and(lon>=-12., lon<=32.)]
            run_data = run_data[:,np.logical_and(lat>=34., lat<=59.),:][:,:,np.logical_and(lon>=-12., lon<=32.)]
            lon = lon[np.logical_and(lon>=-12., lon<=32.)]
            lat = lat[np.logical_and(lat>=34., lat<=59.)]
        
        fig = plt.figure(dpi = 300)
        fig.suptitle(figtitle)
        projection = ccrs.PlateCarree()
        # axes_class = (GeoAxes,
        #           dict(map_projection=projection))
        # axgr = AxesGrid(fig, 111, axes_class=axes_class,
        #             nrows_ncols=(3,4),
        #             axes_pad=0.25,
        #             cbar_location='bottom' if extent != "europe" else 'right',
        #             cbar_mode='single',
        #             cbar_pad=-0.2 if extent != "europe" else 0.2,
        #             cbar_size='2%',
        #             label_mode='')
        
        # for ax in axgr.axes_row[0]+axgr.axes_row[1]:
        #     ax.remove()

        levels = np.linspace(min(year_data.min(),run_data.min()), max(year_data.max(),run_data.max()),8) 

        gs = gridspec.GridSpec(3, 4, figure=fig)
        axes = []
        ax = fig.add_subplot(gs[:2,:2], projection = projection)
        if extent == "global":
            ax.set_global()
        elif extent == "europe":
            ax.set_extent([-12.,32.,34.,58.])
        ax.coastlines()
        cs = ax.contourf(lon, lat, year_data, levels = levels,transform=ccrs.PlateCarree(),cmap=cmap, **kwargs)
        ax.set_title(r'$\sigma_{year}$', loc = 'left', fontsize = 10)
        axes.append(ax)
        ax = fig.add_subplot(gs[:2,2:], projection = projection)
        if extent == "global":
            ax.set_global()
        elif extent == "europe":
            ax.set_extent([-12.,32.,34.,58.])
        ax.coastlines()
        cs = ax.contourf(lon, lat, run_data[0,], levels = levels,transform=ccrs.PlateCarree(),cmap=cmap, **kwargs)
        ax.set_title(r'$\sigma_{run}$ '+years[0], loc = 'left', fontsize = 10)
        axes.append(ax)
        for idx in range(4):
            ax = fig.add_subplot(gs[2,idx], projection = projection)
            if extent == "global":
                ax.set_global()
            elif extent == "europe":
                ax.set_extent([-12.,32.,34.,58.])
            ax.coastlines()
            cs = ax.contourf(lon, lat, run_data[idx+1,:], levels = levels,transform=ccrs.PlateCarree(),cmap=cmap, **kwargs)
            ax.set_title(r'$\sigma_{run}$ '+years[idx+1], loc = 'left', fontsize = 10)
            axes.append(ax)

        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, wspace=0.02, hspace=0.02)
        cb_ax = fig.add_axes([0.88, 0.4, 0.01, 0.2])

        cbar = plt.colorbar(cs, cax = cb_ax, orientation = "vertical", use_gridspec=False)
        cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in levels], fontsize=10)
        cbar.set_label(bartitle, wrap=True, fontsize = 10) 
        return fig

    def compare_temp(self, extent = "global"):
        year_data = np.stack([self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...] for run in self.ds]).mean(0).std(0, ddof = 1)[2,...]
        run_data = np.stack([self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...] for run in self.ds]).std(0, ddof=1)[:5,...][:,2,...]
        years = [str(i) for i in sorted(self.years)[:5]]
        lon = self.ds["run_1"]["lon"][:]
        lat = self.ds["run_1"]["lat"][:]
        fig = self.compare(lat, lon, run_data, year_data, years, extent = extent, cmap = plt.cm.Reds, figtitle = "Compare 2m temp in June", bartitle = "2m temp in K")

        return fig

    def compare_rain(self, extent = "global"):
        year_data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...]) for run in self.ds])*86400).mean(0).std(0, ddof=1)[2,...]
        run_data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...]) for run in self.ds])*86400).std(0, ddof=1)[:5,...][:,2,...]
        years = [str(i) for i in sorted(self.years)[:5]]
        lon = self.ds["run_1"]["lon"][:]
        lat = self.ds["run_1"]["lat"][:]
        fig = self.compare(lat, lon, run_data, year_data, years, extent = extent, cmap = plt.cm.Blues, figtitle = "Compare Precipitation in June", bartitle = r'Precip in mm d$^{-1}$'")

        return fig

    def summer_monthly(self, lon, lat, data, extent, over_dim = "", for_dim = "", variable = "", unit = "", value = "sample std. dev.", cmap = plt.cm.Reds, figtitle = None, bartitle = None, **kwargs):
    
        data = np.concatenate([data[:,:,lon > 180],data[:,:,lon <= 180]], axis = -1)
        lon = np.concatenate([lon[lon > 180] - 360, lon[lon <= 180]], axis = -1)
        
        if extent == "europe":
            data = data[:,np.logical_and(lat>=34., lat<=59.),:][:,:,np.logical_and(lon>=-12., lon<=32.)]
            lon = lon[np.logical_and(lon>=-12., lon<=32.)]
            lat = lat[np.logical_and(lat>=34., lat<=59.)]

        if not set(["vmin", "vmax"]).issubset(set(kwargs.keys())):
            kwargs["vmin"] = data.min()
            kwargs["vmax"] = data.max()
        

        fig = plt.figure(dpi = 300)
        #fig, axes = plt.subplots(2, 3, dpi = 300, projection=ccrs.PlateCarree())
        if figtitle is None:
            fig.suptitle(f"{variable} {value}\nover {over_dim} for {for_dim}")
        else:
            fig.suptitle(figtitle)
        projection = ccrs.PlateCarree()
        axes_class = (GeoAxes,
                  dict(map_projection=projection))
        axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(2,3),
                    axes_pad=0.25,
                    cbar_location='bottom' if extent != "europe" else 'right',
                    cbar_mode='single',
                    cbar_pad=-0.2 if extent != "europe" else 0.2,
                    cbar_size='2%',
                    label_mode='')
        #gs = gridspec.GridSpec(2,3)
        for idx, month in enumerate(["April","May","June","July","August","September"]):
            ax = axgr[idx]
            #ax = axes.flat[idx]
            #ax = plt.subplot(gs[idx], projection=ccrs.PlateCarree())
            #ax = fig.add_subplot(2, 3, idx+1, projection=ccrs.PlateCarree())
            if extent == "global":
                ax.set_global()
            elif extent == "europe":
                ax.set_extent([-12.,32.,34.,58.])
            ax.coastlines()            
            levels = np.linspace(kwargs["vmin"], kwargs["vmax"],8) 
            if "norm" in kwargs and not 0 in levels and kwargs["vmax"]>0:
                levels = np.linspace(kwargs["vmin"], 0, max(2,int((abs(kwargs["vmin"])/(abs(kwargs["vmin"])+kwargs["vmax"]) )* 7))+1)
                levels = np.append(levels, np.linspace(0, kwargs["vmax"], max(2,int((abs(kwargs["vmax"])/(abs(kwargs["vmin"])+kwargs["vmax"]) ) * 7)))[1:])
            #print(levels)
            cs = ax.contourf(lon, lat, data[idx,...], levels = levels,transform=ccrs.PlateCarree(),cmap=cmap, **kwargs)

            #ax.set_title(f"Area: {extent} Month: {month}.", loc='center', wrap=True, fontsize = 10)
            ax.set_title(f"{month}", loc = 'left', fontsize = 10)
            #plt.tight_layout()
            
            # color bar
        # fig.subplots_adjust(bottom=0.85)
        # cbar_ax = fig.add_axes([0.05, 0.05, 0.9, 0.05])
        # if set(["vmin", "vmax"]).issubset(set(kwargs.keys())):
        #     m = plt.cm.ScalarMappable(cmap=cmap)
        #     m.set_array(np.linspace(kwargs["vmin"], kwargs["vmax"],9))
        #     m.set_clim(kwargs["vmin"], kwargs["vmax"])
        #     cbar = fig.colorbar(cs, ax=cbar_ax, ticks = np.linspace(kwargs["vmin"], kwargs["vmax"],5), orientation="horizontal")
        # else:
        #     cs.set_clim(data.min(),data.max())
        #     cbar = fig.colorbar(cs)
        # if bartitle is None:
        #     cbar.set_label(value + f" of\n{variable}\nin {unit}", wrap=True, fontsize = 6)
        # else:
        #     cbar.set_label(bartitle, wrap=True, fontsize = 6)
        #plt.tight_layout()
        #fig.subplots_adjust(hspace=0, top = 1)
        #fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        #fig.subplots_adjust(bottom=0.85)
        #plt.draw()
        #divider = make_axes_locatable(plt.gca())
        #cbar_ax = divider.append_axes("bottom", "5%", pad="3%")
        #cbar_ax = fig.add_axes([0.25, 0.12, 0.5, 0.04])
        #cbar = fig.colorbar(cs, ax=gs.ravel().tolist(), orientation="horizontal", use_gridspec=True, shrink = 0.5)
        # cbar = ax.cax.colorbar(cs)
        # cbar = grid.cbar_axes[0].colorbar(cs)
        # if not "norm" in kwargs:
        #     m = plt.cm.ScalarMappable(cmap=cmap)
        #     m.set_array(np.linspace(kwargs["vmin"], kwargs["vmax"],9))
        #     m.set_clim(kwargs["vmin"], kwargs["vmax"]) 
        #     cbar = fig.colorbar(m, ax=axgr, shrink=0.6, ticks = np.linspace(kwargs["vmin"], kwargs["vmax"],5), orientation = "horizontal" if extent != "europe" else 'vertical', pad = 0.1) 
        # else:
        if extent != "europe":
            cbar = fig.colorbar(cs, ax=axgr, shrink=0.6, orientation = "horizontal" , pad = 0.1) 
            cbar2 = plt.colorbar(cs, cax = axgr.cbar_axes[0], orientation = "horizontal", fraction = 0.5, use_gridspec=True)
            cbar.ax.set_xticklabels(['{:.2f}'.format(x) for x in levels], fontsize=10)
            cbar2.ax.tick_params(size=0)
            cbar2.set_ticks([])
            cbar2.solids.set(alpha=0)
            cbar2.outline.set_visible(False)
        else:
            cbar = plt.colorbar(cs, cax = axgr.cbar_axes[0], orientation = "vertical", fraction = 0.5, use_gridspec=True)
            cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in levels], fontsize=10)
            #cbar = fig.colorbar(cs, ax=axgr, shrink=0.6, orientation = 'vertical', pad = 0.1) 
        #cbar = axgr.cbar_axes[0].colorbar(cs)
        
        #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, orientation = "horizontal")
        if bartitle is None:
            cbar.set_label(value + f" of {variable} in {unit}", wrap=True, fontsize = 10)
        else:
            cbar.set_label(bartitle, wrap=True, fontsize = 10)   
        #fig.subplots_adjust(hspace=0)
        fig.tight_layout()#pad = 0.3)     
        #gs.tight_layout(fig)
        return fig

    def summer_temp_inter_year(self, run, extent = "global"):
        if run == "mean":
            data = np.stack([self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...] for run in self.ds]).mean(0).std(0, ddof = 1)
            lon = self.ds["run_1"]["lon"][:]
            lat = self.ds["run_1"]["lat"][:]
            run = "mean of 11 runs"
        else:
            data = self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...].std(0, ddof=1)
            lon = self.ds[run]["lon"][:]
            lat = self.ds[run]["lat"][:]

        fig = self.summer_monthly(lon, lat, data, extent, over_dim = "years 1979-2005", for_dim = run, variable = "2m Temperature",value = r'$\sigma_{year}$', unit = "K", cmap = plt.cm.Reds)

        return fig

    def summer_temp_inter_run(self, year, extent = "global"):
        data = np.stack([self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...][self.years.index(year),...] for run in self.ds]).std(0, ddof=1)
        lon = self.ds["run_1"]["lon"][:]
        lat = self.ds["run_1"]["lat"][:] 

        fig = self.summer_monthly(lon, lat, data, extent, over_dim = "runs", for_dim = year, variable = "2m Temperature",value = r'$\sigma_{run}$' , unit = "K", cmap = plt.cm.Reds)

        return fig

    def summer_temp_std_diff(self, year, extent = "global"):
        inter_year_data = np.stack([self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...] for run in self.ds]).mean(0).std(0, ddof = 1)
        lon = self.ds["run_1"]["lon"][:]
        lat = self.ds["run_1"]["lat"][:]
        inter_run_data = np.stack([self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...][self.years.index(year),...] for run in self.ds]).std(0, ddof=1)

        fig = self.summer_monthly(lon, lat, inter_year_data-inter_run_data, extent, figtitle = f"(inter-year - inter-run) std. dev.\nfor 2m temperature in {year}", bartitle = r'$\sigma_{year} - \sigma_{run}$' + " for 2m temperature in K", cmap = plt.cm.BrBG, norm=colors.TwoSlopeNorm(vmin=(inter_year_data-inter_run_data).min(), vcenter=0, vmax=(inter_year_data-inter_run_data).max()))

        return fig

    def summer_temp_std_count(self, extent = "global"):
        inter_year_data = np.stack([self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...] for run in self.ds]).mean(0).std(0, ddof = 1)
        lon = self.ds["run_1"]["lon"][:]
        lat = self.ds["run_1"]["lat"][:]
        inter_run_data = np.stack([self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...] for run in self.ds]).std(0, ddof=1)
        count = np.where(inter_run_data < inter_year_data, 1, 0).mean(0)

        fig = self.summer_monthly(lon, lat, count, extent, figtitle = "%years where inter-year > inter-run std. dev.\nfor 2m temperature", bartitle = r'$\sigma_{year} > \sigma_{run}$' + " for 2m temperature in %years", cmap = plt.cm.Greens, vmin = 0, vmax = 1)

        return fig


    def summer_rain_inter_year(self, run, extent = "global"):
        if run == "mean":
            data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...]) for run in self.ds])*86400).mean(0).std(0, ddof=1)
            lon = self.ds["run_1"]["lon"][:]
            lat = self.ds["run_1"]["lat"][:]
            run = "mean of 11 runs"
        else:
            data = ((self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...] + self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...])*86400).std(0, ddof=1)
            lon = self.ds[run]["lon"][:]
            lat = self.ds[run]["lat"][:]

        fig = self.summer_monthly(lon, lat, data, extent, over_dim = "years 1979-2005", for_dim = run, variable = "Precip", value = r'$\sigma_{year}$', unit = r'mm d$^{-1}$', cmap = plt.cm.Blues)

        return fig
    
    def summer_rain_inter_run(self, year, extent = "global"):
        data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...])[self.years.index(year),...] for run in self.ds])*86400).std(0, ddof=1)
        lon = self.ds["run_1"]["lon"][:]
        lat = self.ds["run_1"]["lat"][:] 

        fig = self.summer_monthly(lon, lat, data, extent, over_dim = "runs", for_dim = year, variable = "Precip", value = r'$\sigma_{run}$', unit = r'mm d$^{-1}$', cmap = plt.cm.Blues)

        return fig


    def summer_rain_std_diff(self, year, extent = "global"):
        inter_year_data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...]) for run in self.ds])*86400).mean(0).std(0, ddof=1)
        lon = self.ds["run_1"]["lon"][:]
        lat = self.ds["run_1"]["lat"][:]
        inter_run_data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...])[self.years.index(year),...] for run in self.ds])*86400).std(0, ddof=1)

        fig = self.summer_monthly(lon, lat, inter_year_data-inter_run_data, extent, figtitle = f"(inter-year - inter-run) std. dev.\nfor Precip (large + convective) in {year}", bartitle = r'$\sigma_{year} - \sigma_{run}$' + " for Precip in " + r'mm d$^{-1}$', cmap = plt.cm.BrBG, norm=colors.TwoSlopeNorm(vmin=(inter_year_data-inter_run_data).min(), vcenter=0, vmax=(inter_year_data-inter_run_data).max()))

        return fig

    def summer_rain_std_count(self, extent = "global"):
        inter_year_data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...]) for run in self.ds])*86400).mean(0).std(0, ddof=1)
        lon = self.ds["run_1"]["lon"][:]
        lat = self.ds["run_1"]["lat"][:]
        inter_run_data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...]) for run in self.ds])*86400).std(0, ddof=1)
        count = np.where(inter_run_data < inter_year_data, 1, 0).mean(0)

        fig = self.summer_monthly(lon, lat, count, extent, figtitle = "%years where inter-year > inter-run std. dev.\nfor Precip (large + convective)", bartitle = r'$\sigma_{year} > \sigma_{run}$' + " for Precip in %years", cmap = plt.cm.Greens, vmin = 0, vmax = 1)

        return fig

    def monthly_mean_timeseries(self, data, figtitle = "timeseries of inter-year -- inter-run std over years", ylabel = r'$\sigma_{year} - \sigma_{run}$'):
        
        fig = plt.figure(dpi = 300)
        fig.suptitle(figtitle)
        #ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel(ylabel)
        yearsort = np.array(self.years).argsort()
        years = np.array(self.years)[yearsort]
        colors = ["#003f5c","#444e86","#955196","#dd5182","#ff6e54","#ffa600"]
        for idx, month in enumerate(["April","May","June","July","August","September"]):
            ax.plot(years,data[:,idx][yearsort],label = month, color = colors[idx])

        ax.legend(bbox_to_anchor=(1,1), loc="upper left")

        return fig 

    def summer_temp_std_diff_ts(self, extent = "global"):
        lat = self.ds["run_1"]["lat"][:]
        lon = self.ds["run_1"]["lon"][:]
        raw_data = np.stack([self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...] for run in self.ds])
        if extent == "europe":
            raw_data = raw_data[:,:,:,np.logical_and(lat>=34., lat<=59.),:][:,:,:,:,np.logical_and(lon>=-12., lon<=32.)]
            lat = lat[np.logical_and(lat>=34., lat<=59.)]
        w = np.cos(lat/180*np.pi)
        w = w/w.sum()
        inter_year = (raw_data.mean((0,4))*w).sum(2).std(0, ddof = 1)
        inter_run = (raw_data.mean(4)*w).sum(3).std(0, ddof = 1)
        data = inter_year - inter_run

        fig = self.monthly_mean_timeseries(data, figtitle = f"2m Temp, timeseries (inter-year - inter-run) std. dev.,\nmean over {extent}", ylabel = r'$(\sigma_{year} - \sigma_{run})$ of 2m Temp in K')

        return fig

    def summer_rain_std_diff_ts(self, extent = "global"):
        lat = self.ds["run_1"]["lat"][:]
        lon = self.ds["run_1"]["lon"][:]
        raw_data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...]) for run in self.ds])*86400)
        if extent == "europe":
            raw_data = raw_data[:,:,:,np.logical_and(lat>=34., lat<=59.),:][:,:,:,:,np.logical_and(lon>=-12., lon<=32.)]
            lat = lat[np.logical_and(lat>=34., lat<=59.)]
        w = np.cos(lat/180*np.pi)
        w = w/w.sum()
        inter_year = (raw_data.mean((0,4))*w).sum(2).std(0, ddof = 1)
        #print(inter_year)
        inter_run = (raw_data.mean(4)*w).sum(3).std(0, ddof = 1)
        data = inter_year - inter_run

        fig = self.monthly_mean_timeseries(data, figtitle = f"Precip (large + convective), timeseries (inter-year - inter-run) std. dev.,\nmean over {extent}", ylabel = r'$(\sigma_{year} - \sigma_{run})$ of'+"\nPrecip (large + convective) in"+r"mm d$^{-1}$")


        #print( ((np.stack([(self.ds[run]["aprc"][:]+self.ds[run]["aprl"][:]).reshape(-1,12,96,192) for run in self.ds])*86400).mean((0,2,4))*w).sum(1).std(0, ddof = 1) )

        #print( ((np.stack([(self.ds[run]["aprc"][3:-3,...]+self.ds[run]["aprl"][3:-3,...]) for run in self.ds])*86400).mean((0,3))*w).sum(1).std(0, ddof = 1) )

        return fig

    def summer_temp_anomaly_ts(self, extent = "global"):
        lat = self.ds["run_1"]["lat"][:]
        lon = self.ds["run_1"]["lon"][:]
        raw_data = np.stack([self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...] for run in self.ds])
        if extent == "europe":
            raw_data = raw_data[:,:,:,np.logical_and(lat>=34., lat<=59.),:][:,:,:,:,np.logical_and(lon>=-12., lon<=32.)]
            lat = lat[np.logical_and(lat>=34., lat<=59.)]
        w = np.cos(lat/180*np.pi)
        w = w/w.sum()
        inter_year = (raw_data.mean((0,4))*w).sum(2).mean(0)
        anom = ((raw_data.mean(4)*w).sum(3) - inter_year)
        anom_mean = anom.mean(0) #mean over runs
        data = np.where(np.sign(anom) == np.sign(anom_mean), 1, 0).mean(0)

        fig = self.monthly_mean_timeseries(data, figtitle = f"2m Temp, timeseries %runs same anomaly direction as mean,\nmean over {extent}", ylabel = r"% runs anomaly same direction as mean for 2m Temp")

    def summer_rain_anomaly_ts(self, extent = "global"):
        lat = self.ds["run_1"]["lat"][:]
        lon = self.ds["run_1"]["lon"][:]
        raw_data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...]) for run in self.ds])*86400)
        if extent == "europe":
            raw_data = raw_data[:,:,:,np.logical_and(lat>=34., lat<=59.),:][:,:,:,:,np.logical_and(lon>=-12., lon<=32.)]
            lat = lat[np.logical_and(lat>=34., lat<=59.)]
        w = np.cos(lat/180*np.pi)
        w = w/w.sum()
        inter_year = (raw_data.mean((0,4))*w).sum(2).mean(0)
        anom = ((raw_data.mean(4)*w).sum(3) - inter_year)
        anom_mean = anom.mean(0) #mean over runs
        data = np.where(np.sign(anom) == np.sign(anom_mean), 1, 0).mean(0)

        fig = self.monthly_mean_timeseries(data, figtitle = f"Precip (large + convective), timeseries %runs same anomaly direction as mean,\nmean over {extent}", ylabel = r"% runs anomaly same direction as mean"+"\nfor Precip (large + convective)")


    def density_anomalies(self, x, y, figtitle = "density estimate of anomalies", xlabel = "mean anomaly of all runs", ylabel = "anomaly", xlim = (-4,4), ylim = (-4,4), weights = None, extent = "global"):
        
        fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, dpi = 300)
        fig.suptitle(figtitle)
        colors = ["#003f5c","#444e86","#955196","#dd5182","#ff6e54","#ffa600"]
        for idx, month in enumerate(["April","May","June","July","August","September"]):
            kde = FFTKDE(kernel='gaussian', norm=2)
            grid, points = kde.fit(np.stack([x[:,idx],y[:,idx]]).T, weights = None if weights is None else weights[:, idx]).evaluate(grid_points = (101,101))
            xi, yi = np.meshgrid(np.unique(grid[:, 0]), np.unique(grid[:, 1]))
            zi = points.reshape((101,101)).T
            
            ax = axes[idx//2,idx%2]

            ax.contourf(xi, yi, zi, cmap = plt.cm.Blues, alpha = 1)
            mask = (np.sign(xi) == np.sign(yi)) | (xi == 0) | (yi == 0 )
            zi2 = np.ma.masked_array(zi, mask)
            ax.contourf(xi, yi, zi2, cmap = plt.cm.Oranges, alpha = 1)

            # data = np.stack([x[:,idx],y[:,idx]]).T       
            # u, v = data[np.random.choice(data.shape[0], 10000, replace=False)].T

            # ax.scatter(u, v, s=1, c = "grey", marker = ".", rasterized = True)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.set_title(f"Area: {extent} Month: {month}.", loc='center', wrap=True, fontsize = 10)
            plt.tight_layout()
        
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel) 
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.875, 0.3, 0.04, 0.25])
        cs = ax.contourf(xi, yi, zi, cmap = plt.cm.Greys, alpha = 0)
        cs.set_alpha(1)
        fig.colorbar(cs, cax = cbar_ax )
        return fig

    def summer_temp_anomaly_density(self, extent = "global"):
        lat = self.ds["run_1"]["lat"][:]
        lon = self.ds["run_1"]["lon"][:]
        raw_data = np.stack([self.ds[run]["temp2"][3:-3,...].reshape(-1,6,96,192)[::2,...] for run in self.ds])
        if extent == "europe":
            raw_data = raw_data[:,:,:,np.logical_and(lat>=34., lat<=59.),:][:,:,:,:,np.logical_and(lon>=-12., lon<=32.)]
            lat = lat[np.logical_and(lat>=34., lat<=59.)]
        inter_year = raw_data.mean((0,1))
        anom = raw_data - inter_year
        x = np.transpose(anom.mean(0, keepdims = True).repeat(11, axis = 0), (0,1,3,4,2)).reshape((-1,6))
        y = np.transpose(anom, (0,1,3,4,2)).reshape((-1,6))

        w = np.cos(lat/180*np.pi)
        w = np.transpose(w[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis].repeat(18 if extent == "europe" else 192, 4).repeat(11, 0).repeat(27, 1).repeat(6, 2), (0,1,3,4,2)).reshape((-1,6))
        w = w / w.sum(0)

        fig = self.density_anomalies(x, y, figtitle = "Density estimate of 2m temp anomalies\n(over years, runs, weighted grid cells)", xlabel = "2m temp mean anomaly\nof all runs in K", ylabel = "2m temp anomaly in K", weights = w, extent = extent)

        return fig
    
    def summer_rain_anomaly_density(self, extent = "global"):
        lat = self.ds["run_1"]["lat"][:]
        lon = self.ds["run_1"]["lon"][:]
        raw_data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...]) for run in self.ds])*86400)
        if extent == "europe":
            raw_data = raw_data[:,:,:,np.logical_and(lat>=34., lat<=59.),:][:,:,:,:,np.logical_and(lon>=-12., lon<=32.)]
            lat = lat[np.logical_and(lat>=34., lat<=59.)]
        inter_year = raw_data.mean((0,1))
        anom = raw_data - inter_year
        x = np.transpose(anom.mean(0, keepdims = True).repeat(11, axis = 0), (0,1,3,4,2)).reshape((-1,6))
        y = np.transpose(anom, (0,1,3,4,2)).reshape((-1,6))

        w = np.cos(lat/180*np.pi)
        w = np.transpose(w[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis].repeat(18 if extent == "europe" else 192, 4).repeat(11, 0).repeat(27, 1).repeat(6, 2), (0,1,3,4,2)).reshape((-1,6))
        w = w / w.sum(0)

        fig = self.density_anomalies(x, y, figtitle = "Density estimate of Precip (large + convective) anomalies\n(over years, runs, weighted grid cells)", xlabel = "Precip (large + convective) mean anomaly\nof all runs in "+r"mm d$^{-1}$", ylabel = "Precip (large + convective) anomaly in "+r"mm d$^{-1}$", weights = w, extent = extent)

        return fig

    def joint_density_skill_rain(self, extent = "global"):
        lat = self.ds["run_1"]["lat"][:]
        lon = self.ds["run_1"]["lon"][:]
        raw_data = (np.stack([(self.ds[run]["aprc"][3:-3,...].reshape(-1,6,96,192)[::2,...]+self.ds[run]["aprl"][3:-3,...].reshape(-1,6,96,192)[::2,...]) for run in self.ds])*86400)
        if extent == "europe":
            raw_data = raw_data[:,:,:,np.logical_and(lat>=34., lat<=59.),:][:,:,:,:,np.logical_and(lon>=-12., lon<=32.)]
            lat = lat[np.logical_and(lat>=34., lat<=59.)]
        inter_year_data = raw_data.mean(0).std(0, ddof = 1)
        inter_run_data = raw_data.std(0, ddof = 1)
        y = np.transpose(np.where(inter_run_data < inter_year_data, 1, 0).mean(0),  (1,2,0)).reshape((-1,6))
        x = np.transpose(raw_data.mean((0,1)), (1,2,0)).reshape((-1,6))
        w = np.cos(lat/180*np.pi)
        w = np.transpose(w[np.newaxis,:,np.newaxis].repeat(18 if extent == "europe" else 192, 2).repeat(6,0), (1,2,0)).reshape((-1,6))

        fig = plt.figure(dpi = 300)
        fig.suptitle("Joint density of Precip (large + convective) and Skill in Precip forecasting\n(over years, runs, weighted grid cells)")
        mask0 = (x.flatten() < 8) | (y.flatten() > 0.5)
        mask = (y.flatten() > 0.5) & mask0
        ax = fig.add_subplot(2, 1, 1)
        hist = ax.hist2d(x.flatten()[mask0], y.flatten()[mask0], bins = 30, cmap = plt.cm.Greens, range = [[0,x.flatten()[mask0].max()], [0,1]], weights = w.flatten()[mask0] / (w.flatten()[mask0].sum()))
        ax.set_ylim(0, 1)
        ax.set_title(f"Area: {extent}, All data", loc='center', wrap=True, fontsize = 10)
        cbar = fig.colorbar(hist[3])
        ax = fig.add_subplot(2, 1, 2)
        hist = ax.hist2d(x.flatten()[mask], y.flatten()[mask], bins = 30, cmap = plt.cm.Greens, range = [[0,max(x.flatten()[mask].max(), 0.5)], [0.5,1]], weights = w.flatten()[mask]/ (w.flatten()[mask].sum()))
        ax.set_ylim(0.5, 1)
        ax.set_title(f"Area: {extent}, Data with min. 0.5 Skill", loc='center', wrap=True, fontsize = 10)
        cbar = fig.colorbar(hist[3])

        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel("Skill in Precip (large + convective) in %years") 
        plt.xlabel("Precip (large + convective)"+r"mm d$^{-1}$")

        plt.tight_layout()

        return fig

    @classmethod
    def plot(cls, base_dir, out_file):
        self = cls(base_dir)
        with PdfPages(out_file) as pdf:
            for extent in ["global","europe"]:
                fig = self.joint_density_skill_rain(extent = extent)
                pdf.savefig(fig, bbox_inches='tight') 
                plt.close()
                fig = self.summer_temp_anomaly_density(extent = extent)
                pdf.savefig(fig, bbox_inches='tight') 
                plt.close()
                fig = self.summer_rain_anomaly_density(extent = extent)
                pdf.savefig(fig, bbox_inches='tight') 
                plt.close()
                fig = self.summer_temp_std_diff_ts(extent = extent)
                pdf.savefig(fig, bbox_inches='tight') 
                plt.close()
                fig = self.summer_rain_std_diff_ts(extent = extent)
                pdf.savefig(fig, bbox_inches='tight') 
                plt.close()
                fig = self.summer_temp_anomaly_ts(extent = extent)
                pdf.savefig(fig, bbox_inches='tight') 
                plt.close()
                fig = self.summer_rain_anomaly_ts(extent = extent)
                pdf.savefig(fig, bbox_inches='tight') 
                plt.close()
                fig = self.compare_temp(extent = extent)
                pdf.savefig(fig, bbox_inches='tight') 
                plt.close()
                fig = self.compare_rain(extent = extent)
                pdf.savefig(fig, bbox_inches='tight') 
                plt.close()
                fig = self.summer_temp_inter_year(run = "mean", extent = extent)
                pdf.savefig(fig) 
                plt.close()
                fig = self.summer_rain_inter_year(run = "mean", extent = extent)
                pdf.savefig(fig) 
                plt.close()
                for run in self.ds:
                    fig = self.summer_temp_inter_year(run = run, extent = extent)
                    pdf.savefig(fig) 
                    plt.close()
                    fig = self.summer_rain_inter_year(run = run, extent = extent)
                    pdf.savefig(fig) 
                    plt.close()
                    #break
                
                fig = self.summer_temp_std_count(extent = extent)
                pdf.savefig(fig) 
                plt.close()
                fig = self.summer_rain_std_count(extent = extent)
                pdf.savefig(fig) 
                plt.close()
                for year in sorted(self.years):
                    fig = self.summer_temp_inter_run(year = year, extent = extent)
                    pdf.savefig(fig) 
                    plt.close()
                    fig = self.summer_temp_std_diff(year = year, extent = extent)
                    pdf.savefig(fig) 
                    plt.close()
                    fig = self.summer_rain_inter_run(year = year, extent = extent)
                    pdf.savefig(fig) 
                    plt.close()
                    fig = self.summer_rain_std_diff(year = year, extent = extent)
                    pdf.savefig(fig) 
                    plt.close()
                    #break

if __name__ == "__main__":
    import fire
    fire.Fire(PlotsAVSF.plot)