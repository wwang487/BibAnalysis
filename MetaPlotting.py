import MetaAnalysis as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy import feature as cfeature
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import LogLocator, FuncFormatter
import numpy as np
import seaborn as sns

from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from geopy.geocoders import Nominatim
import pycountry
import json
import pandas as pd
import geopandas as gpd
from matplotlib.ticker import ScalarFormatter
import squarify
import scipy.stats as stats
import plotly.graph_objects as go
from copy import deepcopy
import textwrap

def wrap_labels(labels, width=14):
    wrapped = []

    for label in labels:
        # Split by space or slash to prioritize logical breaks
        parts = label.replace("/", " / ").split()
        current_line = ""
        lines = []

        for part in parts:
            # If this part makes the line too long, wrap to next line
            if len(current_line) + len(part) + (1 if current_line else 0) > width:
                if current_line:
                    lines.append(current_line.rjust(width))
                if len(part) > width:
                    # Hyphenate very long word
                    for i in range(0, len(part), width - 1):
                        chunk = part[i:i + width - 1]
                        if i + width - 1 < len(part):
                            lines.append((chunk + "-").rjust(width))
                        else:
                            lines.append(chunk.rjust(width))
                    current_line = ""
                else:
                    current_line = part
            else:
                current_line += (" " if current_line else "") + part

        if current_line:
            lines.append(current_line.rjust(width))

        wrapped.append("\n".join(lines))

    return wrapped
# Initialize GeoPy geolocator
geolocator = Nominatim(user_agent="geoapi")

def get_coordinates_with_geopy(country_code):
    """
    Fetch latitude and longitude for a given country code using GeoPy.
    """
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        if not country:
            print(f"Invalid country code: {country_code}")
            return None
        location = geolocator.geocode(country.name)
        if location:
            coords = location[-1]
            return coords
        else:
            print(f"Could not find location for {country.name}")
            return None
    except Exception as e:
        print(f"Error with geopy for {country_code}: {e}")
        return None

def draw_country_pair_map(
    country_pair_dict, 
    country_pair_data, 
    counted_thresh=0, 
    transparent_thresh=None, 
    save_path=None, 
    dpi=300, 
    figsize=(12, 8), 
    title="Country Pair Connections", 
    colorbar_label="Pair Count", 
    title_fontsize=16, 
    label_fontsize=12, 
    tick_fontsize=10,
    color_ramp="hot", 
    line_width=1.5, 
    thinner_ratio=0.5
):
    """
    Draw a map with arcs connecting country pairs, with customizable figure properties, color ramp, and line width.
    """

    def get_coords(alpha2):
        for entry in country_pair_data:
            if entry['alpha2'] == alpha2:
                return entry['longitude'], entry['latitude']
        return get_coordinates_with_geopy(alpha2)

    fig, ax = plt.subplots(figsize=figsize)
    m = Basemap(projection='robin', lon_0=0, resolution='c', ax=ax)

    # Draw coastlines and countries with black boundaries
    m.drawcoastlines(linewidth=0.5, color="black")
    m.drawcountries(linewidth=0.5, color="black")

    # Filter and sort by count
    filtered_pairs = [(k, v) for k, v in country_pair_dict.items() if v > counted_thresh]
    if not filtered_pairs:
        print("No pairs meet the threshold criteria.")
        return
    filtered_pairs.sort(key=lambda x: x[1], reverse=True)

    norm = Normalize(vmin=min(v[1] for v in filtered_pairs), vmax=max(v[1] for v in filtered_pairs))
    cmap = cm.get_cmap(color_ramp)

    for (country1, country2), count in filtered_pairs:
        coord1 = get_coords(country1)
        coord2 = get_coords(country2)
        # print(country1, country2, coord1, coord2)

        if coord1 and coord2:
            # Ensure arcs go from smaller (lat, lon) to larger (lat, lon)
            lon1, lat1 = coord1
            lon2, lat2 = coord2
            if (lat1, lon1) > (lat2, lon2):  # Swap if necessary
                lon1, lat1, lon2, lat2 = lon2, lat2, lon1, lat1

            # Adjust transparency and thickness
            alpha = 0.7
            linewidth = line_width
            if transparent_thresh is not None and count < transparent_thresh:
                alpha = 0.5
                linewidth *= thinner_ratio  # Thinner line for lower counts

            m.drawgreatcircle(lon1, lat1, lon2, lat2,
                              linewidth=linewidth, color=cmap(norm(count)), alpha=alpha)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar.set_label(colorbar_label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    plt.title(title, fontsize=title_fontsize)
    if save_path:
        plt.savefig(save_path, dpi=dpi, transparent=True)
    plt.show()

def format_longitude(x, pos):
    if x > 0:
        return f"{int(x):d}°E"
    elif x < 0:
        return f"{int(-x):d}°W"
    else:
        return "0°"

    # Custom formatter for y-axis (latitude)
def format_latitude(y, pos):
    if y > 0:
        return f"{int(y):d}°N"
    elif y < 0:
        return f"{int(-y):d}°S"
    else:
        return "0°"

def draw_country_pair_map_with_shapefile(
    country_pair_dict,
    country_pair_data,
    shp_folder,
    shp_file,
    counted_thresh=0,
    transparent_thresh=None,
    save_path=None,
    dpi=300,
    figsize=(12, 8),
    title="Country Pair Connections",
    colorbar_label="Pair Count",
    title_fontsize=16,
    label_fontsize=12,
    tick_fontsize=10,
    color_ramp="hot",
    line_width=1.5,
    thinner_ratio=0.5,
    boundary_color="black",
    boundary_width=0.5,
    is_grid=True,
    is_border=True,
    is_tick=True,
    log_scale=False,
    colorbar_orientation="horizontal",
    colorbar_position="bottom",
    group_code=None  # New parameter
):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib import cm
    from matplotlib.ticker import FuncFormatter

    def format_longitude(x, pos):
        return f"{int(x)}°E" if x >= 0 else f"{int(abs(x))}°W"

    def format_latitude(y, pos):
        return f"{int(y)}°N" if y >= 0 else f"{int(abs(y))}°S"

    def get_coords(alpha2):
        for entry in country_pair_data:
            if entry['alpha2'] == alpha2:
                return entry['longitude'], entry['latitude']
        return None

    def create_curve(coord1, coord2, num_points=100):
        lon1, lat1 = coord1
        lon2, lat2 = coord2
        mid_lon = (lon1 + lon2) / 2
        mid_lat = (lat1 + lat2) / 2 + 10
        t = np.linspace(0, 1, num_points)
        curve_lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * mid_lon + t ** 2 * lon2
        curve_lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * mid_lat + t ** 2 * lat2
        return curve_lon, curve_lat

    try:
        gdf = gpd.read_file(f"{shp_folder}/{shp_file}")
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return

    if "Two_letter" not in gdf.columns:
        print("The shapefile does not have a 'Two_letter' column.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Optional background group coloring
    if group_code:
        group_keys = list(group_code.keys())
        num_groups = len(group_keys)
        
        # From 30% gray (0.7) to 0% gray (1.0)
        grays = np.linspace(0.8, 1.0, num_groups+1)  # 0.7 = 30% gray, 1.0 = white
        country_shades = {}

        # Map each country code to a gray level (first key = darkest)
        for gray, key in zip(grays, group_keys):
            for code in group_code[key]:
                country_shades[code] = str(gray)  # Matplotlib expects grayscale as string

        # Plot each country with the assigned gray
        for code, shade in country_shades.items():
            selected = gdf[gdf['Two_letter'] == code]
            selected.plot(ax=ax, color=shade, edgecolor='none')

        # Plot remaining countries in white
        remaining = gdf[~gdf['Two_letter'].isin(country_shades.keys())]
        remaining.plot(ax=ax, color='white', edgecolor='none')
    else:
        gdf.plot(ax=ax, color='white', edgecolor='none')

    if is_border:
        gdf.boundary.plot(ax=ax, color=boundary_color, linewidth=boundary_width)

    filtered_pairs = [(k, v) for k, v in country_pair_dict.items() if v > counted_thresh]
    if not filtered_pairs:
        print("No pairs meet the threshold criteria.")
        return
    filtered_pairs.sort(key=lambda x: x[1], reverse=True)

    norm = LogNorm(vmin=min(v[1] for v in filtered_pairs), vmax=max(v[1] for v in filtered_pairs)) if log_scale else Normalize(vmin=min(v[1] for v in filtered_pairs), vmax=max(v[1] for v in filtered_pairs))
    cmap = cm.get_cmap(color_ramp)

    for (country1, country2), count in filtered_pairs:
        coord1 = get_coords(country1)
        coord2 = get_coords(country2)
        if coord1 and coord2:
            lon1, lat1 = coord1
            lon2, lat2 = coord2
            if (lat1, lon1) > (lat2, lon2):
                lon1, lat1, lon2, lat2 = lon2, lat2, lon1, lat1

            curve_lon, curve_lat = create_curve((lon1, lat1), (lon2, lat2))

            alpha = 0.7
            linewidth = line_width
            if transparent_thresh is not None and count < transparent_thresh:
                alpha = 0.5
                linewidth *= thinner_ratio

            ax.plot(curve_lon, curve_lat, color=cmap(norm(count)), linewidth=linewidth, alpha=alpha)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation=colorbar_orientation, pad=0.05, shrink=0.8, location=colorbar_position)
    cbar.set_label(colorbar_label, fontsize=label_fontsize)

    if log_scale:
        ticks = [10 ** i for i in range(int(np.log10(norm.vmin)), int(np.log10(norm.vmax)) + 1)]
        cbar.set_ticks(ticks)
        cbar.ax.set_xticklabels([f"{int(tick):,}" for tick in ticks])

    if is_grid:
        ax.grid(visible=True, which="major", color="gray", linestyle="--", linewidth=0.5)
    else:
        ax.grid(False)

    if not is_tick:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
        ax.yaxis.set_major_formatter(FuncFormatter(format_latitude))
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, direction="out", length=2.5, width=1)

    plt.title(title, fontsize=title_fontsize)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.show()
    
def plot_country_publications(
    publication_counts,
    shp_folder,
    shp_file,
    title="Global Publications by Country",
    cmap="viridis",
    figsize=(15, 10),
    boundary_color="black",
    boundary_width=0.5,
    gray_zero=True,
    log_scale=False,
    save_path=None,
    dpi=300,
    colorbar_orientation="horizontal",
    colorbar_pad=0.05,
    colorbar_shrink=0.8,
    colorbar_label="Count",
    title_font_size=14,
    tick_fontsize=12,
    is_tick=True,
    lon_range=None,  # Longitude limits
    lat_range=None   # Latitude limits
):
    # Load the shapefile
    try:
        gdf = gpd.read_file(f"{shp_folder}/{shp_file}")
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return

    if "Two_letter" not in gdf.columns:
        print("The shapefile does not have a 'Two_letter' column.")
        return

    pub_df = pd.DataFrame(list(publication_counts.items()), columns=["Two_letter", "Count"])
    gdf = gdf.merge(pub_df, on="Two_letter", how="left")
    gdf["Count"] = gdf["Count"].fillna(0)

    if log_scale:
        gdf["Count"] = gdf["Count"].apply(lambda x: np.log10(x + 1))

    fig, ax = plt.subplots(figsize=figsize)
    norm = plt.Normalize(vmin=np.log10(1), vmax=np.log10(gdf["Count"].max() + 1)) if log_scale else None

    plot = gdf.plot(
        column="Count",
        cmap=cmap,
        linewidth=boundary_width,
        edgecolor=boundary_color,
        ax=ax,
        legend=not log_scale,
        legend_kwds={
            "label": colorbar_label + (" (Log Scale)" if log_scale else ""),
            "orientation": colorbar_orientation,
            "pad": colorbar_pad,
            "shrink": colorbar_shrink,
            "norm": norm,
        }
    )

    if gray_zero:
        zero_gdf = gdf[gdf["Count"] == 0]
        zero_gdf.plot(color="lightgray", linewidth=boundary_width, edgecolor=boundary_color, ax=ax)

    if log_scale:
        cbar = plot.get_figure().colorbar(plot.collections[0], ax=ax, orientation=colorbar_orientation)
        cbar.set_ticks([0, 1, 2, 3, 4])
        cbar.set_ticklabels([10**int(tick) for tick in cbar.get_ticks()])
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(10**x):,}"))

    if not is_tick:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
        ax.yaxis.set_major_formatter(FuncFormatter(format_latitude))
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, direction="out", length=2.5, width=1)

    ax.set_title(title, fontsize=title_font_size)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(is_tick)

    # Apply custom latitude and longitude limits if provided
    if lon_range:
        ax.set_xlim(lon_range)
    if lat_range:
        ax.set_ylim(lat_range)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def draw_country_heatmap_with_disputes(
    country_count_dict,
    disputed_geometries=None,
    save_path=None,
    dpi=300,
    figsize=(16, 8),
    cmap_name='viridis',
    title="Country Heatmap with Disputed Areas",
    title_fontsize=16,
    country_edgecolor="black",
    country_linewidth=0.2,
    country_default_color=(0.9, 0.9, 0.9, 1),
    colorbar_orientation="horizontal",
    colorbar_pad=0.05,
    colorbar_shrink=0.8,
    colorbar_label="Count",
    disputed_color="gray",
    disputed_linewidth=0.5,
    disputed_linestyle="--",
    disputed_edgecolor="black",
    coastline_color="black",
    coastline_linewidth=0.5,
    border_linestyle=":",
    border_edgecolor="black",
    border_linewidth=0.5,
):
    """
    Draws a heatmap of countries using Cartopy, with disputed regions marked.

    Parameters:
    - country_count_dict: dict, keys are two-letter country codes (ISO A2), values are counts.
    - disputed_geometries: dict, FIDs as keys, geometries as values for disputed regions.
    - save_path: str, optional, path to save the figure.
    - dpi: int, optional, resolution of the saved figure.
    - figsize: tuple, optional, size of the figure.
    - cmap_name: str, optional, colormap for the heatmap.
    - title: str, optional, title of the plot.
    - title_fontsize: int, optional, fontsize of the plot title.
    - country_edgecolor: str, edge color for country boundaries.
    - country_linewidth: float, line width for country boundaries.
    - country_default_color: tuple, default color for countries not in data.
    - colorbar_orientation: str, orientation of the colorbar ('horizontal' or 'vertical').
    - colorbar_pad: float, padding for the colorbar.
    - colorbar_shrink: float, shrink factor for the colorbar.
    - colorbar_label: str, label for the colorbar.
    - disputed_color: str or tuple, color for disputed regions.
    - disputed_linewidth: float, linewidth for disputed region boundaries.
    - disputed_linestyle: str, linestyle for disputed region boundaries.
    - disputed_edgecolor: str, edge color for disputed regions.
    - coastline_color: str, color for coastlines.
    - coastline_linewidth: float, linewidth for coastlines.
    - border_linestyle: str, linestyle for borders.
    - border_edgecolor: str, edge color for borders.
    - border_linewidth: float, linewidth for borders.
    """
    # Load country shapefiles
    shapename = 'admin_0_countries'
    countries_shp = shpreader.natural_earth(resolution='110m',
                                            category='cultural', name=shapename)

    # Prepare figure and axis
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(color=coastline_color, linewidth=coastline_linewidth)
    ax.add_feature(cfeature.BORDERS, linestyle=border_linestyle,
                   edgecolor=border_edgecolor, linewidth=border_linewidth)

    # Normalize counts and set colormap
    counts = list(country_count_dict.values())
    norm = LogNorm(vmin=max(min(counts), 1), vmax=max(counts))  # Avoid LogNorm errors with zero counts
    cmap = cm.get_cmap(cmap_name)

    # Plot country heatmap
    for country in shpreader.Reader(countries_shp).records():
        country_code = country.attributes.get('ISO_A2')  # Two-letter ISO code
        if country_code in country_count_dict:
            count = country_count_dict[country_code]
            color = cmap(norm(count))
        elif country_code == 'CN-TW':
            country_code = 'CN'
            count = country_count_dict[country_code]
            color = cmap(norm(count))
        else:
            country_code = country.attributes.get('FIPS_10')
            # print(country_code)
            if country_code in country_count_dict:
                count = country_count_dict[country_code]
                color = cmap(norm(count))
            else:
                country_code = country.attributes.get('ISO_A2_EH')  
                
                if country_code in country_count_dict:
                    count = country_count_dict[country_code]
                    color = cmap(norm(count))
                else:
                    color = country_default_color 

        ax.add_geometries(
            country.geometry, ccrs.PlateCarree(),
            facecolor=color, edgecolor=country_edgecolor, linewidth=country_linewidth
        )

    # Plot disputed regions
    if disputed_geometries is not None:
        for fid, geom in disputed_geometries.items():
            ax.add_geometries(
                [geom], ccrs.PlateCarree(),
                facecolor=disputed_color,
                edgecolor=disputed_edgecolor,
                linewidth=disputed_linewidth,
                linestyle=disputed_linestyle
            )

    # Add a colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation=colorbar_orientation,
                        pad=colorbar_pad, shrink=colorbar_shrink)
    cbar.set_label(colorbar_label)

    plt.title(title, fontsize=title_fontsize)
    if save_path:
        plt.savefig(save_path, dpi=dpi, transparent=True)
    plt.show()
    
def plot_yearly_publications(yearly_report, save_path=None, dpi=300, 
                             figsize=(10, 6), font_size=12,
                             start_year=None, end_year=None, x_tick_interval=1,
                             y_range_left=None, y_range_right=None, 
                             y_tick_interval_left=None, y_tick_interval_right=None,
                             bar_color='blue', bar_edge_color='black',
                             polyline_colors=None, markers=None,
                             legend=True, bar_width=0.8,
                             title="Yearly Publications and Country Percentages",
                             x_label="Year",
                             y_left_label="Total Publications",
                             y_right_label="Country Percentage"):
    """
    Creates a combined plot for total publications and country percentages.

    :param yearly_report: Dictionary containing yearly data (output from analyze_yearly_publications)
    :param save_path: File path to save the figure (optional)
    :param dpi: Resolution of the saved figure (default: 300)
    :param figsize: Tuple specifying the size of the figure (default: (10, 6))
    :param font_size: Font size for labels and titles
    :param start_year: Start year for the x-axis (default: None, auto-detect)
    :param end_year: End year for the x-axis (default: None, auto-detect)
    :param x_tick_interval: Interval for x-axis ticks (default: 1)
    :param y_range_left: Tuple specifying the y-axis range for the left axis (optional)
    :param y_range_right: Tuple specifying the y-axis range for the right axis (optional)
    :param y_tick_interval_left: Interval for y-axis ticks on the left axis (optional)
    :param y_tick_interval_right: Interval for y-axis ticks on the right axis (optional)
    :param bar_color: Color of the bars (default: 'blue')
    :param bar_edge_color: Color of the bar edges (default: 'black')
    :param polyline_colors: Dictionary specifying colors for each country's line (default: None)
    :param markers: Dictionary specifying markers for each country's line (default: None)
    :param legend: Whether to display the legend (default: True)
    :param bar_width: Width of the bars (default: 0.8)
    :param title: Title of the plot (default: "Yearly Publications and Country Percentages")
    :param x_label: Label for the x-axis (default: "Year")
    :param y_left_label: Label for the left y-axis (default: "Total Publications")
    :param y_right_label: Label for the right y-axis (default: "Country Percentage")
    """
    # Extract years and data
    years = list(map(int, yearly_report.keys()))
    total_publications = [yearly_report[str(year)]['total_publications'] for year in years]
    countries = next(iter(yearly_report.values()))['countries'].keys()  # Get country names
    country_percentages = {country: [yearly_report[str(year)]['countries'][country]['percentage'] 
                                     for year in years] for country in countries}

    # Filter data by start_year and end_year
    if start_year is not None:
        start_year = int(start_year)
        years = [year for year in years if year >= start_year]
    if end_year is not None:
        end_year = int(end_year)
        years = [year for year in years if year <= end_year]

    # Cut-off data for filtered years
    total_publications = [yearly_report[str(year)]['total_publications'] for year in years]
    country_percentages = {country: [yearly_report[str(year)]['countries'][country]['percentage']
                                     for year in years] for country in countries}

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Bar plot for total publications
    bars = ax1.bar(years, total_publications, color=bar_color, edgecolor=bar_edge_color, 
                   width=bar_width, label="Total Publications")
    ax1.set_ylabel(y_left_label, fontsize=font_size)
    ax1.set_xlabel(x_label, fontsize=font_size)
    ax1.tick_params(axis='both', labelsize=font_size, length=2.5, width=1, colors='black')

    # Set x-axis range and ticks
    ax1.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax1.set_xticks(range(min(years), max(years) + 1, x_tick_interval))

    # Apply y-range and ticks for the left axis if provided
    if y_range_left:
        ax1.set_ylim(y_range_left)
    if y_tick_interval_left:
        ax1.set_yticks(range(y_range_left[0], y_range_left[1] + 1, y_tick_interval_left))

    # Create a secondary axis for percentages
    ax2 = ax1.twinx()

    # Plot country percentages as polylines
    lines = []
    line_labels = []
    for country, percentages in country_percentages.items():
        color = polyline_colors.get(country, None) if polyline_colors else None
        marker = markers.get(country, 'o') if markers else 'o'
        line, = ax2.plot(years, percentages, label=f"{country.title()}", marker=marker, color=color, linewidth=2)
        lines.append(line)
        line_labels.append(f"{country.title()}")  # Use title case for legend labels

    # Add combined legend
    if legend:
        # Combine bar and line handles and labels
        bar_handle = bars[0]
        bar_label = "Total Publications"
        handles = [bar_handle] + lines
        labels = [bar_label] + line_labels
        ax1.legend(handles, labels, loc="best", fontsize=font_size)

    # Add combined legend
    #ax1.legend(handles, labels, loc="best", fontsize=font_size)
    

    ax2.set_ylabel(y_right_label, fontsize=font_size + 1)
    ax2.tick_params(axis='both', labelsize=font_size + 1, length=2.5, width=1, colors='black')

    # Apply y-range and ticks for the right axis if provided
    if y_range_right:
        ax2.set_ylim(y_range_right)
    if y_tick_interval_right:
        ax2.set_yticks(range(y_range_right[0], y_range_right[1] + 1, y_tick_interval_right))  

    # Add the title
    plt.title(title, fontsize=font_size + 2)

    # Adjust layout and save if needed
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    plt.show()
    
    
def draw_keyword_trends_from_subset(df, keyword_subset, year_column,
                                    save_path=None, dpi=300, figsize=(10, 6), font_size=12,tick_label_size=12,
                                    start_year=None, end_year=None, x_tick_interval=1,is_legend_sorted=True,
                                    y_range=None, y_tick_interval=None, polyline_colors=None, markers=None,
                                    marker_sizes=None,  # Added marker size parameter
                                    legend=True, title="Keyword Trends Over Time", x_label="Year", y_label="Publication Count"):
    """
    Draws a polyline plot for yearly publication trends based on a keyword subset.
    """
    # Ensure year column is integer, and take care of the NaN values
    df[year_column] = df[year_column].fillna(0)
    df[year_column] = df[year_column].astype(int) 

    # Initialize a dictionary to store yearly counts for each keyword
    unique_years = sorted(df[year_column].unique())
    yearly_counts = {keyword: {year: 0 for year in unique_years} for keyword in keyword_subset}

    # Count yearly occurrences for each keyword
    for keyword, indices in keyword_subset.items():
        subset = df.loc[indices]
        for year, count in subset[year_column].value_counts().items():
            yearly_counts[keyword][year] += count

    # Filter yearly counts based on start_year and end_year
    if start_year is not None:
        unique_years = [year for year in unique_years if year >= start_year]
    if end_year is not None:
        unique_years = [year for year in unique_years if year <= end_year]

    # Create a figure
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot polylines for each keyword
    for keyword, counts in yearly_counts.items():
        years = [year for year in unique_years if year in counts]
        counts_list = [counts[year] for year in years]
        color = polyline_colors.get(keyword, None) if polyline_colors else None
        marker = markers.get(keyword, 'o') if markers else 'o'
        size = marker_sizes.get(keyword, 4) if isinstance(marker_sizes, dict) else marker_sizes or 4  # Default size 6
        ax.plot(years, counts_list, label=keyword, marker=marker, markersize=size, linewidth=2, color=color)

    # Set x-axis and y-axis labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)

    # Set x-axis and y-axis ticks
    if x_tick_interval:
        x_ticks = range(min(unique_years), max(unique_years) + 1, x_tick_interval)
        ax.set_xticks(x_ticks)
    if y_range:
        ax.set_ylim(y_range)
    if y_tick_interval:
        y_ticks = range(y_range[0], y_range[1] + 1, y_tick_interval)
        ax.set_yticks(y_ticks)

    # Ensure x-axis uses plain numbers, not scientific notation
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    ax.tick_params(axis='x', which='both', direction='out', left=True, right=False,length=2.5, width=1, colors='black')
    ax.tick_params(axis='y', which='both', direction='out', top=False, bottom=True, length=2.5, width=1, colors='black')
    
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
    # Add title
    plt.title(title, fontsize=font_size + 2)

    # Add legend
    if legend:
        if is_legend_sorted:
            # Sort legend by keyword name
            handles, labels = ax.get_legend_handles_labels()
            sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
            handles = [handles[i] for i in sorted_indices]
            labels = [labels[i] for i in sorted_indices]
            ax.legend(handles, labels, loc="best", fontsize=font_size - 1)
        else:
            ax.legend(loc="best", fontsize=font_size - 1)

    # Remove only top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("black")
    ax.spines['bottom'].set_color("black")
    ax.grid(False)

    # Save the figure if a path is specified
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    plt.show()

def draw_horizontal_violin_with_means(
    df,
    keyword_subset,
    sentiment_col,
    save_path=None,
    dpi=300,
    figsize=(10, 6),
    palette="Set2",
    violin_width=0.8,
    mean_color="black",
    mean_marker="o",
    mean_size=8,
    title="Horizontal Violin Plot with Mean Points (Flipped y-Axis)",
    title_fontsize=16,
    xlabel="Sentiment",
    ylabel="Keywords",
    label_fontsize=12,
    violin_colors=None,
    value_range=(-0.5, 0.5),
    padding=0.1,
    keyword_sorted = True,
    is_cat_ticklabel_visible=True
):
    sns.set(style="whitegrid")

    # Sort keywords alphabetically in reverse order (so y-axis is flipped)
    if keyword_sorted:
        sorted_keywords = sorted(keyword_subset.keys(), reverse=True)
    else:
        sorted_keywords = list(keyword_subset.keys())

    # Collect subset data and compute mean sentiment values
    subset_data = []
    keyword_means = {}
    for keyword in sorted_keywords:
        indices = keyword_subset[keyword]
        subset_df = df.loc[indices]
        if not subset_df.empty:
            for val in subset_df[sentiment_col]:
                subset_data.append({"Keyword": keyword, "Sentiment": val})
            keyword_means[keyword] = subset_df[sentiment_col].mean()

    subset_df = pd.DataFrame(subset_data)
    if subset_df.empty:
        print("No data available for plotting.")
        return

    # Make Keyword a categorical variable for consistent order
    subset_df["Keyword"] = pd.Categorical(subset_df["Keyword"], categories=sorted_keywords, ordered=True)
    subset_df = subset_df.sort_values(by="Keyword")

    # Determine custom palette
    custom_palette = None
    if isinstance(violin_colors, dict):
        custom_palette = violin_colors
    elif isinstance(violin_colors, (list, tuple)):
        unique_keywords = subset_df["Keyword"].unique()
        if len(violin_colors) >= len(unique_keywords):
            custom_palette = dict(zip(unique_keywords, violin_colors))
        else:
            print("Warning: violin_colors list has fewer colors than keywords. Using default palette.")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        x="Sentiment",
        y="Keyword",
        data=subset_df,
        density_norm="width",
        width=violin_width,
        inner=None,
        orient="horizontal",
        ax=ax,
        palette=custom_palette if custom_palette else palette,
        hue=None,
        legend=False
    )

    # Add mean points
    for keyword, mean_val in keyword_means.items():
        ax.scatter(mean_val, keyword, color=mean_color, marker=mean_marker, s=mean_size, zorder=10)

    # Styling and layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    ax.set_xlim(value_range)
    ax.set_ylim(-0.5 - padding, len(sorted_keywords) - 0.5 + padding)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize, color="black")
    ax.set_ylabel(ylabel, fontsize=label_fontsize, color="black")

    if not is_cat_ticklabel_visible:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
    else:
        wrapped_labels = wrap_labels(subset_df["Keyword"].cat.categories)
        ax.set_yticklabels(wrapped_labels, fontsize=label_fontsize)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.show()
    
def draw_vertical_violin_with_means(
    df,
    keyword_subset,
    sentiment_col,
    save_path=None,
    dpi=300,
    figsize=(10, 6),
    palette="Set2",
    violin_width=0.8,
    mean_color="black",
    mean_marker="o",
    mean_size=8,
    title="Vertical Violin Plot with Mean Points",
    title_fontsize=16,
    xlabel="Keywords",
    ylabel="Sentiment",
    label_fontsize=12,
    violin_colors=None,
    value_range=(-0.5, 0.5),
    padding=0.1
):
    sns.set(style="whitegrid")

    # Sort keywords alphabetically
    sorted_keywords = sorted(keyword_subset.keys())

    # Prepare subset data
    subset_data = []
    keyword_means = {}
    for keyword in sorted_keywords:
        indices = keyword_subset[keyword]
        subset_df = df.loc[indices]
        if not subset_df.empty:
            for val in subset_df[sentiment_col]:
                subset_data.append({"Keyword": keyword, "Sentiment": val})
            # Calculate the mean sentiment for the keyword
            keyword_means[keyword] = subset_df[sentiment_col].mean()

    subset_df = pd.DataFrame(subset_data)

    if subset_df.empty:
        print("No data available for plotting.")
        return

    # Create the plot
    f, ax = plt.subplots(figsize=figsize)

    # Set custom violin colors if provided
    if violin_colors:
        unique_keywords = subset_df["Keyword"].unique()
        custom_palette = dict(zip(unique_keywords, violin_colors))
        sns.violinplot(
            x="Keyword",
            y="Sentiment",
            data=subset_df,
            density_norm="width",
            width=violin_width,
            inner=None,
            orient="vertical",
            ax=ax,
            palette=custom_palette,
            hue=None,
            legend=False
        )
    else:
        sns.violinplot(
            x="Keyword",
            y="Sentiment",
            data=subset_df,
            palette=palette,
            density_norm="width",
            width=violin_width,
            inner=None,
            orient="vertical",
            ax=ax,
            hue=None,
            legend=False
        )

    # Add mean points
    for keyword, mean_val in keyword_means.items():
        ax.scatter(keyword, mean_val, color=mean_color, marker=mean_marker, s=mean_size, zorder=10)

    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Set axis limits based on value_range and add padding
    ax.set_xlim(-0.5 - padding, len(keyword_subset) - 0.5 + padding)
    ax.set_ylim(value_range[0] - padding, value_range[1] + padding)

    # Customize labels and title
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    plt.tight_layout()
    ax.grid(True, linestyle="--", axis="y", linewidth=0.5)
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.show()


def plot_barplot(
    data, 
    title="Barplot", 
    xlabel="Keys", 
    ylabel="Count", 
    figsize=(10, 6), 
    palette="viridis", 
    bar_width=0.8, 
    bar_interval=0.2, 
    save_path=None, 
    dpi=300, 
    orientation="vertical", 
    tick_rotation=45, 
    bar_color=None, 
    edge_color="black", 
    edge_width=0.5, 
    font_size=12,
    xlim=None, 
    ylim=None, 
    abbreviate_labels=False, 
    abbre_dict=None,
    is_grid=False, 
    sort_by="count",          # NEW: "count" or "key"
    flip_order=False,         # NEW: reverse order
    is_cat_ticklabel_visible=True,
    tick_interval=None
):
    sns.set(style="whitegrid" if is_grid else "white")

    # Normalize keys (replace \& with &)
    if isinstance(data, dict):
        normalized_data = {key.replace(r"\&", "&"): value for key, value in data.items()}
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] != 2:
            raise ValueError("DataFrame must have exactly two columns.")
        data.iloc[:, 0] = data.iloc[:, 0].str.replace(r"\&", "&", regex=False)
        keys, counts = data.iloc[:, 0].tolist(), data.iloc[:, 1].tolist()
        normalized_data = dict(zip(keys, counts))
    else:
        raise TypeError("Input must be a dictionary or a DataFrame with two columns.")

    # Sort data by count or key
    if sort_by == "count":
        sorted_items = sorted(normalized_data.items(), key=lambda x: x[1], reverse=flip_order)
    elif sort_by == "key":
        sorted_items = sorted(normalized_data.items(), key=lambda x: x[0].lower(), reverse=flip_order)
    else:
        raise ValueError("sort_by must be either 'count' or 'key'")

    original_keys, counts = zip(*sorted_items)

    # Abbreviate labels if needed
    if abbreviate_labels:
        if abbre_dict:
            abbre_dict_lower = {k.replace(r"\&", "&").lower(): v for k, v in abbre_dict.items()}
            keys = [abbre_dict_lower.get(k.lower(), k.title() if " " not in k else k) for k in original_keys]
        else:
            keys = ["".join([word[0] for word in k.split()]).upper() for k in original_keys]
    else:
        keys = list(original_keys)

    fig, ax = plt.subplots(figsize=figsize)
    indices = range(len(keys))
    adjusted_positions = [i * (bar_width + bar_interval) for i in indices]

    # Resolve bar colors
    if isinstance(bar_color, dict):
        color_list = [bar_color.get(k, 'gray') for k in original_keys]
    elif isinstance(bar_color, (list, tuple)):
        if len(bar_color) < len(keys):
            print("Warning: bar_color list has fewer colors than keys. Some bars may repeat colors.")
        color_list = list(bar_color)[:len(keys)]
    else:
        color_list = sns.color_palette(palette, len(keys))

    # Plot bars
    if orientation == "vertical":
        bars = ax.bar(
            adjusted_positions, counts,
            color=color_list,
            edgecolor=edge_color,
            linewidth=edge_width,
            width=bar_width
        )
        ax.set_xticks(adjusted_positions)
        ax.set_xticklabels(keys, rotation=tick_rotation, fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size, color="black")
        ax.set_ylabel(ylabel, fontsize=font_size, color="black")
        if ylim:
            ax.set_ylim(ylim)
        if tick_interval:
            ax.set_yticks(range(0, int(max(counts)) + 1, tick_interval))
        ax.tick_params(axis="y", which="both", direction="out", left=True, right=False, length=2.5, width=1, colors="black")
    elif orientation == "horizontal":
        bars = ax.barh(
            adjusted_positions, counts,
            color=color_list,
            edgecolor=edge_color,
            linewidth=edge_width,
            height=bar_width
        )
        ax.set_yticks(adjusted_positions)
        wrapped_labels = wrap_labels(keys)
        ax.set_yticklabels(wrapped_labels, rotation=tick_rotation, fontsize=font_size)
        # ax.set_yticklabels(keys, rotation=tick_rotation, fontsize=font_size)
        ax.set_ylabel(xlabel, fontsize=font_size, color="black")
        ax.set_xlabel(ylabel, fontsize=font_size, color="black")
        if xlim:
            ax.set_xlim(xlim)
        if tick_interval:
            ax.set_xticks(range(0, int(max(counts)) + 1, tick_interval))
        ax.tick_params(axis="x", which="both", direction="out", bottom=True, top=False, length=2.5, width=1, colors="black")
    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'.")

    if is_grid:
        ax.grid(axis="y" if orientation == "vertical" else "x", linestyle="--", color="gray", linewidth=0.5)

    ax.set_title(title, fontsize=font_size + 2, color="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not is_cat_ticklabel_visible:
        if orientation == "vertical":
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", length=0)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", which="both", length=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.show()
    
def plot_poly_curve_from_dataframe(
    data, 
    title="Poly-Curve Plot", 
    xlabel="Year", 
    ylabel="Count", 
    figsize=(10, 6), 
    line_width=None, 
    marker_size=None, 
    marker_color=None, 
    line_color=None, 
    line_type=None, 
    palette="rainbow", 
    save_path=None, 
    dpi=300, 
    font_size=12,
    year_start=None, 
    year_end=None, 
    is_legend=True, 
    is_grid=True, 
    abbre_dict=None, 
    abbreviate_labels=False,
    xlim=None,
    ylim=None,
    tick_interval=5
):
    # Reset styles
    plt.style.use('default')

    # Convert the index to integers if they are strings
    if data.index.dtype == object:
        data.index = data.index.astype(int)

    # Filter data by year range
    if year_start is not None or year_end is not None:
        if year_start is None:
            year_start = data.index.min()
        if year_end is None:
            year_end = data.index.max()
        data = data.loc[year_start:year_end]

    # Abbreviation handling
    keys = [key.replace(r"\&", "&") for key in data.columns]
    if abbreviate_labels:
        if abbre_dict:
            abbre_dict_lower = {k.replace(r"\&", "&").lower(): v for k, v in abbre_dict.items()}
            keys = [
                abbre_dict_lower.get(key.lower(), key.title() if " " not in key else key)
                for key in keys
            ]
        else:
            keys = ["".join([word[0] for word in key.split()]).upper() for key in keys]
        data.columns = keys

    # Get list of years and journals
    years = data.index
    journals = data.columns
    colors = sns.color_palette(palette, len(journals)) if line_color is None else None

    fig, ax = plt.subplots(figsize=figsize)

    # Order the journals by the start year of non-zero values
    first_non_zero_year = {}
    for journal in journals:
        journal_data = data[journal]
        non_zero_years = journal_data[journal_data > 0]
        if non_zero_years.empty:
            first_non_zero_year[journal] = float("inf")  # Place journals with no data at the end
        else:
            first_non_zero_year[journal] = non_zero_years.index[0]
    journals = sorted(journals, key=lambda x: first_non_zero_year[x])

    # Plot each journal
    for i, journal in enumerate(journals):
        journal_data = data[journal]

        # Start plotting from the first non-zero year
        non_zero_years = journal_data[journal_data > 0]
        if non_zero_years.empty:
            continue
        journal_data = journal_data.loc[non_zero_years.index[0]:]

        # Get line and marker properties
        lw = line_width[journal] if isinstance(line_width, dict) else line_width or 1.5
        ms = marker_size[journal] if isinstance(marker_size, dict) else marker_size or 6
        mc = marker_color[journal] if isinstance(marker_color, dict) else marker_color or None
        lc = line_color[journal] if isinstance(line_color, dict) else (colors[i] if colors else "black")
        lt = line_type[journal] if isinstance(line_type, dict) else line_type or "-"
        
        # Ensure sorted order for plotting
        journal_data = journal_data.sort_index()
        ax.plot(
            journal_data.index, journal_data.values, 
            label=journal, 
            linewidth=lw, 
            linestyle=lt,
            marker='o', 
            markersize=ms, 
            markerfacecolor=mc, 
            color=lc
        )

    # Customize plot
    ax.set_title(title, fontsize=font_size + 2)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    
    # Explicitly enable ticks
    ax.tick_params(axis='x', which='both', bottom=True, top=False, direction='out', length=2.5, width=1, colors="black")
    ax.tick_params(axis='y', which='both', left=True, right=False, direction='out', length=2.5, width=1, colors="black")
    
    # Ensure plain numbers on x-axis
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    # Set x-axis ticks with proper intervals
    year_range = range(data.index.min(), data.index.max() + 1, tick_interval)
    ax.set_xticks(year_range)
    ax.set_xticklabels([str(year) for year in year_range], fontsize=font_size-1)
    # also, set y-axis tick font size
    ax.set_yticklabels([f"{int(yi):d}" for yi in ax.get_yticks()], fontsize=font_size-1)

    # Make spines visible
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)

    # Set limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Set gridlines
    if is_grid:
        ax.grid(visible=True, which="major", color="gray", linestyle="--", linewidth=0.5)
    else:
        ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Display legend if enabled
    if is_legend:
        ax.legend(title="Journals", loc="best")
    plt.tight_layout()

    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.show()

def draw_keyword_relationship_diagram(
    pair_count,
    keyword_info,
    save_path=None,
    figsize=(18, 18),
    category_distance=10,  # Distance between category centers
    node_base_size=5,  # Base node size
    edge_base_width=1,  # Base edge width
    font_size=16,
    colormap="tab10",
    edge_alpha=0.5,
    log_scale_nodes=False,  # Whether to use log scaling for node sizes
    log_scale_edges=True,  # Whether to use log scaling for edge widths
    text_offset=(0, 0),  # Offset for node labels (closer to nodes)
    title="Keyword Relationship Diagram",
    title_fontsize=16,
    legend_title="",
    legend_fontsize=12,
    legend_position="best",
    edge_color="gray",
    node_edge_color="#a6cee3",
    node_alpha=0.8,
    background_color="white",
    axis_margin=0.5,  # Extra margin around the nodes
    dpi=300
):
    """
    Draws a keyword relationship diagram with categories centered around preassigned positions.

    Parameters:
    - pair_count: dict, e.g., {("keyword1", "keyword2"): count}.
    - keyword_info: dict, e.g., {"keyword1": {"count": 5, "category": "Water"}}.
    - save_path: Optional path to save the plot.
    - figsize: Tuple specifying figure size.
    - category_distance: Distance between category centers.
    - node_base_size: Base size for nodes.
    - edge_base_width: Base width for edges.
    - font_size: Font size for node labels.
    - colormap: Matplotlib colormap for categories, or a color dictionary for categories.
    - edge_alpha: Transparency level for edges.
    - log_scale_nodes: Whether to use log scaling for node sizes.
    - log_scale_edges: Whether to use log scaling for edge widths.
    - text_offset: Tuple specifying the offset for node labels (x, y).
    - title: Title for the plot.
    - title_fontsize: Font size for the title.
    - legend_title: Title for the legend.
    - legend_fontsize: Font size for the legend.
    - legend_position: Position of the legend (e.g., "best", "upper right").
    - edge_color: Color of the edges.
    - node_edge_color: Border color of the nodes.
    - node_alpha: Transparency for the nodes.
    - background_color: Background color of the plot.
    - axis_margin: Margin around the plot to avoid nodes near the edges.
    """
    # Initialize figure
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_facecolor(background_color)

    # Extract categories and assign cluster centers
    categories = list(set(info["category"] for info in keyword_info.values()))
    num_categories = len(categories)
    theta = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)
    category_centers = {cat: (category_distance * np.cos(t), category_distance * np.sin(t)) for cat, t in zip(categories, theta)}

    # Generate category colors
    if isinstance(colormap, str):
        cmap = plt.cm.get_cmap(colormap, num_categories)
        category_colors = {cat: cmap(i / num_categories) for i, cat in enumerate(categories)}
    else:
        category_colors = colormap
    # Assign node positions around category centers
    node_positions = {}
    for keyword, info in keyword_info.items():
        category_center = category_centers[info["category"]]
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, 2)  # Randomly distributed around the center
        node_positions[keyword] = (
            category_center[0] + radius * np.cos(angle),
            category_center[1] + radius * np.sin(angle),
        )

    # Determine axis limits with margin
    all_x = [pos[0] for pos in node_positions.values()]
    all_y = [pos[1] for pos in node_positions.values()]
    x_min, x_max = min(all_x) - axis_margin, max(all_x) + axis_margin
    y_min, y_max = min(all_y) - axis_margin, max(all_y) + axis_margin

    # Draw edges (arcs between nodes)
    for (kw1, kw2), count in pair_count.items():
        if kw1 in node_positions and kw2 in node_positions:
            x1, y1 = node_positions[kw1]
            x2, y2 = node_positions[kw2]
            arc_width = edge_base_width * (np.log10(count + 1) if log_scale_edges else count)
            ax.add_patch(
                FancyArrowPatch(
                    (x1, y1), (x2, y2),
                    color=edge_color,
                    linewidth=arc_width,
                    alpha=edge_alpha,
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0.2",  # Adjust curvature here
                )
            )

    # Draw nodes
    for keyword, pos in node_positions.items():
        size = node_base_size * (log10(keyword_info[keyword]["count"] + 1) if log_scale_nodes else keyword_info[keyword]["count"])
        color = category_colors[keyword_info[keyword]["category"]]
        ax.scatter(
            pos[0], pos[1],
            s=size,
            c=[color],
            alpha=node_alpha,
            edgecolors=node_edge_color,
            linewidths=1,
        )

    # Add node labels with closer offset
    for keyword, pos in node_positions.items():
        ax.text(
            pos[0] + text_offset[0],  # Apply x-offset
            pos[1] + text_offset[1],  # Apply y-offset
            keyword,
            fontsize=font_size,
            ha="left",
            va="center",
        )

    # Add legend
    for category, color in category_colors.items():
        ax.scatter([], [], c=[color], s=100, label=category)
    if legend_title:
        plt.legend(title=legend_title, scatterpoints=1, frameon=False, loc=legend_position, fontsize=legend_fontsize)

    # Customize plot
    plt.axis("off")
    plt.title(title, fontsize=title_fontsize)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.show()
    
def draw_horizontal_box_plot_with_means(
    df,
    keyword_subset,
    sentiment_col,
    save_path=None,
    dpi=300,
    figsize=(10, 6),
    palette="Set2",
    mean_color="red",
    mean_marker="o",
    mean_size=8,
    title="Horizontal Box Plot with Mean Points",
    title_fontsize=14,
    xlabel="Sentiment",
    ylabel="Keywords",
    label_fontsize=12,
    box_width=0.5,
    whisker_width=1.0,
    value_range=(-0.5, 0.5),
    is_cat_ticklabel_visible=True,
    is_median=True,            # If True, plot medians; else, plot means
    flip_order=True,           # NEW: Control alphabetical flip of keyword order
    box_color=None
):
    sns.set(style="whitegrid")

    # Prepare subset data
    subset_data = []
    for keyword, indices in keyword_subset.items():
        subset_df = df.loc[indices]
        if not subset_df.empty:
            for val in subset_df[sentiment_col]:
                subset_data.append({"Keyword": keyword, "Sentiment": val})

    subset_df = pd.DataFrame(subset_data)
    if subset_df.empty:
        print("No data available for plotting.")
        return

    # Set consistent keyword order based on flip_order
    sorted_keywords = sorted(keyword_subset.keys(), reverse=flip_order)
    subset_df["Keyword"] = pd.Categorical(subset_df["Keyword"], categories=sorted_keywords, ordered=True)

    # Resolve box colors
    unique_keywords = subset_df["Keyword"].cat.categories
    if isinstance(box_color, dict):
        box_palette = [box_color.get(k, "gray") for k in unique_keywords]
    elif isinstance(box_color, (list, tuple)):
        box_palette = list(box_color[:len(unique_keywords)])
    else:
        box_palette = sns.color_palette(palette, len(unique_keywords))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        x="Sentiment",
        y="Keyword",
        data=subset_df,
        palette=box_palette,
        width=box_width,
        orient="horizontal",
        ax=ax,
        linewidth=whisker_width,
        showfliers=False,
        boxprops={"edgecolor": "black", "linewidth": whisker_width}
    )

    # Add central tendency points
    group = subset_df.groupby("Keyword")["Sentiment"]
    values = group.median() if is_median else group.mean()
    for keyword, val in values.items():
        ax.scatter(val, keyword, color=mean_color, marker=mean_marker, s=mean_size, zorder=10)

    # Style adjustments
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    ax.set_xlim(value_range)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize, color="black")
    ax.set_ylabel(ylabel, fontsize=label_fontsize, color="black")
    
    if not is_cat_ticklabel_visible:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
    else:
        wrapped_labels = wrap_labels(subset_df["Keyword"].cat.categories)
        ax.set_yticklabels(wrapped_labels, fontsize=label_fontsize)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.show()

def draw_vertical_box_plot_with_means(
    df,
    keyword_subset,
    sentiment_col,
    save_path=None,
    dpi=300,
    figsize=(10, 6),
    palette="Set2",
    mean_color="red",
    mean_marker="o",
    mean_size=8,
    title="Vertical Box Plot with Mean Points",
    title_fontsize=14,
    xlabel="Keywords",
    ylabel="Sentiment",
    label_fontsize=12,
    box_width=0.5,
    whisker_width=1.0,
    value_range=(-0.5, 0.5),
    box_color=None,
):
    sns.set(style="whitegrid")

    # Prepare subset data
    subset_data = []
    for keyword, indices in keyword_subset.items():
        subset_df = df.loc[indices]
        if not subset_df.empty:
            for val in subset_df[sentiment_col]:
                subset_data.append({"Keyword": keyword, "Sentiment": val})
    subset_df = pd.DataFrame(subset_data)
    subset_df = subset_df.sort_values(by="Keyword", ascending=True)
    if subset_df.empty:
        print("No data available for plotting.")
        return

    # Handle box color
    unique_keywords = subset_df["Keyword"].unique()
    if box_color is None:
        box_color = sns.color_palette(palette, len(unique_keywords))
        box_color = {keyword: box_color[i] for i, keyword in enumerate(unique_keywords)}
    elif isinstance(box_color, dict):
        box_color = {k: v for k, v in box_color.items()}
    else:
        box_color = {keyword: box_color for keyword in unique_keywords}

    # Create the plot
    f, ax = plt.subplots(figsize=figsize)

    # Draw the box plot
    sns.boxplot(
        x="Keyword",
        y="Sentiment",
        data=subset_df,
        palette=[box_color.get(keyword, "none") for keyword in unique_keywords],
        width=box_width,
        orient="vertical",
        ax=ax,
        linewidth=whisker_width,
        showfliers=False,
        boxprops={"edgecolor": "black", "linewidth": whisker_width},
    )

    # Add mean points
    means = subset_df.groupby("Keyword")["Sentiment"].mean()
    for keyword, mean_val in means.items():
        ax.scatter(keyword, mean_val, color=mean_color, marker=mean_marker, s=mean_size, zorder=10)

    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)  # Horizontal grid lines only

    # Set axis limits and labels
    ax.set_ylim(value_range)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize, color="black")
    ax.set_ylabel(ylabel, fontsize=label_fontsize, color="black")

    plt.tight_layout()

    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.show()


def draw_pie_plot(
    data_dict,
    top_n=None,
    figsize=(8, 6),
    title="Pie Chart",
    title_fontsize=16,
    legend_fontsize=12,
    label_fontsize=12,
    is_legend=True,
    colors=None,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=None,
    textprops=None,
    save_path=None,
    dpi=300,
    is_annotate=True  # Whether to display percentages on the pie
):
    """
    Draw a pie plot from a dictionary with customizable options.

    Parameters:
    - data_dict: The input dictionary, where keys represent labels and values can be:
      - List: The length of the list will be used as the value.
      - Dict: A sub-key 'count' will be used as the value.
      - Single value: The value will be used directly.
    - top_n: Number of top entries to keep in the plot. Others will be merged as 'Others'.
    - figsize: Size of the figure.
    - title: Title of the plot.
    - title_fontsize: Font size for the title.
    - legend_fontsize: Font size for the legend.
    - label_fontsize: Font size for the labels.
    - is_legend: Whether to display a legend.
    - colors: Colors for the pie segments.
    - autopct: Format for displaying percentages.
    - startangle: Starting angle for the pie chart.
    - wedgeprops: Properties for the wedge (e.g., edge color, width).
    - textprops: Text properties for the labels.
    - save_path: Path to save the plot (if specified).
    - dpi: Resolution for saving the plot.
    - is_annotate: Whether to annotate percentages on the pie chart.
    """
    # Process the data
    processed_data = {}
    for key, value in data_dict.items():
        if isinstance(value, list):
            processed_data[key] = len(value)
        elif isinstance(value, dict):
            processed_data[key] = value.get('count', 0)
        else:
            processed_data[key] = value

    # Sort the data by value in descending order
    sorted_data = dict(sorted(processed_data.items(), key=lambda x: x[1], reverse=True))

    # Keep the top N entries and merge others
    if top_n is not None:
        sorted_keys = list(sorted_data.keys())
        sorted_values = list(sorted_data.values())
        top_keys = sorted_keys[:top_n]
        top_values = sorted_values[:top_n]
        other_value = sum(sorted_values[top_n:])
        if other_value > 0:
            top_keys.append("Others")
            top_values.append(other_value)
        sorted_data = dict(zip(top_keys, top_values))

    # Prepare data for plotting
    labels = list(sorted_data.keys())
    sizes = list(sorted_data.values())

    # Create the pie chart
    fig, ax = plt.subplots(figsize=figsize)

    if is_annotate:
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels if not is_legend else None,  # Hide labels if legend is enabled
            autopct=autopct,  # Annotate if enabled
            startangle=startangle,
            colors=colors,
            wedgeprops=wedgeprops,
            textprops=textprops
        )
        for autotext in autotexts:
            autotext.set_fontsize(label_fontsize)
    else:
        wedges, texts = ax.pie(
            sizes,
            labels=labels if not is_legend else None,  # Hide labels if legend is enabled
            startangle=startangle,
            colors=colors,
            wedgeprops=wedgeprops,
            textprops=textprops
        )

    # Customize title
    ax.set_title(title, fontsize=title_fontsize)

    # Add legend if enabled
    if is_legend:
        ax.legend(
            wedges,
            labels,
            title="Categories",
            loc="center left",
            bbox_to_anchor=(1, 0.5),  # Move legend outside the pie
            fontsize=legend_fontsize
        )

    # Adjust font sizes for pie labels
    if not is_legend:
        for text in texts:
            text.set_fontsize(label_fontsize)

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)

    # Show the plot
    plt.tight_layout()
    plt.show()
    
def draw_nested_pie_chart_with_regions(
    data_dict,
    top_n=3,
    threshold=15,
    figsize=(12, 10),
    title="Nested Regional Distribution",
    title_fontsize=16,
    legend_fontsize=12,
    label_fontsize=10,
    is_legend=True,
    region_colors=None,
    edge_color="black",
    startangle=90,
    wedgeprops=None,
    save_path=None,
    dpi=300
):
    # Predefined continent mapping
    continent_mapping = {
        "Asia": ["AF", "AM", "AZ", "BH", "BD", "BT", "BN", "KH", "CN", "CY", "GE", "IN", "ID", "IR", "IQ", "IL", "JP",
                 "JO", "KZ", "KW", "KG", "LA", "LB", "MY", "MV", "MN", "MM", "NP", "KP", "OM", "PK", "PS", "PH", "QA",
                 "SA", "SG", "KR", "LK", "SY", "TJ", "TH", "TL", "TM", "AE", "UZ", "VN", "YE"],
        "Africa": ["DZ", "AO", "BJ", "BW", "BF", "BI", "CM", "CV", "CF", "TD", "KM", "CG", "CD", "DJ", "EG", "GQ", "ER",
                   "ET", "GA", "GM", "GH", "GN", "GW", "CI", "KE", "LS", "LR", "LY", "MG", "MW", "ML", "MR", "MU", "MA",
                   "MZ", "NA", "NE", "NG", "RW", "ST", "SN", "SC", "SL", "SO", "ZA", "SS", "SD", "SZ", "TZ", "TG", "TN",
                   "UG", "ZM", "ZW"],
        "Europe": ["AL", "AD", "AT", "BY", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU",
                   "IS", "IE", "IT", "XK", "LV", "LI", "LT", "LU", "MT", "MC", "ME", "NL", "MK", "NO", "PL", "PT", "RO",
                   "RU", "SM", "RS", "SK", "SI", "ES", "SE", "CH", "TR", "UA", "GB", "VA"],
        "North America": ["AG", "BS", "BB", "BZ", "CA", "CR", "CU", "DM", "DO", "SV", "GD", "GT", "HT", "HN", "JM",
                          "MX", "NI", "PA", "KN", "LC", "VC", "TT", "US"],
        "South America": ["AR", "BO", "BR", "CL", "CO", "EC", "GY", "PY", "PE", "SR", "UY", "VE"],
        "Oceania": ["AU", "FJ", "KI", "MH", "FM", "NR", "NZ", "PW", "PG", "WS", "SB", "TO", "TV", "VU"],
        "Antarctica": ["AQ"]
    }

    # Assign default colors if not provided
    if region_colors is None:
        region_colors = {
            "Asia": "#FF9999",
            "Africa": "#66B2FF",
            "Europe": "#99FF99",
            "North America": "#FFCC99",
            "South America": "#FF9966",
            "Oceania": "#99CCFF",
            "Antarctica": "#CCCCCC"
        }

    # Aggregate data by region
    region_data = {region: 0 for region in continent_mapping.keys()}
    country_to_region = {}

    for country_code, value in data_dict.items():
        for region, codes in continent_mapping.items():
            if country_code in codes:
                region_data[region] += value
                country_to_region[country_code] = region
                break

    # Prepare data for the pie chart
    plot_data = []
    plot_labels = []
    plot_colors = []

    for region, region_value in region_data.items():
        if (region_value / sum(region_data.values())) * 100 > threshold:
            countries_in_region = [
                (country, data_dict[country])
                for country, reg in country_to_region.items()
                if reg == region
            ]
            countries_in_region.sort(key=lambda x: x[1], reverse=True)
            top_countries = countries_in_region[:top_n]
            others_value = region_value - sum([val for _, val in top_countries])

            # Add top-N countries and others
            for country, val in top_countries:
                plot_data.append(val)
                plot_labels.append(country)
                plot_colors.append(region_colors[region])

            plot_data.append(others_value)
            plot_labels.append("")  # No label for "Others"
            plot_colors.append(region_colors[region] + "CC")  # Add transparency to "Others"
        else:
            # Add region as a whole
            plot_data.append(region_value)
            plot_labels.append("")  # No label for whole regions
            plot_colors.append(region_colors[region])

    # Draw the pie chart
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts = ax.pie(
        plot_data,
        labels=None,  # Hide labels on the pie
        colors=plot_colors,
        startangle=startangle,
        wedgeprops=wedgeprops,
    )

    # Annotate top countries
    for i, wedge in enumerate(wedges):
        if plot_labels[i]:
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = np.cos(np.radians(angle))
            y = np.sin(np.radians(angle))
            ax.text(
                1.1 * x, 1.1 * y,
                plot_labels[i],
                fontsize=label_fontsize,
                ha="center",
                va="center"
            )

    # Add legend for regions
    if is_legend:
        legend_elements = [
            plt.Line2D([0], [0], color=color, lw=10, label=region)
            for region, color in region_colors.items()
        ]
        ax.legend(handles=legend_elements, title="Regions", loc="center left", bbox_to_anchor=(1, 0.5),
                  fontsize=legend_fontsize)

    # Add title
    ax.set_title(title, fontsize=title_fontsize)

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)

    plt.tight_layout()
    plt.show()

def plot_word_clusters(
    reduced_vectors, filtered_words, word_counts, kmeans, num_clusters,
    figsize=(12, 10), title="Word Clusters for Provided Keywords (Dot Size by Frequency)",
    xlabel="PCA Dimension 1", ylabel="PCA Dimension 2", font_size=10,
    title_font_size=16, label_font_size=12, dot_alpha=0.7,
    x_range=None, y_range=None, dpi=300, save_path=None, grid=False, seed=0,
    cluster_colors=None  # Add customizable cluster colors
):
    """
    Plots word clusters with keywords, colored by clusters, and dot size by frequency.

    Parameters:
    - reduced_vectors: 2D array-like, coordinates of words in reduced space.
    - filtered_words: List of words to plot.
    - word_counts: Dictionary of word frequencies.
    - kmeans: Fitted KMeans model for cluster assignments.
    - num_clusters: Number of clusters from the KMeans model.
    - figsize: Tuple specifying the figure size (default: (12, 10)).
    - title: Title of the plot (default: "Word Clusters for Provided Keywords").
    - xlabel, ylabel: Labels for the x and y axes (default: "PCA Dimension 1", "PCA Dimension 2").
    - font_size: Font size for keywords (default: 10).
    - title_font_size: Font size for the title (default: 16).
    - label_font_size: Font size for axis labels (default: 12).
    - dot_alpha: Transparency of the dots (default: 0.7).
    - x_range, y_range: Range for the x and y axes (default: None, auto-detected).
    - dpi: Resolution for saving the figure (default: 300).
    - save_path: File path to save the figure (default: None, does not save).
    - grid: Whether to show gridlines (default: False).
    - seed: Random seed for reproducible cluster colors (default: 0).
    - cluster_colors: List of custom colors for clusters (default: None, generates random colors).
    """
    # Set random seed for reproducible colors if not provided
    if cluster_colors is None:
        np.random.seed(seed)
        cluster_colors = np.random.rand(num_clusters, 3)  # Generate random colors for clusters
    
    filtered_words = [word for word in filtered_words if word in word_counts]
    # Compute dot sizes based on log-scaled frequencies
    dot_sizes = [np.log(word_counts[word] + 1) * 100 for word in filtered_words]  # Add 1 to avoid log(0)
    
    # Create the plot
    plt.figure(figsize=figsize)
    for i, word in enumerate(filtered_words):
        x, y = reduced_vectors[i][0], reduced_vectors[i][1]
        cluster_color = cluster_colors[kmeans.labels_[i] % len(cluster_colors)]  # Assign color based on cluster
        plt.scatter(x, y, color=cluster_color, s=dot_sizes[i], alpha=dot_alpha)
        plt.text(x, y + 0.002, word, fontsize=font_size, color="black", ha='center')  # Place keyword on top of dot

    # Set plot title and labels
    plt.title(title, fontsize=title_font_size)
    plt.xlabel(xlabel, fontsize=label_font_size)
    plt.ylabel(ylabel, fontsize=label_font_size)

    # Apply axis ranges if specified
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)

    # Show or hide gridlines
    plt.grid(grid)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

def draw_multi_polyline_curve(
    data_dict,
    figsize=(12, 8),
    dpi=600,
    title="Multi-Polyline Curve Plot",
    xlabel="",
    ylabel="Count",
    save_path=None,
    x_tick_rotation=0,  # Rotation for x-axis ticks
    tick_label_fontsize=12,  # Font size for tick labels
    axis_label_fontsize=12,  # Font size for axis labels
    legend_fontsize=11,  # Font size for legend
    title_fontsize=16,  # Font size for title
    y_value_range=None,  # Y-axis value range, defaults to None
    line_colors=None,  # Line colors for each country, defaults to None
    is_legend=True  # Show legend or not
):
    """
    Draws a multi-polyline curve plot for the given data dictionary.

    Parameters:
    - data_dict: Dictionary where the key is a tuple representing the year range (e.g., (1990, 1994)),
                 and the value is a sub-dictionary where keys are two-letter country abbreviations
                 and values are counts for the year range.
    - figsize: Tuple specifying figure size.
    - dpi: Resolution of the plot.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - save_path: Optional path to save the plot.
    - x_tick_rotation: Rotation angle for x-axis tick labels.
    - tick_label_fontsize: Font size for tick labels.
    - axis_label_fontsize: Font size for axis labels.
    - legend_fontsize: Font size for legend.
    - title_fontsize: Font size for the title.
    - y_value_range: Tuple specifying the y-axis range (min, max). Defaults to None for auto-scaling.
    - line_colors: Dictionary mapping country codes to line colors. Defaults to None for automatic coloring.
    - is_legend: Boolean indicating whether to display the legend.
    """

    # Extract and sort year ranges
    year_ranges = sorted(data_dict.keys(), key=lambda x: x[0])  # Sort by the start year
    countries = sorted(set(country for sub_dict in data_dict.values() for country in sub_dict.keys()))  # Sorted alphabetically

    # Format year ranges for x-axis labels
    if figsize[0] <= 6:
        x_labels = [f"{str(start)[-2:]}-{str(end)[-2:]}" for start, end in year_ranges]
    else:
        x_labels = [f"{start}-{end}" for start, end in year_ranges]

    # Initialize plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Prepare data for plotting
    for country in countries:
        counts = [data_dict[year_range].get(country, 0) for year_range in year_ranges]
        color = line_colors[country] if line_colors and country in line_colors else None
        ax.plot(
            x_labels, counts, label=country, marker='o', linestyle='-', linewidth=2, color=color
        )

    # Customize the plot
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize, color="black")
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize, color="black")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=tick_label_fontsize, rotation=x_tick_rotation, color="black")
    # also set y-axis tick font size
    ax.tick_params(axis='y', labelsize=tick_label_fontsize)
    ax.tick_params(axis='both', which='both', direction='out', top = False, left=True, \
        bottom = True, right = False, length=2.5, width=1, colors='black')

    # Add legend if enabled
    if is_legend:
        handles, labels = ax.get_legend_handles_labels()
        sorted_legend = sorted(zip(labels, handles), key=lambda x: x[0])  # Sort legend alphabetically
        sorted_labels, sorted_handles = zip(*sorted_legend)
        ax.legend(
            sorted_handles,
            sorted_labels,
            title="Countries",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize + 2,
            loc='upper left',
            bbox_to_anchor=(1, 1),
        )

    # Set y-axis value range if provided
    if y_value_range is not None:
        ax.set_ylim(y_value_range)

    # Make top and right axis invisible; left and bottom axes black
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

def plot_yearly_trends_3d(
    df, subsets, year_col, value_col=None, start_year=None, end_year=None, 
    xlabel="Year", ylabel="Groups (Offset for Separation)", zlabel=None, title="Yearly Trends of Different Groups (3D View)", 
    label_font_size=12, tick_font_size=10, title_font_size=14, line_width=1.2, zlim=(0, 100),
    cmap="viridis", view_elev=30, view_azim=225, save_path=None, dpi=300, figsize=(10, 8)
):
    """
    Plots multiple parallel curves in a 3D view with filled areas.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least a year column.
    - subsets (dict): Dictionary where keys are group names and values are lists of row indices.
    - year_col (str): Column name representing the year.
    - value_col (str or None): Column name representing values to analyze, or None to use counts.
    - start_year (int or None): Start year for filtering.
    - end_year (int or None): End year for filtering.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - zlabel (str or None): Label for the Z-axis. Defaults to "Average Value" or "Count".
    - title (str): Plot title.
    - label_font_size (int): Font size for axis labels.
    - tick_font_size (int): Font size for axis ticks.
    - title_font_size (int): Font size for the title.
    - line_width (float): Width of the plot lines.
    - cmap (str): Color ramp for the curves (any valid matplotlib colormap).
    - view_elev (int): Elevation angle for the 3D plot.
    - view_azim (int): Azimuth angle for the 3D plot.
    - save_path (str or None): Path to save the figure. If None, it is not saved.
    - dpi (int): Resolution for saving the figure.
    """
    # Compute yearly trends for each subset
    yearly_trends = {}
    for key, indices in subsets.items():
        subset_df = df.loc[indices]
        
        if value_col:
            yearly_trend = subset_df.groupby(year_col)[value_col].mean()  # Compute mean value per year
        else:
            yearly_trend = subset_df.groupby(year_col).size()  # Count occurrences per year
        
        yearly_trends[key] = yearly_trend

    # Convert trends into a DataFrame, filling missing years with 0
    trend_df = pd.DataFrame(yearly_trends).fillna(0)

    # Ensure index (years) is numeric and sorted
    trend_df = trend_df.sort_index()
    years = trend_df.index.to_numpy()

    # Apply year filtering
    if start_year is not None:
        trend_df = trend_df[trend_df.index >= start_year]
    if end_year is not None:
        trend_df = trend_df[trend_df.index <= end_year]
    
    # Update years after filtering
    years = trend_df.index.to_numpy()
    
    # Generate distinct colors for groups
    norm = mcolors.Normalize(vmin=0, vmax=len(trend_df.columns))
    cmap = plt.get_cmap(cmap)  # Convert colormap name to colormap object

    # Create the 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Set a baseline (y-offset) for separating curves
    y_offset = 0
    y_gap = np.max(trend_df.values) * 1.5  # Adjust spacing dynamically

    # Plot each subset's trend in 3D
    for i, key in enumerate(trend_df.columns):
        values = trend_df[key].values
        color = cmap(norm(i))

        # Define X, Y, and Z values
        x_vals = years
        y_vals = np.full_like(years, y_offset)  # Y-axis offset to separate curves
        z_vals = values

        # Plot the 3D curve
        ax.plot(x_vals, y_vals, z_vals, color=color, lw=line_width, label=key)

        # Plot projections onto XY plane
        ax.plot(x_vals, y_vals, np.zeros_like(z_vals), color=color, lw=line_width, linestyle='dotted')

        # Fill between the curve and XY plane
        verts = [(x, y_offset, z) for x, z in zip(x_vals, z_vals)]  # Curve points
        verts += [(x, y_offset, 0) for x in x_vals[::-1]]  # Base points (XY plane)

        poly = Poly3DCollection([verts], color=color, alpha=0.3)  # Semi-transparent fill
        ax.add_collection3d(poly)

        # Increase y_offset for next curve
        y_offset += y_gap

    # Labels and formatting
    ax.set_xlabel(xlabel, fontsize=label_font_size)
    ax.set_ylabel(ylabel, fontsize=label_font_size)
    ax.set_zlabel(zlabel if zlabel else ("Average Value" if value_col else "Count"), fontsize=label_font_size)
    ax.set_title(title, fontsize=title_font_size)

    # Set y-ticks to group names at correct offsets
    ax.set_yticks(np.arange(0, y_offset, y_gap))
    ax.set_yticklabels(trend_df.columns, fontsize=tick_font_size)

    # Set tick font sizes
    ax.tick_params(axis='both', labelsize=tick_font_size)
    
    if zlim:
        # set the zlim
        ax.set_zlim(zlim)
        
    # Adjust view angle
    ax.view_init(elev=view_elev, azim=view_azim)

    # Show legend
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # Save plot if required
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    
def wrap_label(text, max_len=20):
    return '\n'.join([text[i:i+max_len] for i in range(0, len(text), max_len)])
def plot_treemap_top_n(data_dict, top_n=8, title="Treemap", cmap=plt.cm.Set3, save_path=None):
    """
    Plot a treemap from a dictionary, keeping only the top N entries.
    Others are merged into an 'Other' category.
    
    Parameters:
        data_dict (dict): Dictionary of {label: value}
        top_n (int): Number of top items to show individually
        title (str): Title of the plot
        cmap (matplotlib colormap): Colormap for the rectangles
        save_path (str or None): If provided, saves the figure to this path
    """
    # Sort and split
    sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_n]
    other_items = sorted_items[top_n:]
    
    # Merge 'Other'
    final_data = dict(top_items)
    other_total = sum(v for _, v in other_items)
    if other_total > 0:
        final_data['Other'] = other_total

    # Prepare data
    labels = [f"{wrap_label(k)}\n{v}" for k, v in final_data.items()]
    sizes = list(final_data.values())
    colors = cmap(range(len(sizes)))

    # Plot
    plt.figure(figsize=(10, 6))
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.85, text_kwargs={'fontsize':10})
    plt.axis('off')
    plt.title(f"{title} (Top {top_n} + Other)", fontsize=14)
    plt.tight_layout()

    # Save if needed
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    
def plot_yearly_boxplot(
    df, subsets, year_col, value_col, start_year=None, end_year=None,
    xlabel="Year", ylabel="Values", title="Yearly Boxplot", 
    label_font_size=12, tick_font_size=10, title_font_size=14, 
    figsize=(12, 6), save_path=None, dpi=300
):
    """
    Plots a yearly boxplot for different groups.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least a year column.
    - subsets (dict): Dictionary where keys are group names and values are lists of row indices.
    - year_col (str): Column name representing the year.
    - value_col (str): Column name representing values to analyze.
    - start_year (int or None): Start year for filtering.
    - end_year (int or None): End year for filtering.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - title (str): Plot title.
    - label_font_size (int): Font size for axis labels.
    - tick_font_size (int): Font size for axis ticks.
    - title_font_size (int): Font size for the title.
    - figsize (tuple): Figure size (width, height).
    - save_path (str or None): Path to save the figure. If None, it is not saved.
    - dpi (int): Resolution for saving the figure.
    """
    
    # Create a new DataFrame for plotting
    plot_data = []
    
    for key, indices in subsets.items():
        subset_df = df.loc[indices]
        
        # Apply filtering on years
        if start_year is not None:
            subset_df = subset_df[subset_df[year_col] >= start_year]
        if end_year is not None:
            subset_df = subset_df[subset_df[year_col] <= end_year]
        
        # Append data for plotting
        for year, values in subset_df.groupby(year_col)[value_col]:
            for value in values:
                plot_data.append((year, value, key))
    
    # Convert list to DataFrame
    plot_df = pd.DataFrame(plot_data, columns=[year_col, value_col, "Group"])
    
    # Create the boxplot
    plt.figure(figsize=figsize)
    sns.boxplot(data=plot_df, x=year_col, y=value_col, hue="Group", dodge=True, width=0.6)

    # Labels and formatting
    plt.xlabel(xlabel, fontsize=label_font_size)
    plt.ylabel(ylabel, fontsize=label_font_size)
    plt.title(title, fontsize=title_font_size)
    plt.xticks(rotation=45, fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.legend(title="Group", fontsize=tick_font_size)
    
    # Save plot if required
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()

def plot_yearly_trend_with_regression(
    df, subsets, year_col, value_col, start_year=None, end_year=None,
    xlabel="Year", ylabel="Mean Value", title="Yearly Trends with Regression",
    xlabel_fontsize=12, ylabel_fontsize=12, tick_font_size=10, marker_size=20,
    xtick_rotation=45, ytick_rotation=0, x_lim=None, y_lim=None, is_median=True,
    title_fontsize=14, legend_fontsize=10, is_legend=True, colors=None, markers=None,
    trendline_width=1.5, figsize=(10, 6), save_path=None, dpi=300
):
    """
    Plots the mean value per year with a regression trendline for different subsets.
    Supports color and marker customization via dict or list.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    plt.figure(figsize=figsize)
    
    default_colors = sns.color_palette("tab10", len(subsets))
    default_markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '+']
    keys = list(subsets.keys())

    # Resolve colors
    if isinstance(colors, dict):
        color_map = {k: colors.get(k, default_colors[i % len(default_colors)]) for i, k in enumerate(keys)}
    elif isinstance(colors, list):
        color_map = {k: colors[i % len(colors)] for i, k in enumerate(keys)}
    else:
        color_map = {k: default_colors[i % len(default_colors)] for i, k in enumerate(keys)}

    # Resolve markers
    if isinstance(markers, dict):
        marker_map = {k: markers.get(k, default_markers[i % len(default_markers)]) for i, k in enumerate(keys)}
    elif isinstance(markers, list):
        marker_map = {k: markers[i % len(markers)] for i, k in enumerate(keys)}
    else:
        marker_map = {k: default_markers[i % len(default_markers)] for i, k in enumerate(keys)}

    for i, (key, indices) in enumerate(subsets.items()):
        subset_df = df.loc[indices]

        if start_year is not None:
            subset_df = subset_df[subset_df[year_col] >= start_year]
        if end_year is not None:
            subset_df = subset_df[subset_df[year_col] <= end_year]
        if is_median:
            mean_values = subset_df.groupby(year_col)[value_col].median()
        else:
            mean_values = subset_df.groupby(year_col)[value_col].mean()
        years = mean_values.index.to_numpy()
        mean_vals = mean_values.to_numpy()

        slope, intercept, r_value, p_value, std_err = stats.linregress(years, mean_vals)
        trendline = slope * years + intercept

        # Print regression statistics
        print(f"Subset: {key}")
        print(f"  Start year: {years[0]}")
        print(f"  End year: {years[-1]}")
        print(f"  Start value: {mean_vals[0]:.4f}")
        print(f"  End value: {mean_vals[-1]:.4f}")
        print(f"  Slope: {slope:.4f}")
        print(f"  Intercept: {intercept:.4f}")
        print(f"  p-value: {p_value:.4g}")
        print("-" * 40)

        plt.scatter(
            years, mean_vals,
            marker=marker_map[key],
            s=marker_size, label=f"{key}",
            color=color_map[key]
        )
        plt.plot(
            years, trendline,
            linestyle='dashed',
            linewidth=trendline_width,
            color=color_map[key],
            alpha=0.7,
            label=f"{key} Trend"
        )

    # Axis formatting
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.title(title, fontsize=title_fontsize)
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='out', length=5, width=1,
                   labelsize=tick_font_size, colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    plt.xticks(rotation=xtick_rotation)
    plt.yticks(rotation=ytick_rotation)

    if is_legend:
        plt.legend(fontsize=legend_fontsize)
    plt.grid(False)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()

# ---------------------------------------------------------------------------
# helper – centre positions from *flow* weights
# ---------------------------------------------------------------------------
def _calc_y_coords(weights, margin=0.02, eps=1e-4):
    """
    Parameters
    ----------
    weights : list[float]   # already in top→bottom order for one column
    margin  : float         # fraction of total height reserved for gaps
    eps     : float         # keep centres strictly within (0,1)

    Returns
    -------
    list[float]             # y‑centres in [0,1]
    """
    n = len(weights)
    if n == 1:
        return [0.5]

    w = np.array(weights, dtype=float)
    if w.sum() == 0:
        w[:] = 1.0          # fallback to equal blocks
    w /= w.sum()            # now Σw = 1

    gap_total = margin
    node_band = 1.0 - gap_total
    heights   = w * node_band
    gap       = gap_total / (n - 1)

    y_lower = 0.0
    centres = []
    for h in heights:
        centres.append(y_lower + 0.5 * h)
        y_lower += h + gap

    return [max(eps, min(1 - eps, y)) for y in centres]


# ---------------------------------------------------------------------------
# main plotting function
# ---------------------------------------------------------------------------
def plot_keyword_sankey(keyword_counts,  # still needed for categories / colours
                        keyword_pairs,
                        is_labels=True,
                        left_colormap='tab10',
                        right_colormap='Set2',
                        margin=0.02,
                        width=1000,
                        height=900,
                        font_size=12):
    """
    A two‑column Sankey where node heights and % labels are based ONLY on the
    sums of flows in `keyword_pairs`, not on `keyword_counts`.

    keyword_counts : {node: {'count': int, 'category': <left>|<right>}}
                     (counts are ignored for layout, but categories define sides)
    keyword_pairs  : {(right_node, left_node): flow_value}
                     – assumes orientation “(k2, k1)”  (k1 on left, k2 on right)
    """

    # ---- 1. split nodes by category ---------------------------------------
    categories = list({v['category'] for v in keyword_counts.values()})
    if len(categories) != 2:
        raise ValueError("Expected exactly two distinct categories.")
    cat_left, cat_right = categories

    group_left  = sorted([k for k, v in keyword_counts.items() if v['category'] == cat_left])
    group_right = sorted([k for k, v in keyword_counts.items() if v['category'] == cat_right])

    all_nodes = group_left + group_right
    node_idx  = {k: i for i, k in enumerate(all_nodes)}

    # ---- 2. derive weights from flows -------------------------------------
    # initialise to zero
    left_w  = {k: 0.0 for k in group_left}
    right_w = {k: 0.0 for k in group_right}

    for (k2, k1), v in keyword_pairs.items():  # (right, left) as per your data
        if k1 in left_w:
            left_w[k1]  += v     # outgoing flow from left node
        if k2 in right_w:
            right_w[k2] += v     # incoming flow to right node

    left_weights  = [left_w[k]  for k in group_left]
    right_weights = [right_w[k] for k in group_right]

    # ---- 3. colours -------------------------------------------------------
    cmap_L, cmap_R = plt.get_cmap(left_colormap), plt.get_cmap(right_colormap)
    col_left  = [mcolors.to_hex(cmap_L(i / max(len(group_left)  - 1, 1)))
                 for i in range(len(group_left))]
    col_right = [mcolors.to_hex(cmap_R(i / max(len(group_right) - 1, 1)))
                 for i in range(len(group_right))]
    col_L_map = dict(zip(group_left,  col_left))
    col_R_map = dict(zip(group_right, col_right))

    node_colors = [col_L_map.get(k, col_R_map.get(k, '#aaaaaa')) for k in all_nodes]

    # ---- 4. build link lists ---------------------------------------------
    keyword_pairs = dict(sorted(keyword_pairs.items(), key=lambda p: (p[0][0], p[0][1])))
    src, tgt, val, link_col = [], [], [], []
    for (k2, k1), v in keyword_pairs.items():
        src.append(node_idx[k1])              # left index
        tgt.append(node_idx[k2])              # right index
        val.append(v)
        link_col.append(col_L_map[k1])        # colour inherits the left node

    # ---- 5. labels based on flow shares -----------------------------------
    if is_labels:
        total_left  = sum(left_weights)
        total_right = sum(right_weights)

        def lbl_L(k):
            pct = 100.0 * left_w[k]  / total_left  if total_left  else 0
            return f"{k} {pct:.1f}%"

        def lbl_R(k):
            pct = 100.0 * right_w[k] / total_right if total_right else 0
            return f"{pct:.1f}% {k}"

        labels = [lbl_L(k) for k in group_left] + [lbl_R(k) for k in group_right]
    else:
        labels = [""] * len(all_nodes)

    # ---- 6. manual y‑coordinates from weights ----------------------------
    x_coords = [0.001] * len(group_left) + [0.999] * len(group_right)
    y_left   = _calc_y_coords(left_weights,  margin=margin)
    y_right  = _calc_y_coords(right_weights, margin=margin)
    y_coords = y_left + y_right

    # ---- 7. plot ----------------------------------------------------------
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            x=x_coords,
            y=y_coords
        ),
        link=dict(source=src, target=tgt, value=val, color=link_col)
    ))

    fig.update_layout(width=width, height=height,
                      font_size=font_size,
                      margin=dict(l=200, r=200, t=50, b=50))
    fig.show()
    

# ---------------------------------------------------------------------------
# helper – nice colour ramp
# ---------------------------------------------------------------------------
def _ramp(cmap_name, n):
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def plot_ABC_sankey(keyword_counts1, keyword_pairs1,
                    keyword_counts2, keyword_pairs2,
                    show_labels=True,
                    cmap_B="Blues", cmap_A="Greens", cmap_C="Oranges",
                    width=1100, height=650, font_size=12):
    """
    keyword_counts1 / keyword_pairs1  ->  dataset for A‑B
    keyword_counts2 / keyword_pairs2  ->  dataset for A‑C

    keyword_counts*  :  {node: {'count': int, 'category': 'A'|'B'|'C'}}
    keyword_pairs*   :  {(node1, node2): flow_value}   (order unimportant)

    The script merges both, fixes any A‑in ≠ A‑out, then plots.
    """

    # ---- 1. merge counts, summing duplicates for A -------------------------
    merged_counts = deepcopy(keyword_counts1)
    for k, v in keyword_counts2.items():
        if k in merged_counts:
            merged_counts[k]["count"] += v["count"]
        else:
            merged_counts[k] = deepcopy(v)

    # ---- 2. split into the 3 groups ---------------------------------------
    group_B = sorted([k for k, v in merged_counts.items() if v["category"] == "B"])
    group_A = sorted([k for k, v in merged_counts.items() if v["category"] == "A"])
    group_C = sorted([k for k, v in merged_counts.items() if v["category"] == "C"])

    # we will append dummy nodes later if needed
    all_nodes = group_B + group_A + group_C
    node_cat  = {k: v["category"] for k, v in merged_counts.items()}
    node_idx  = {k: i for i, k in enumerate(all_nodes)}

    # ---- 3. colours --------------------------------------------------------
    col_B = dict(zip(group_B, _ramp(cmap_B, len(group_B))))
    col_A = dict(zip(group_A, _ramp(cmap_A, len(group_A))))
    col_C = dict(zip(group_C, _ramp(cmap_C, len(group_C))))
    node_colour = {**col_B, **col_A, **col_C}

    # default colour for future dummy nodes
    DUMMY_COLOUR = "#DDDDDD"

    # ---- 4. build links (B→A and A→C) -------------------------------------
    src, tgt, val, link_col = [], [], [], []

    def add_flow(u, v, flow):
        """Orient flow so it is always B→A or A→C."""
        cu, cv = node_cat[u], node_cat[v]
        # B–A block
        if {"A", "B"} == {cu, cv}:
            if cu == "B":   s, t = u, v
            else:           s, t = v, u
            src.append(node_idx[s]);  tgt.append(node_idx[t])
            link_col.append(node_colour[s])
            val.append(flow)
        # A–C block
        elif {"A", "C"} == {cu, cv}:
            if cu == "A":   s, t = u, v
            else:           s, t = v, u
            src.append(node_idx[s]);  tgt.append(node_idx[t])
            link_col.append(node_colour[s])
            val.append(flow)
        # ignore pairs that don't fit the expected categories

    for (u, v), f in {**keyword_pairs1, **keyword_pairs2}.items():
        add_flow(u, v, f)

    # ---- 5. auto‑balance every A node --------------------------------------
    def balance_node(a_label):
        idx = node_idx[a_label]
        in_tot  = sum(v for s, t, v in zip(src, tgt, val) if t == idx)
        out_tot = sum(v for s, t, v in zip(src, tgt, val) if s == idx)
        diff = in_tot - out_tot
        if diff == 0:
            return                       # already balanced

        if diff > 0:   # more IN than OUT → need extra OUT to dummy C
            dummy = f"{a_label}_balC"
            cat = "C"
            src_node, tgt_node = a_label, dummy
            flow = diff
        else:          # more OUT than IN → need extra IN from dummy B
            dummy = f"{a_label}_balB"
            cat = "B"
            src_node, tgt_node = dummy, a_label
            flow = -diff

        # register dummy node
        node_cat[dummy] = cat
        node_idx[dummy] = len(all_nodes)
        all_nodes.append(dummy)
        node_colour[dummy] = DUMMY_COLOUR
        # add to correct group list (keeps columns consistent)
        if cat == "B":
            group_B.append(dummy)
        else:
            group_C.append(dummy)

        # add hidden label slot
        labels.append("")                # keep labels list length in sync later

        # append the balancing link
        src.append(node_idx[src_node])
        tgt.append(node_idx[tgt_node])
        val.append(flow)
        link_col.append(DUMMY_COLOUR)

    # initialise labels before balancing loop
    labels = []

    for a in group_A:
        balance_node(a)

    # ---- 6. labels & colours arrays ---------------------------------------
    # percentage labels (only for real nodes)
    if show_labels:
        col_totals = {
            "B": sum(merged_counts[k]["count"] for k in group_B if k in merged_counts),
            "A": sum(merged_counts[k]["count"] for k in group_A if k in merged_counts),
            "C": sum(merged_counts[k]["count"] for k in group_C if k in merged_counts)
        }

        def pct_lbl(k):
            if k not in merged_counts:   # dummy → blank
                return ""
            tot = col_totals[merged_counts[k]["category"]]
            pct = 100.0 * merged_counts[k]["count"] / tot if tot else 0
            return f"{k} {pct:.1f}%"

        labels = [pct_lbl(k) for k in all_nodes]
    else:
        labels = [""] * len(all_nodes)

    node_colors = [node_colour.get(k, DUMMY_COLOUR) for k in all_nodes]

    # ---- 7. draw -----------------------------------------------------------
    fig = go.Figure(go.Sankey(
        arrangement="snap",           # auto‑places by edge direction
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=src,
            target=tgt,
            value=val,
            color=link_col
        )
    ))

    fig.update_layout(width=width, height=height,
                      font_size=font_size,
                      margin=dict(l=130, r=130, t=40, b=40))
    fig.show()

def draw_horizontal_violin_with_central_tendency(
    df,
    keyword_subset,
    sentiment_col,
    save_path=None,
    dpi=300,
    figsize=(10, 6),
    palette="Set2",
    violin_width=0.8,
    point_color="black",
    point_marker="o",
    point_size=8,
    title="Horizontal Violin Plot with Central Tendency Points",
    title_fontsize=16,
    xlabel="Sentiment",
    ylabel="Keywords",
    label_fontsize=12,
    violin_colors=None,
    value_range=(-0.5, 0.5),
    padding=0.1,
    is_cat_ticklabel_visible=True,
    keyword_sorted=True,  # If True, sort keywords alphabetically
    central_tendency="mean",  # Options: "mean", "median"
    show_quartiles=False,     # If True, show 25th and 75th percentiles
    quartile_tick_length=0.2, # Length of the quartile tick
    quartile_color="gray",    # Color for quartile ticks
    quartile_linewidth=1      # Line width for quartile ticks
):
    sns.set(style="whitegrid")
    if keyword_sorted:
        sorted_keywords = sorted(keyword_subset.keys(), reverse=True)
    else:
        sorted_keywords = list(keyword_subset.keys())
    subset_data = []
    central_values = {}
    quartile_values = {}

    for keyword in sorted_keywords:
        indices = keyword_subset[keyword]
        subset_df = df.loc[indices]
        if not subset_df.empty:
            values = subset_df[sentiment_col].dropna()
            for val in values:
                subset_data.append({"Keyword": keyword, "Sentiment": val})
            if central_tendency == "mean":
                central_values[keyword] = values.mean()
            elif central_tendency == "median":
                central_values[keyword] = values.median()
            if show_quartiles:
                quartile_values[keyword] = (values.quantile(0.25), values.quantile(0.75))

    subset_df = pd.DataFrame(subset_data)

    if subset_df.empty:
        print("No data available for plotting.")
        return

    subset_df["Keyword"] = pd.Categorical(subset_df["Keyword"], categories=sorted_keywords, ordered=True)
    subset_df = subset_df.sort_values(by="Keyword")

    fig, ax = plt.subplots(figsize=figsize)

    # Determine violin color mapping
    unique_keywords = subset_df["Keyword"].cat.categories
    if isinstance(violin_colors, dict):
        custom_palette = [violin_colors.get(k, "gray") for k in unique_keywords]
    elif isinstance(violin_colors, (list, tuple)):
        custom_palette = list(violin_colors[:len(unique_keywords)])
    else:
        custom_palette = sns.color_palette(palette, len(unique_keywords))

    # Plot violins
    sns.violinplot(
        x="Sentiment",
        y="Keyword",
        data=subset_df,
        palette=custom_palette,
        density_norm="width",
        width=violin_width,
        inner=None,
        orient="horizontal",
        ax=ax
    )

    # Plot central tendency points
    for keyword, val in central_values.items():
        ax.scatter(val, keyword, color=point_color, marker=point_marker, s=point_size, zorder=10)

    # Plot quartiles if enabled
    if show_quartiles:
        for keyword, (q25, q75) in quartile_values.items():
            y_val = sorted_keywords.index(keyword)
            ax.vlines(x=q25, ymin=y_val - quartile_tick_length / 2, ymax=y_val + quartile_tick_length / 2,
                      color=quartile_color, linewidth=quartile_linewidth, zorder=9)
            ax.vlines(x=q75, ymin=y_val - quartile_tick_length / 2, ymax=y_val + quartile_tick_length / 2,
                      color=quartile_color, linewidth=quartile_linewidth, zorder=9)

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    ax.set_xlim(value_range)
    ax.set_ylim(-0.5 - padding, len(sorted_keywords) - 0.5 + padding)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize, color="black")
    ax.set_ylabel(ylabel, fontsize=label_fontsize, color="black")

    if not is_cat_ticklabel_visible:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
    else:
        wrapped_labels = wrap_labels(subset_df["Keyword"].cat.categories)
        ax.set_yticklabels(wrapped_labels, fontsize=label_fontsize)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.show()