"""
Geographic visualization using Folium.

Provides interactive map visualizations for location-based financial data analysis.
"""

from typing import Optional, List, Union, Dict, Any, Tuple
import pandas as pd
import folium
from folium import plugins


class GeographicMaps:
    """Geographic map generator using Folium."""
    
    def __init__(
        self,
        tiles: str = 'OpenStreetMap',
        zoom_start: int = 10,
        width: str = '100%',
        height: str = '600px'
    ):
        """
        Initialize geographic maps generator.
        
        Args:
            tiles: Tile provider (OpenStreetMap, CartoDB positron, CartoDB dark_matter, etc.)
            zoom_start: Default zoom level
            width: Map width
            height: Map height
        """
        self.tiles = tiles
        self.zoom_start = zoom_start
        self.width = width
        self.height = height
    
    def create_point_map(
        self,
        data: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        popup_fields: Optional[List[str]] = None,
        tooltip_fields: Optional[List[str]] = None,
        center: Optional[Tuple[float, float]] = None,
        marker_color: str = 'blue',
        icon: str = 'info-sign'
    ) -> folium.Map:
        """
        Create map with point markers.
        
        Args:
            data: DataFrame with location data
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            popup_fields: Fields to show in popup
            tooltip_fields: Fields to show in tooltip
            center: Map center (lat, lon), auto-calculated if None
            marker_color: Marker color
            icon: Marker icon name
            
        Returns:
            Folium Map object
        """
        # Calculate center if not provided
        if center is None:
            center = (data[lat_col].mean(), data[lon_col].mean())
        
        # Create base map
        m = folium.Map(
            location=center,
            zoom_start=self.zoom_start,
            tiles=self.tiles,
            width=self.width,
            height=self.height
        )
        
        # Add markers
        for idx, row in data.iterrows():
            lat = row[lat_col]
            lon = row[lon_col]
            
            if pd.notna(lat) and pd.notna(lon):
                # Create popup
                popup_html = None
                if popup_fields:
                    popup_content = "<br>".join([f"<b>{field}:</b> {row[field]}" for field in popup_fields if field in row])
                    popup_html = folium.Popup(popup_content, max_width=300)
                
                # Create tooltip
                tooltip_html = None
                if tooltip_fields:
                    tooltip_content = "<br>".join([f"{field}: {row[field]}" for field in tooltip_fields if field in row])
                    tooltip_html = folium.Tooltip(tooltip_content)
                
                folium.Marker(
                    location=[lat, lon],
                    popup=popup_html,
                    tooltip=tooltip_html,
                    icon=folium.Icon(color=marker_color, icon=icon)
                ).add_to(m)
        
        return m
    
    def create_heatmap(
        self,
        data: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        intensity_col: Optional[str] = None,
        center: Optional[Tuple[float, float]] = None,
        radius: int = 15,
        blur: int = 15,
        min_opacity: float = 0.4,
        gradient: Optional[Dict[float, str]] = None
    ) -> folium.Map:
        """
        Create density heatmap.
        
        Args:
            data: DataFrame with location data
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            intensity_col: Column for intensity values
            center: Map center (lat, lon)
            radius: Heatmap radius
            blur: Heatmap blur
            min_opacity: Minimum opacity
            gradient: Custom color gradient
            
        Returns:
            Folium Map object
        """
        # Calculate center if not provided
        if center is None:
            center = (data[lat_col].mean(), data[lon_col].mean())
        
        # Create base map
        m = folium.Map(
            location=center,
            zoom_start=self.zoom_start,
            tiles=self.tiles,
            width=self.width,
            height=self.height
        )
        
        # Prepare heatmap data
        if intensity_col:
            heat_data = [
                [row[lat_col], row[lon_col], row[intensity_col]]
                for idx, row in data.iterrows()
                if pd.notna(row[lat_col]) and pd.notna(row[lon_col]) and pd.notna(row[intensity_col])
            ]
        else:
            heat_data = [
                [row[lat_col], row[lon_col]]
                for idx, row in data.iterrows()
                if pd.notna(row[lat_col]) and pd.notna(row[lon_col])
            ]
        
        # Add heatmap layer
        plugins.HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            min_opacity=min_opacity,
            gradient=gradient
        ).add_to(m)
        
        return m
    
    def create_choropleth(
        self,
        data: pd.DataFrame,
        geo_data: Union[str, Dict],
        columns: List[str],
        key_on: str,
        fill_color: str = 'YlOrRd',
        fill_opacity: float = 0.7,
        line_opacity: float = 0.2,
        legend_name: str = 'Value',
        center: Optional[Tuple[float, float]] = None
    ) -> folium.Map:
        """
        Create choropleth map for regional statistics.
        
        Args:
            data: DataFrame with regional data
            geo_data: GeoJSON file path or dict
            columns: [key_column, value_column] for mapping
            key_on: GeoJSON property to match (e.g., 'feature.properties.name')
            fill_color: Color scheme
            fill_opacity: Fill opacity
            line_opacity: Border opacity
            legend_name: Legend title
            center: Map center (lat, lon)
            
        Returns:
            Folium Map object
        """
        # Default center if not provided
        if center is None:
            center = (37.0902, -95.7129)  # US center
        
        # Create base map
        m = folium.Map(
            location=center,
            zoom_start=4,
            tiles=self.tiles,
            width=self.width,
            height=self.height
        )
        
        # Add choropleth layer
        folium.Choropleth(
            geo_data=geo_data,
            data=data,
            columns=columns,
            key_on=key_on,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            line_opacity=line_opacity,
            legend_name=legend_name
        ).add_to(m)
        
        return m
    
    def create_cluster_map(
        self,
        data: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        popup_fields: Optional[List[str]] = None,
        tooltip_fields: Optional[List[str]] = None,
        center: Optional[Tuple[float, float]] = None,
        icon_create_function: Optional[str] = None
    ) -> folium.Map:
        """
        Create map with clustered markers for performance.
        
        Args:
            data: DataFrame with location data
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            popup_fields: Fields to show in popup
            tooltip_fields: Fields to show in tooltip
            center: Map center (lat, lon)
            icon_create_function: JavaScript function for custom cluster icons
            
        Returns:
            Folium Map object
        """
        # Calculate center if not provided
        if center is None:
            center = (data[lat_col].mean(), data[lon_col].mean())
        
        # Create base map
        m = folium.Map(
            location=center,
            zoom_start=self.zoom_start,
            tiles=self.tiles,
            width=self.width,
            height=self.height
        )
        
        # Create marker cluster
        marker_cluster = plugins.MarkerCluster(
            icon_create_function=icon_create_function
        ).add_to(m)
        
        # Add markers to cluster
        for idx, row in data.iterrows():
            lat = row[lat_col]
            lon = row[lon_col]
            
            if pd.notna(lat) and pd.notna(lon):
                # Create popup
                popup_html = None
                if popup_fields:
                    popup_content = "<br>".join([f"<b>{field}:</b> {row[field]}" for field in popup_fields if field in row])
                    popup_html = folium.Popup(popup_content, max_width=300)
                
                # Create tooltip
                tooltip_html = None
                if tooltip_fields:
                    tooltip_content = "<br>".join([f"{field}: {row[field]}" for field in tooltip_fields if field in row])
                    tooltip_html = folium.Tooltip(tooltip_content)
                
                folium.Marker(
                    location=[lat, lon],
                    popup=popup_html,
                    tooltip=tooltip_html
                ).add_to(marker_cluster)
        
        return m
    
    def create_route_map(
        self,
        data: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        order_col: Optional[str] = None,
        center: Optional[Tuple[float, float]] = None,
        line_color: str = 'blue',
        line_weight: int = 3,
        line_opacity: float = 0.7
    ) -> folium.Map:
        """
        Create map showing routes between points.
        
        Args:
            data: DataFrame with location data
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            order_col: Column defining point order
            center: Map center (lat, lon)
            line_color: Route line color
            line_weight: Route line width
            line_opacity: Route line opacity
            
        Returns:
            Folium Map object
        """
        # Sort by order if provided
        if order_col:
            data = data.sort_values(order_col)
        
        # Calculate center if not provided
        if center is None:
            center = (data[lat_col].mean(), data[lon_col].mean())
        
        # Create base map
        m = folium.Map(
            location=center,
            zoom_start=self.zoom_start,
            tiles=self.tiles,
            width=self.width,
            height=self.height
        )
        
        # Extract coordinates
        coordinates = [
            [row[lat_col], row[lon_col]]
            for idx, row in data.iterrows()
            if pd.notna(row[lat_col]) and pd.notna(row[lon_col])
        ]
        
        # Add route line
        if len(coordinates) > 1:
            folium.PolyLine(
                coordinates,
                color=line_color,
                weight=line_weight,
                opacity=line_opacity
            ).add_to(m)
        
        # Add markers for start and end
        if len(coordinates) > 0:
            folium.Marker(
                coordinates[0],
                popup='Start',
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
            
            if len(coordinates) > 1:
                folium.Marker(
                    coordinates[-1],
                    popup='End',
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(m)
        
        return m
    
    def add_custom_markers(
        self,
        map_obj: folium.Map,
        locations: List[Tuple[float, float]],
        popups: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        icons: Optional[List[str]] = None
    ) -> folium.Map:
        """
        Add custom markers to existing map.
        
        Args:
            map_obj: Existing folium Map
            locations: List of (lat, lon) tuples
            popups: List of popup texts
            colors: List of marker colors
            icons: List of icon names
            
        Returns:
            Updated folium Map object
        """
        for i, loc in enumerate(locations):
            marker_kwargs = {'location': loc}
            
            if popups and i < len(popups):
                marker_kwargs['popup'] = popups[i]
            
            if colors or icons:
                color = colors[i] if colors and i < len(colors) else 'blue'
                icon = icons[i] if icons and i < len(icons) else 'info-sign'
                marker_kwargs['icon'] = folium.Icon(color=color, icon=icon)
            
            folium.Marker(**marker_kwargs).add_to(map_obj)
        
        return map_obj
    
    def add_circle_markers(
        self,
        map_obj: folium.Map,
        data: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        radius_col: Optional[str] = None,
        color_col: Optional[str] = None,
        default_radius: int = 10,
        default_color: str = 'blue',
        popup_fields: Optional[List[str]] = None
    ) -> folium.Map:
        """
        Add circle markers to existing map.
        
        Args:
            map_obj: Existing folium Map
            data: DataFrame with location data
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            radius_col: Column for circle radius
            color_col: Column for circle color
            default_radius: Default radius if radius_col not provided
            default_color: Default color if color_col not provided
            popup_fields: Fields to show in popup
            
        Returns:
            Updated folium Map object
        """
        for idx, row in data.iterrows():
            lat = row[lat_col]
            lon = row[lon_col]
            
            if pd.notna(lat) and pd.notna(lon):
                radius = row[radius_col] if radius_col and radius_col in row else default_radius
                color = row[color_col] if color_col and color_col in row else default_color
                
                # Create popup
                popup_html = None
                if popup_fields:
                    popup_content = "<br>".join([f"<b>{field}:</b> {row[field]}" for field in popup_fields if field in row])
                    popup_html = folium.Popup(popup_content, max_width=300)
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    popup=popup_html,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6
                ).add_to(map_obj)
        
        return map_obj
    
    def save(self, map_obj: folium.Map, filepath: str) -> None:
        """
        Save map to HTML file.
        
        Args:
            map_obj: Folium Map object
            filepath: Output file path
        """
        map_obj.save(filepath)
