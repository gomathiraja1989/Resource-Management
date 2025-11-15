import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import DataLoader, preprocess_data
from utils.eda import EDA

# Set page config
st.set_page_config(
    page_title="Industrial Human Resource Geo-Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data."""
    try:
        # Load and preprocess data
        loader = DataLoader()
        loader.load_data()
        merged_df = loader.get_merged_data()
        
        if merged_df is None or merged_df.empty:
            st.error("No data loaded. Please check the data files and try again.")
            st.stop()
            
        merged_df = preprocess_data(merged_df)
        return merged_df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    st.title("Industrial Human Resource Geo-Visualization Dashboard")
    st.markdown("""
    This dashboard provides insights into the distribution of industrial workers across different 
    states in India, categorized by worker type (main/marginal) and gender.
    """)
    
    # Add debug toggle to sidebar
    debug_info = st.sidebar.checkbox("Show debug info", key="debug_info")
    
    # Add a note about data source
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Data Source: Census 2011 - Worker Population by Main and Marginal Workers, State: ARUNACHAL PRADESH"
    )

    # Load data
    merged_df = load_data()
    
    if merged_df is None:
        st.error("Failed to load data. Please check the data files and try again.")
        return

    # Get unique states for the filter
    states = sorted([str(s) for s in merged_df['state'].unique()]) if 'state' in merged_df.columns else []

    # Sidebar filters
    st.sidebar.title("Filters")
    
    # State filter
    if states:
        selected_states = st.sidebar.multiselect(
            "Select States", 
            states, 
            default=states[0] if states else None
        )
    else:
        st.warning("No state information found in the data.")

    # Worker type filter
    worker_type = st.sidebar.selectbox(
        "Worker Type",
        ["All Workers", "Main Workers", "Marginal Workers"],
        index=0  # Default to "All Workers"
    )

    # Gender filter
    gender = st.sidebar.selectbox(
        "Gender",
        ["All", "Male", "Female"],
        index=0  # Default to "All"
    )

    # Apply filters
    filtered_df = merged_df.copy()

    # Debug: Show available columns
    if 'state' not in filtered_df.columns:
        st.warning("Warning: 'state' column not found in data. Available columns: " + 
                  ", ".join(filtered_df.columns))

    if states and selected_states:
        filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]

    # Determine which columns to use based on filters
    value_col = None
    if worker_type == "Main Workers":
        base_col = 'main_workers_total'
        if gender == "Male" and 'main_workers_male' in filtered_df.columns:
            value_col = 'main_workers_male'
        elif gender == "Female" and 'main_workers_female' in filtered_df.columns:
            value_col = 'main_workers_female'
        elif 'main_workers_total' in filtered_df.columns:
            value_col = 'main_workers_total'
        
    elif worker_type == "Marginal Workers":
        base_col = 'marginal_workers_total'
        if gender == "Male" and 'marginal_workers_male' in filtered_df.columns:
            value_col = 'marginal_workers_male'
        elif gender == "Female" and 'marginal_workers_female' in filtered_df.columns:
            value_col = 'marginal_workers_female'
        elif 'marginal_workers_total' in filtered_df.columns:
            value_col = 'marginal_workers_total'
        
    else:  # All Workers
        base_col = 'total_workers'
        if gender == "Male" and 'total_workers_male' in filtered_df.columns:
            value_col = 'total_workers_male'
        elif gender == "Female" and 'total_workers_female' in filtered_df.columns:
            value_col = 'total_workers_female'
        elif 'total_workers' in filtered_df.columns:
            value_col = 'total_workers'

    # Display the map
    st.subheader("Geographical Distribution")

    # Debug info
    if debug_info:
        st.sidebar.subheader("Debug Information")
        st.sidebar.write("### Raw Data Info")
        st.sidebar.write("Columns in data:", list(merged_df.columns))
        st.sidebar.write("First few rows:", merged_df.head(3).to_dict('records'))
        
        st.sidebar.write("### Filtered Data Info")
        st.sidebar.write("Selected value column:", value_col)
        st.sidebar.write("Available columns:", list(filtered_df.columns))
        st.sidebar.write("Sample data:", filtered_df.head(3).to_dict('records'))

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    elif 'state' not in filtered_df.columns:
        st.error("State information is missing from the data. Cannot create geographical visualization.")
    elif value_col is None or value_col not in filtered_df.columns:
        st.warning(f"No {gender.lower()} {worker_type.lower()} data available for the selected filters.")
        st.info(f"Available columns: {', '.join(filtered_df.columns)}")
    else:
        # Ensure we have valid numeric data
        if not pd.api.types.is_numeric_dtype(filtered_df[value_col]):
            st.error(f"Selected column '{value_col}' does not contain numeric data.")
        else:
            # Show some statistics
            total_workers = filtered_df[value_col].sum()
            st.metric(f"Total {gender} {worker_type}", f"{total_workers:,}")
            
            try:
                # Ensure we have valid numeric data
                if pd.api.types.is_numeric_dtype(filtered_df[value_col]):
                    # Group by state and sum the values
                    state_data = filtered_df.groupby('state', as_index=False).agg({value_col: 'sum'})
                    
                    # Debug: Show state data
                    if debug_info:
                        st.sidebar.write("State aggregation:", state_data.to_dict('records'))
                    
                    if not state_data.empty:
                        # Create the choropleth map with a more visible color scale
                        fig = px.choropleth(
                            state_data,
                            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
                            featureidkey='properties.ST_NM',
                            locations='state',
                            color=value_col,
                            color_continuous_scale='Viridis',
                            title=f"{worker_type} - {gender} Distribution by State",
                            labels={value_col: 'Number of Workers'},
                            hover_data={
                                'state': True,
                                value_col: ':'  # Format with thousands separator
                            }
                        )
                        
                        # Add hover template
                        fig.update_traces(
                            hovertemplate='<b>%{location}</b><br>' +
                                        f'Workers: %{{z:,}}<br>' +
                                        '<extra></extra>'
                        )
                        
                        # Update map layout
                        fig.update_geos(
                            visible=False,
                            projection_scale=5,
                            center={"lat": 20.5937, "lon": 78.9629},  # Center on India
                            showcountries=False,
                            showsubunits=True,
                            subunitcolor="grey",
                            subunitwidth=0.5
                        )
                        
                        # Adjust layout for better display
                        fig.update_layout(
                            margin={"r": 0, "t": 40, "l": 0, "b": 0},
                            height=600,
                            coloraxis_colorbar={
                                'title': 'Number of Workers',
                                'thicknessmode': 'pixels',
                                'thickness': 15,
                                'lenmode': 'pixels',
                                'len': 300,
                                'y': 0.5,
                                'yanchor': 'middle',
                                'tickformat': ','
                            }
                        )
                        
                        # Show the plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add a bar chart for better comparison
                        st.subheader(f"Top 10 States by {worker_type} ({gender})")
                        top_states = state_data.nlargest(10, value_col)
                        
                        fig_bar = px.bar(
                            top_states,
                            x='state',
                            y=value_col,
                            text=value_col,
                            labels={'state': 'State', value_col: 'Number of Workers'},
                            color=value_col,
                            color_continuous_scale='Viridis'
                        )
                        
                        # Format the bar chart
                        fig_bar.update_traces(
                            texttemplate='%{text:,}',
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Workers: %{y:,}<extra></extra>'
                        )
                        
                        fig_bar.update_layout(
                            xaxis_tickangle=-45,
                            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
                            plot_bgcolor='white',
                            showlegend=False,
                            height=500,
                            margin=dict(t=30, b=150)  # Add more bottom margin for state names
                        )
                        
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Show data table for debugging
                        if debug_info:
                            st.subheader("Data Preview")
                            st.dataframe(state_data)
                    else:
                        st.warning("No data available for the selected filters after aggregation.")
                else:
                    st.error(f"Selected column '{value_col}' does not contain numeric data.")
                    
            except Exception as e:
                st.error(f"An error occurred while creating the visualization: {str(e)}")
                if debug_info:
                    import traceback
                    st.text(traceback.format_exc())

if __name__ == "__main__":
    main()