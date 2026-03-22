"""
Streamlit application for evolved photo search with multi-model support.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from datetime import datetime, date
from search.vector_search import VectorSearch
from config.models import list_models, ModelType


# Page configuration
st.set_page_config(
    page_title="Photo Search - Embedding Evolution",
    page_icon="📸",
    layout="wide"
)

# Initialize search engine
@st.cache_resource
def get_searcher():
    return VectorSearch(default_model="florence-2-base")


searcher = get_searcher()


# Sidebar filters
st.sidebar.title("🔍 Filters")

# Model selection
st.sidebar.subheader("Embedding Model")
available_models = searcher.get_available_models()
model_options = {m['name']: m for m in available_models if m['photo_count'] > 0}

if model_options:
    selected_model = st.sidebar.selectbox(
        "Search using embeddings from:",
        options=list(model_options.keys()),
        format_func=lambda x: f"{x} ({model_options[x]['dimension']}d, {model_options[x]['photo_count']} photos)"
    )
else:
    st.sidebar.warning("No models with embeddings found. Please ingest photos first.")
    selected_model = "florence-2-base"

# Date range filter
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    year_from = st.number_input("From Year", min_value=1900, max_value=2100, value=2000, key="year_from")
with col2:
    year_to = st.number_input("To Year", min_value=1900, max_value=2100, value=datetime.now().year, key="year_to")

# Location filter (optional)
st.sidebar.subheader("Location (Optional)")
use_location = st.sidebar.checkbox("Filter by GPS bounds")
if use_location:
    lat_min = st.sidebar.number_input("Min Latitude", min_value=-90.0, max_value=90.0, value=-90.0, step=0.1)
    lat_max = st.sidebar.number_input("Max Latitude", min_value=-90.0, max_value=90.0, value=90.0, step=0.1)
    lon_min = st.sidebar.number_input("Min Longitude", min_value=-180.0, max_value=180.0, value=-180.0, step=0.1)
    lon_max = st.sidebar.number_input("Max Longitude", min_value=-180.0, max_value=180.0, value=180.0, step=0.1)
else:
    lat_min = lat_max = lon_min = lon_max = None

# Black & White filter
st.sidebar.subheader("Photo Type")
bw_filter = st.sidebar.radio(
    "Color/B&W",
    options=["All", "Color Only", "B&W Only"],
    index=0
)

is_black_white = None
if bw_filter == "B&W Only":
    is_black_white = True
elif bw_filter == "Color Only":
    is_black_white = False

# Main content
st.title("📸 Photo Search - Embedding Evolution")
st.markdown("""
Search your photo collection using natural language queries. 
The search uses vector similarity on AI-generated captions.
""")

# Display database stats
stats = searcher.get_stats()
st.metric("Total Photos", stats['total_photos'])

# Search query
st.subheader("🔎 Search")
query = st.text_input(
    "Describe the photos you're looking for:",
    placeholder="e.g., 'kids playing at the beach', 'sunset over mountains', 'birthday party'",
    label_visibility="collapsed"
)

# Results limit
limit = st.slider("Number of results", min_value=5, max_value=50, value=20)

# Perform search
if query:
    # Build filters
    filters = {
        'year_from': year_from,
        'year_to': year_to,
    }
    
    if use_location and lat_min is not None:
        filters['lat_min'] = lat_min
        filters['lat_max'] = lat_max
        filters['lon_min'] = lon_min
        filters['lon_max'] = lon_max
    
    if is_black_white is not None:
        filters['is_black_white'] = is_black_white
    
    with st.spinner(f"Searching for '{query}' using {selected_model}..."):
        results = searcher.search_by_text(
            query_text=query,
            model_name=selected_model,
            limit=limit,
            filters=filters
        )
    
    if results:
        st.success(f"Found {len(results)} matching photos")
        
        # Display results in grid
        cols = st.columns(3)
        for i, result in enumerate(results):
            with cols[i % 3]:
                # Create card for each photo
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 20px;">
                    <img src="file://{result['file_path']}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 4px;">
                    <p style="font-size: 12px; color: #666; margin-top: 8px;">
                        <b>Score:</b> {result['similarity_score']:.3f}<br>
                        <b>Date:</b> {result['capture_date'] or 'Unknown'}<br>
                        <b>Caption:</b> {result['caption_text'][:100] if result['caption_text'] else 'No caption'}...
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show metadata in expander
                with st.expander("Details"):
                    st.write(f"**File:** {result['file_name']}")
                    st.write(f"**Path:** {result['file_path']}")
                    st.write(f"**Date:** {result['capture_date']}")
                    if result['latitude'] and result['longitude']:
                        st.write(f"**Location:** {result['latitude']:.4f}, {result['longitude']:.4f}")
                    st.write(f"**B&W:** {'Yes' if result['is_black_white'] else 'No'}")
                    if result['caption_text']:
                        st.write(f"**Caption:** {result['caption_text']}")
    else:
        st.info("No photos found matching your criteria. Try adjusting your filters or search query.")

# Sidebar: Database info
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Database Info")
st.sidebar.write(f"**Total Photos:** {stats['total_photos']}")
st.sidebar.write(f"**B&W Photos:** {stats['black_and_white_count']}")

if stats['models']:
    st.sidebar.write("**Models with embeddings:**")
    for model in stats['models']:
        st.sidebar.write(f"  • {model['model']} ({model['dimension']}d): {model['count']} photos")

# Help section
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    ### Search Tips
    - Use natural language descriptions like "kids at the beach" or "mountain sunset"
    - Combine with filters to narrow down results
    - Higher similarity scores indicate better matches
    
    ### Model Selection
    - Different embedding models capture different aspects of image captions
    - Select a model based on which was used when photos were ingested
    - Newer models may provide better semantic understanding
    
    ### Filters
    - **Date Range**: Filter by year the photo was taken
    - **Location**: Filter by GPS coordinates if available
    - **Photo Type**: Filter for black & white or color photos
    
    ### Adding New Photos
    Use the ingestion tool to add new photos:
    ```bash
    python ingestion/photo_ingest.py --photos-dir /path/to/photos --caption-model florence-2-base
    ```
    """)
