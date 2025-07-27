import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="2D to 3D Image Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .step-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🖼️➡️📦 2D to 3D Image Generator</h1>
    <p>Upload a 2D image and transform it into a 3D representation with AI-powered depth estimation</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'depth_map' not in st.session_state:
    st.session_state.depth_map = None
if 'segmentation_masks' not in st.session_state:
    st.session_state.segmentation_masks = None

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Processing Controls")
    
    # Model selection
    st.subheader("Model Selection")
    depth_model = st.selectbox(
        "Depth Estimation Model",
        ["MiDaS", "DPT", "DenseDepth"],
        index=0,
        help="Choose the depth estimation model"
    )
    
    segmentation_model = st.selectbox(
        "Segmentation Model",
        ["SAM (Segment Anything)", "DeepLab", "Mask R-CNN"],
        index=0,
        help="Choose the segmentation model"
    )
    
    # Processing parameters
    st.subheader("3D Parameters")
    depth_scale = st.slider("Depth Scale", 0.1, 5.0, 1.0, 0.1)
    extrusion_strength = st.slider("Extrusion Strength", 0.1, 2.0, 0.5, 0.1)
    smooth_depth = st.checkbox("Smooth Depth Map", value=True)
    
    # Rendering options
    st.subheader("Rendering Options")
    render_mode = st.selectbox(
        "3D Render Mode",
        ["Point Cloud", "Mesh", "Layered Planes"],
        index=0
    )
    
    show_wireframe = st.checkbox("Show Wireframe", value=False)
    lighting_intensity = st.slider("Lighting Intensity", 0.1, 2.0, 1.0, 0.1)

# Mock functions for AI models (replace with actual implementations)
@st.cache_data
def generate_depth_map(_image, model_name="MiDaS"):
    """Generate depth map from image using selected model"""
    # Mock depth map generation
    # In real implementation, load and use MiDaS, DPT, or DenseDepth
    
    img_array = np.array(_image.convert('RGB'))
    height, width = img_array.shape[:2]
    
    # Create a realistic-looking depth map
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    depth = np.exp(-(x**2 + y**2) * 0.5)  # Gaussian-like depth
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.1, depth.shape)
    depth = np.clip(depth + noise, 0, 1)
    
    return depth

@st.cache_data
def segment_image(_image, model_name="SAM"):
    """Segment image into objects using selected model"""
    # Mock segmentation
    # In real implementation, use SAM, DeepLab, or Mask R-CNN
    
    img_array = np.array(_image.convert('RGB'))
    height, width = img_array.shape[:2]
    
    # Create mock segmentation masks
    masks = []
    num_segments = np.random.randint(3, 7)
    
    for i in range(num_segments):
        mask = np.zeros((height, width))
        # Random circular/elliptical segments
        center_x = np.random.randint(width//4, 3*width//4)
        center_y = np.random.randint(height//4, 3*height//4)
        radius_x = np.random.randint(30, width//4)
        radius_y = np.random.randint(30, height//4)
        
        y, x = np.ogrid[:height, :width]
        mask_condition = ((x - center_x)**2 / radius_x**2 + 
                         (y - center_y)**2 / radius_y**2) <= 1
        mask[mask_condition] = 1
        masks.append(mask)
    
    return masks

def create_3d_visualization(image, depth_map, render_mode="Point Cloud"):
    """Create 3D visualization with 360° rotation and enhanced clarity"""
    img_array = np.array(image.convert('RGB'))
    height, width = img_array.shape[:2]
    
    # Create coordinate grids - center them for better rotation
    x = np.arange(width) - width/2
    y = np.arange(height) - height/2
    X, Y = np.meshgrid(x, y)
    
    # Apply depth scaling with better range
    Z = (depth_map - 0.5) * depth_scale * 150  # Center and scale depth
    
    if render_mode == "Point Cloud":
        # Adaptive sampling based on image size for better detail
        target_points = 15000  # Optimal point count for performance vs quality
        sample_rate = max(1, (height * width) // target_points)
        
        # Create more uniform sampling
        step = int(np.sqrt(sample_rate))
        y_indices, x_indices = np.mgrid[0:height:step, 0:width:step]
        
        x_flat = X[y_indices, x_indices].flatten()
        y_flat = Y[y_indices, x_indices].flatten() 
        z_flat = Z[y_indices, x_indices].flatten()
        
        # Get colors with better sampling
        colors = img_array[y_indices, x_indices].reshape(-1, 3)
        colors_rgb = [f'rgb({r},{g},{b})' for r, g, b in colors]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            mode='markers',
            marker=dict(
                size=3,  # Slightly larger points for better visibility
                color=colors_rgb,
                opacity=0.8,
                line=dict(width=0)  # Remove marker outlines
            ),
            name='Point Cloud',
            hovertemplate='<b>Position:</b><br>X: %{x}<br>Y: %{y}<br>Depth: %{z}<extra></extra>'
        )])
        
    elif render_mode == "Mesh":
        # Enhanced mesh with better color mapping
        # Downsample for performance
        downsample = max(1, max(height, width) // 200)
        X_mesh = X[::downsample, ::downsample]
        Y_mesh = Y[::downsample, ::downsample]
        Z_mesh = Z[::downsample, ::downsample]
        colors_mesh = img_array[::downsample, ::downsample]
        
        # Create RGB surface coloring
        surfacecolor = np.sqrt(colors_mesh[:,:,0]**2 + colors_mesh[:,:,1]**2 + colors_mesh[:,:,2]**2)
        
        fig = go.Figure(data=[go.Surface(
            x=X_mesh, y=Y_mesh, z=Z_mesh,
            surfacecolor=surfacecolor,
            colorscale='Viridis',
            showscale=False,
            opacity=0.9,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
            contours=dict(
                x=dict(show=show_wireframe, color="white", width=1),
                y=dict(show=show_wireframe, color="white", width=1),
                z=dict(show=show_wireframe, color="white", width=1)
            )
        )])
        
    else:  # Enhanced Layered Planes
        num_layers = 8  # More layers for better depth separation
        depth_min, depth_max = Z.min(), Z.max()
        depth_levels = np.linspace(depth_min, depth_max, num_layers)
        
        fig = go.Figure()
        
        # Color palette for layers
        layer_colors = px.colors.qualitative.Set3[:num_layers]
        
        for i, level in enumerate(depth_levels):
            # Create layer mask with overlap for smooth transition
            layer_thickness = (depth_max - depth_min) / (num_layers * 0.8)
            mask = np.abs(Z - level) < layer_thickness
            
            if np.any(mask):
                x_layer = X[mask]
                y_layer = Y[mask]
                z_layer = np.full_like(x_layer, level)
                
                # Use original colors but add layer tinting
                colors = img_array[mask]
                colors_rgb = [f'rgba({r},{g},{b},0.7)' for r, g, b in colors]
                
                fig.add_trace(go.Scatter3d(
                    x=x_layer,
                    y=y_layer,
                    z=z_layer,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=colors_rgb,
                        line=dict(width=0.5, color=layer_colors[i])
                    ),
                    name=f'Layer {i+1} (depth: {level:.1f})',
                    hovertemplate=f'<b>Layer {i+1}</b><br>X: %{{x}}<br>Y: %{{y}}<br>Depth: {level:.1f}<extra></extra>'
                ))
    
    # Enhanced layout with 360° rotation capabilities
    fig.update_layout(
        title=dict(
            text=f"3D Visualization - {render_mode}",
            x=0.5,
            font=dict(size=18, color='white')
        ),
        paper_bgcolor='rgba(17,17,17,1)',  # Dark background
        plot_bgcolor='rgba(17,17,17,1)',
        scene=dict(
            # Enhanced axis styling
            xaxis=dict(
                title="X",
                backgroundcolor="rgba(17,17,17,1)",
                gridcolor="rgba(255,255,255,0.2)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.4)",
            ),
            yaxis=dict(
                title="Y", 
                backgroundcolor="rgba(17,17,17,1)",
                gridcolor="rgba(255,255,255,0.2)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.4)",
            ),
            zaxis=dict(
                title="Depth",
                backgroundcolor="rgba(17,17,17,1)", 
                gridcolor="rgba(255,255,255,0.2)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.4)",
            ),
            # Enhanced camera with better initial position
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            # Enhanced lighting
            aspectratio=dict(x=1, y=1, z=0.8),
            bgcolor="rgba(17,17,17,1)"
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        # Add animation controls for smooth rotation
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True]},
                              {"scene.camera.eye": dict(x=2*np.cos(0), y=2*np.sin(0), z=1.2)}],
                        label="▶ Auto Rotate",
                        method="animate"
                    ),
                    dict(
                        args=[{"visible": [True]},
                              {"scene.camera.eye": dict(x=1.8, y=1.8, z=1.2)}],
                        label="⏸ Stop", 
                        method="animate"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.02,
                yanchor="top"
            ),
        ]
    )
    
    # Add custom JavaScript for smooth 360° rotation
    fig.add_annotation(
        text="💡 Click and drag to rotate • Scroll to zoom • Double-click to reset view",
        xref="paper", yref="paper",
        x=0.5, y=-0.05, xanchor='center', yanchor='top',
        showarrow=False,
        font=dict(size=12, color="rgba(255,255,255,0.7)")
    )
    
    return fig

# Main app layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.subheader("📤 Step 1: Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a 2D image to convert to 3D"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Display image info
        st.info(f"📊 Image Size: {image.size[0]} x {image.size[1]} pixels")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.subheader("🧠 Step 2: AI Processing")
        
        if st.button("🚀 Generate 3D", type="primary", use_container_width=True):
            with st.spinner("Processing with AI models..."):
                # Generate depth map
                progress_bar = st.progress(0)
                st.text("Generating depth map...")
                depth_map = generate_depth_map(image, model_name=depth_model)
                progress_bar.progress(33)
                
                # Segment image
                st.text("Segmenting objects...")
                segmentation_masks = segment_image(image, segmentation_model)
                progress_bar.progress(66)
                
                # Store in session state
                st.session_state.depth_map = depth_map
                st.session_state.segmentation_masks = segmentation_masks
                st.session_state.processed_image = image
                
                progress_bar.progress(100)
                st.success("✅ Processing complete!")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Display results
if st.session_state.depth_map is not None:
    st.markdown("---")
    st.subheader("📊 Processing Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Depth Map", "🎯 Segmentation", "📦 3D View", "📈 Analytics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Depth Map")
            fig_depth = px.imshow(st.session_state.depth_map, color_continuous_scale='viridis')
            fig_depth.update_layout(title="Estimated Depth Map")
            st.plotly_chart(fig_depth, use_container_width=True)
        
        with col2:
            st.subheader("Depth Statistics")
            depth_stats = {
                "Min Depth": f"{st.session_state.depth_map.min():.3f}",
                "Max Depth": f"{st.session_state.depth_map.max():.3f}",
                "Mean Depth": f"{st.session_state.depth_map.mean():.3f}",
                "Std Deviation": f"{st.session_state.depth_map.std():.3f}"
            }
            
            for key, value in depth_stats.items():
                st.metric(key, value)
    
    with tab2:
        st.subheader("Object Segmentation")
        if st.session_state.segmentation_masks:
            num_cols = min(3, len(st.session_state.segmentation_masks))
            cols = st.columns(num_cols)
            
            for i, mask in enumerate(st.session_state.segmentation_masks[:6]):
                with cols[i % num_cols]:
                    fig_mask = px.imshow(mask, color_continuous_scale='Blues')
                    fig_mask.update_layout(title=f"Segment {i+1}")
                    st.plotly_chart(fig_mask, use_container_width=True)
    
    with tab3:
        st.subheader("3D Visualization")
        
        # Enhanced controls for 3D viewing
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            view_preset = st.selectbox(
                "📷 Camera Preset Views",
                ["Free Rotation", "Front View", "Top View", "Side View", "Isometric"],
                help="Choose a preset camera angle or use Free Rotation for manual control"
            )
        
        with col2:
            auto_rotate = st.checkbox("🔄 Auto Rotate", help="Automatically rotate the 3D view")
        
        with col3:
            point_density = st.slider("Point Density", 5000, 25000, 15000, 1000, 
                                    help="Adjust number of points displayed")
        
        # Generate 3D visualization
        fig_3d = create_3d_visualization(
            st.session_state.processed_image, 
            st.session_state.depth_map, 
            render_mode
        )
        
        # Apply camera presets
        if view_preset == "Front View":
            fig_3d.update_layout(scene_camera=dict(eye=dict(x=0, y=-2.5, z=0)))
        elif view_preset == "Top View":
            fig_3d.update_layout(scene_camera=dict(eye=dict(x=0, y=0, z=2.5)))
        elif view_preset == "Side View":
            fig_3d.update_layout(scene_camera=dict(eye=dict(x=2.5, y=0, z=0)))
        elif view_preset == "Isometric":
            fig_3d.update_layout(scene_camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)))
        
        # Display the 3D plot
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Add viewing instructions
        st.info("""
        🎮 **3D Navigation Controls:**
        - **Click & Drag**: Rotate the view 360°
        - **Scroll Wheel**: Zoom in/out  
        - **Double Click**: Reset to default view
        - **Shift + Drag**: Pan the view
        - Use the camera presets above for quick angle changes
        """)
        
        # Performance metrics
        if st.session_state.processed_image:
            img_array = np.array(st.session_state.processed_image)
            total_pixels = img_array.shape[0] * img_array.shape[1]
            displayed_points = min(point_density, total_pixels)
            
            st.caption(f"💾 Rendering {displayed_points:,} points from {total_pixels:,} total pixels ({displayed_points/total_pixels*100:.1f}% sample rate)")
        
        # Export options
        st.subheader("💾 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Depth Map"):
                # Create download for depth map
                depth_img = Image.fromarray((st.session_state.depth_map * 255).astype(np.uint8))
                buf = BytesIO()
                depth_img.save(buf, format="PNG")
                st.download_button(
                    label="💾 depth_map.png",
                    data=buf.getvalue(),
                    file_name="depth_map.png",
                    mime="image/png"
                )
        
        with col2:
            if st.button("📥 Download Point Cloud"):
                st.info("Point cloud export would generate PLY/OBJ file")
        
        with col3:
            if st.button("📥 Download 3D Mesh"):
                st.info("3D mesh export would generate STL/OBJ file")
    
    with tab4:
        st.subheader("📈 Processing Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Objects Detected", 
                len(st.session_state.segmentation_masks) if st.session_state.segmentation_masks else 0
            )
        
        with col2:
            st.metric("Depth Range", f"{st.session_state.depth_map.max() - st.session_state.depth_map.min():.3f}")
        
        with col3:
            st.metric("Processing Model", f"{depth_model} + {segmentation_model.split()[0]}")
        
        with col4:
            st.metric("Render Mode", render_mode)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🖼️➡️📦 2D to 3D Image Generator | Built with Streamlit</p>
    <p>Powered by AI: Depth Estimation + Object Segmentation + 3D Rendering</p>
</div>
""", unsafe_allow_html=True)