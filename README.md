# 🖼️➡️📦 2D to 3D Image Generator

An AI-powered web app that transforms 2D images into interactive 3D representations using depth estimation and object segmentation.

## ✨ Features

- **AI Depth Estimation**: Simulates depth maps from 2D images
- **Object Segmentation**: Identifies and isolates objects in the image
- **3D Visualization**: Interactive point cloud, mesh, and layered plane views
- **Customizable Rendering**: Adjust depth scale, extrusion, and lighting
- **Export Options**: Download depth maps and 3D representations

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **3D Visualization**: Plotly
- **Image Processing**: OpenCV, PIL
- **Mock AI Models**: NumPy (replace with real models like MiDaS/SAM)
- **Utilities**: NumPy, Matplotlib

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/2d-to-3d-generator.git
   cd 2d-to-3d-generator

2.Install dependencies:
       pip install -r requirements.txt

3.Run the app: 
     streamlit run app.py

📸 Example Usage
Upload an image through the interface

Click "Generate 3D" to process the image

Explore different 3D visualization modes:

Point Cloud

Mesh

Layered Planes

Adjust parameters like depth scale and lighting

Export your results

🔧 Customization
To use real AI models instead of mock implementations:

Replace generate_depth_map() with actual MiDaS/DPT implementation

Replace segment_image() with actual SAM/Mask R-CNN implementation

Update the model selection dropdowns in the UI   
