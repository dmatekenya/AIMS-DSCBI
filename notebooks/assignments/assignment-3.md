# Nightlight Analysis Environment Setup

## Introduction

The goal of this assignment is to help you practice Python environment setup, specifically for geospatial data analysis projects. You'll learn to create isolated virtual environments, manage package dependencies, and integrate your work with version control using GitHub. This exercise simulates a real-world scenario where you need to set up a reproducible analysis environment for spatial data processing.

## Scenario
You're tasked with analyzing nightlight data to understand economic activity patterns across districts and cells. This requires setting up a geospatial analysis environment and sharing your work via GitHub.

## Data
### Night Lights Dataset
We will use the [2024 annual nightlights file](https://drive.google.com/file/d/1bH-IiSHHsUqJXEkVrbD2uT7xholbIFkN/view?usp=share_link). For this exercise, you don’t need to worry about the technical details of nightlights imagery. If you’re interested in learning more, you can check out our [previous assignment](https://github.com/dmatekenya/AIMS-DSCBI/blob/main/notebooks/assignments/assignment-2.ipynb), which explains nightlights in greater depth.
### Administrative Boundaries

The administrative boundaries for Rwanda are provided as shapefiles, a widely used GIS data format. Each shapefile consists of at least four associated files, so be sure to download all components—not just the file with the .shp extension. The relevant datasets are:
- [Rwanda national boundaries](https://drive.google.com/drive/folders/1cPwbcclnt0UcSkUPeYCjdFUWlhzyPO9N?usp=share_link). This will be used to clip the global night lights raster.
- [Rwanda admin4 (cell) boundaries](https://drive.google.com/drive/folders/1zC_qFY2svEyi8QAIhjL82C2DJLYSTwZw?usp=share_link). We will use the cells as zones for generating summary statistics.

---

## Part 1: GitHub Repository Setup

### Tasks:
1. Create a new GitHub repository named `nightlight-analysis-[yourname]`
2. Initialize with README.md
3. Create the following folder structure:
```text
nightlight-analysis/
├── data/           (leave empty - for provided files)
├── src/            (for Python scripts)
├── outputs/        (for results)
├── requirements.txt
└── README.md
```
### Deliverables for Submission
1. A link to your GitHub repository

### Other Requirements
1. Make sure the `data/` folder is included in your `.gitignore` file so that no data files are pushed to GitHub.



## Part 2: Virtual Environment Creation
### Required Setup:
Create a virtual environment using **pip/venv** with these specific package versions:
- python>=3.9,<3.12
- gdal==3.6.2
- geopandas==0.12.2
- rasterio==1.3.6
- pandas==1.5.3
- numpy==1.24.3
- matplotlib==3.6.3
- seaborn==0.12.2
- jupyter==1.0.0

### Steps:
1. Create virtual environment: `python -m venv .venv-ntl`
2. Activate the environment
3. Install all required packages
4. Create requirements.txt file
5. Test your installation

### Deliverables for Submission
1. **requirements.txt** pushed to your GitHub repo
2. **Environment test**: Screenshot of Python importing all packages successfully:
   ```python
   import gdal, geopandas, rasterio, pandas, numpy, matplotlib
   print("All packages imported successfully!")
   print(f"GDAL version: {gdal.__version__}")
   print(f"GeoPandas version: {geopandas.__version__}")



## Spatial Analysis

In this section, you will replicate the workflow demonstrated in the provided notebook to generate zonal statistics for nightlight intensity across administrative regions in Rwanda. Follow the steps below to complete your analysis:

### Instructions

1. **Get Updates from Course Repository**
   - Follow instructions from GitHub Workflow
   - Ensure you have used to pull to get updates from the course repository

2. **Copy Materials**
   - assignment instructions. This is this markdown 
   - [assignment-3-zonal-statistics notebook](https://github.com/dmatekenya/AIMS-DSCBI/blob/main/notebooks/assignments/assignment-3-zonal-stats.ipynb). Copy this notebook into your project repo. The repo you create in step-1
   - [spatial_utils](https://github.com/dmatekenya/AIMS-DSCBI/tree/main/src/spatial_utils). Copy this module and add it to the src fodler within your repository.

3. **Complete Exercise**
   - Update the **assignment-3-zonal-statistics notebook** to ensure you use correct file paths.
   - Go through and run the notebook cells.

### Deliverables

- A Jupyter notebook (.ipynb) that follows the workflow above, with clear code, comments, and outputs.
- The resulting CSV file with zonal statistics.

## Submission Instructions

Please provide the following for your assignment submission:

1. **GitHub Repository:** Share a link to your project repository.
2. **Environment Verification:** Upload a screenshot of your Python environment, ensuring your terminal displays your name.
3. **Zonal Statistics Notebook:** Submit the completed Jupyter notebook.
4. **CSV Output:** Include a GitHub link to the CSV file containing your zonal statistics results. Although the `data/` folder is typically excluded from version control, make an exception to include this CSV file.

