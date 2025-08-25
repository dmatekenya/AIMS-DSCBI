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

1. **Workspace Preparation**
   - Ensure your virtual environment is activated and all required packages are installed.
   - Set up your workspace directories as shown in the notebook (`data/`, `src/`, etc.).

2. **Import Required Packages**
   - Import all necessary Python libraries for geospatial analysis, data handling, and visualization (see the notebook for reference).

3. **Prepare Data Paths**
   - Define file paths for the nightlight raster and the shapefiles for both national and cell-level boundaries.
   - Make sure your data files are unzipped and located in the correct directories.

4. **Import Custom Scripts**
   - Add your `src/` directory to the Python path and import the custom `ZonalStatistics` class as shown in the notebook.

5. **Clip the Raster**
   - Use the provided function to clip the global nightlight raster to the Rwanda national boundary.
   - Save the clipped raster for further analysis.

6. **Visualize the Data**
   - Load and visualize the clipped raster along with the cell boundaries to confirm the data alignment.

7. **Generate Zonal Statistics**
   - Use the `ZonalStatistics` class to calculate summary statistics (mean, median, count, sum) for each cell (admin4 region).
   - Display and interpret the results.

8. **Save and Report Results**
   - Merge the zonal statistics with cell attribute data and save the results as a CSV file.
   - Identify and report the cells with the highest nightlight values.

### Deliverables

- A Jupyter notebook (.ipynb) that follows the workflow above, with clear code, comments, and outputs.
- The resulting CSV file with zonal statistics (do not push large data files to GitHub).
- Short written answers or markdown cells explaining your findings and any challenges encountered.
