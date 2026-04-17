# AllSky Light Pollution Time Series Analysis
## Project Description

This project aimed to analyze time series data from AllSky cameras to understand and visualize light pollution trends over time, focusing on the Allegheny Observatory in Pittsburgh, PA. By leveraging advanced data analysis techniques, we hoped to gain insights into the changes in light pollution seen in the sky over Pittsburgh's Northside. In particular, we investigated the effect of the recent [streetlight modernization](https://www.pghled.org/) on the light pollution trends.

Unfortunately, due to complications in data acquisition and storage, we were unable to make rigorous conclusions about the light pollution trends over the primary period of interest. This is detailed in the [Project Writeup](docs/writeup.md).

We make this project publicly available to the community for educational purposes. The code and notes are licensed under the MIT License.

## Selected Plots
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/ad1a1586-aebe-4997-ac8c-bd7d283fae36" />
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/2504a1f8-6514-429e-94a8-074fbf56a64f" />


## Table of Contents

### [Data](data/)
We include preprocessed data from the image set in a columnar format. There are two files:
- `ideal_df.parquet`: Contains parsed information from the images, limited to nighttime images with no moon illumination but otherwise unenriched and unfiltered.
- `enriched_df.parquet`: Contains the same information as `ideal_df.parquet`, but this enriched version of the data includes statistics from each image, particularly pixel brightness information.

### [Images](images/)
We do not include the raw images in this repository, but we do include [instructions](images/instructions.md) for acquiring the images and how to store them for compatibility with our code.

<img width="640" height="480" alt="AllSkyImage000197127" src="https://github.com/user-attachments/assets/aba36e21-e85b-4a42-aa23-6ad785db52ad" />

### [Writeup](docs/writeup.md)

### [Notebooks](notebooks/)
We include Jupyter notebooks that demonstrate the data analysis and visualization techniques used in this project.

- Conclusions
  - [acquisition-method-unsuitable.ipynb](notebooks/conclusions/acquistion-method-unsuitable.ipynb): This notebook shows the two main plots that support our conclusion that our acquisition method was unsuitable for scientific use.
- Experimental
  - [apply-pca-2d-to-images.ipynb](notebooks/experimental/apply-pca-2d-to-images.ipynb): This notebook demonstrates how to apply PCA to images to reduce the dimensionality. We investigated machine learning classification techniques to select clear night imagery, but ultimately hand-filtered the images for a narrow time window covering the primary period of interest.
  - [apply-pca-3d-visualization.ipynb](notebooks/experimental/apply-pca-3d-visualization.ipynb): We visualize the PCA projections of some images in 3D. The subsets of imagery clearly group together, implying the feasibility of automated classification techniques.
  - [solar-lunar-ephem.ipynb](notebooks/experimental/solar-lunar-ephem.ipynb): We studied the `skyfield` package to determine the position of the sun and moon in the sky, which we used to filter out images that were illuminated by the moon or the sun.
- Exploratory
  - [explore-fits-exposures.ipynb](notebooks/exploratory/explore-fits-exposures.ipynb): We explore the fits files and exposure times of the images. The objective was to determine if we could learn what the FITS format would provide our analysis.
  - [solve-antialiased-parse.ipynb](notebooks/exploratory/solve-antialiased-parse.ipynb): This early notebook investigated the parsing approach we used to read the text off the images.
  - [study-fit-vs-jpg-calibration.ipynb](notebooks/exploratory/study-fit-vs-jpg-calibration.ipynb): We studied if we could understand the relationship between the JPG and FITS outputs in hopes to conclusively determine the calibration of the camera, and when the settings were changed - automatically or otherwise.
- Exploratory: All Parsed Data
  - [exposure-time-only-analysis.ipynb](notebooks/exploratory-all-parsed/exposure-time-only-analysis.ipynb): This notebook explores the exposure time of the images, attempting to determine a relationship between exposure time and light pollution without needing to process the imagery itself (beyond parsing).
- Exploratory: Selected data subset
  - [full-range-enriched.ipynb](notebooks/exploratory-nights-filtered/full-range-enriched.ipynb): This notebook explores the enriched data, factoring exposure time and image brightness into the analysis.
  - [full-range-exposure-only.ipynb](notebooks/exploratory-nights-filtered/full-range-exposure-only.ipynb): This notebook studied the exposure time more closely, attempting to ascertain how the camera chose a time for each exposure.
  - [streetlight-modernization-optimized.ipynb](notebooks/exploratory-nights-filtered/streetlight-modernization-optimized.ipynb): This notebook explores the data specifically during the streetlight modernization, taking place from May 2025 onward.
- Processing
  - [alias-parse-regression-testing.ipynb](notebooks/processing/alias-parse-regression-testing.ipynb): This notebook was used to ensure changes to parsing wouldn't break earlier work.
  - [antialias-parse.ipynb](notebooks/processing/antialias-parse.ipynb): This notebook was used to parse the images and monitor the progress of the parsing process.
  - [brute-force-parse.ipynb](notebooks/processing/brute-force-parse.ipynb): This notebook was used to parse the images and monitor the progress of the parsing process, using a brute force approach.
  - [generate-enriched-df.ipynb](notebooks/processing/generate-enriched-df.ipynb): This process loaded each image, masked it to exclude the horizon, and then calculated statistics from the image.
  - [generate-ideal-df.ipynb](notebooks/processing/generate-ideal-df.ipynb): This process loaded the parsed results from all images and then filtered out images that were illuminated by the moon or the sun, using the Skyfield library for ephemeris calculations.
  - [validate-missing-source-data.ipynb](notebooks/processing/validate-missing-source-data.ipynb): This notebook was used to test an approach to determining which entries in the dataframes were excluded from the analysis by hand-filtering.

## Data Sources
The project uses data from the Allegheny Observatory's AllSky camera, which captures images of the sky at regular intervals. 

The Observatory's AllSky camera is an SBIG AllSky-340, with a Kodak KAI-340 CCD capturing at 640 x 480 pixels.

## Credits

### University of Pittsburgh (Allegheny Observatory)
For data access and assistance. See the [Allegheny Observatory website](https://www.observatory.pitt.edu/) for more information about the Observatory. See [images/instructions.md](images/instructions.md) for more information about acquiring the data.
### Carnegie Mellon University
- Diane Turnshek (Special Lecturer)
- Joshua Freudenhammer (Alumni)

