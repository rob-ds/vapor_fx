# VAPOR_FX

## Validation Approach for Pattern-based Object Recognition with Fuzzy Similarity for Extreme events

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

VAPOR_FX is a comprehensive framework for retrieval, processing, and object-based validation of extreme events. It implements a novel approach using fuzzy pattern recognition for comparing patterns between data to be validated (test data) and the observational benchmark (reference data). Current implementation deploys a case study for validation of ERA5 using E-OBS as ground truth over the Western Mediterranean region.

## Repository Note

This is the public release of vapor_fx version 1.0, previously developed as an internal project. The Git history has been reset for this public GitHub repository to provide a clean version control starting point. This public GitHub repository has been created specifically to showcase the project for professional purposes.


## Key Features

- **Advanced Event Detection**: Object-based identification of extreme precipitation events
- **Novel Validation Methodology**: Pattern recognition with fuzzy similarity metrics rather than traditional point-based comparison (i.e., confusion matrix)
- **Fragmentation Handling**: Detects cases where a single reference event may be represented as multiple (fragmented) test objects
- **Statistical Significance**: Rigorous significance testing through non-parametric permutation methods
- **Multiple Temporal Aggregations**: Analyzes ERA5 data with different 24-hour aggregation periods (00, 06, 12, 18 UTC starting times)
- **Complete Workflow**: End-to-end pipeline from data retrieval to validation results
- **European Domain Focus**: Case study designed for European precipitation patterns

## Framework Overview

VAPOR_FX implements a complete workflow for precipitation analysis:

1. **Data Retrieval**: Download test and reference data via Climate Data Store (CDS) API
2. **Data Processing**: 
   - Standardize domains and create land-sea masks
   - Create multiple temporal aggregations of test data (6h and four different 24h periods)
   - Regrid reference data to test data resolution
   - Calculate basic statistics between datasets
3. **Threshold Calculation**: Compute percentile-based thresholds (e.g., R95p) to identify extreme events
4. **Object-Based Validation**: Apply the novel VAPOR_FX methodology to evaluate detection performance across different temporal aggregations

### Validation Methodology

The core innovation of VAPOR_FX is its object-based validation approach, which represents a significant advancement over traditional point-by-point comparison methods.

#### Why Object-Based Validation?

Traditional point-by-point validation methods often fail to capture the true performance of models, particularly for extreme events, because:
- Small spatial displacements can result in "double penalty" errors
- They don't recognize when a pattern is correctly predicted but slightly offset
- They struggle with capturing fragmentation (when one event becomes multiple smaller events)

#### The VAPOR_FX Approach

VAPOR_FX addresses these limitations through:

- **Advanced Object Identification**: 
  - Connected cells are identified as coherent objects
  - Uses image processing techniques (connected component labeling) to identify physically meaningful structures
  - Allows configurable minimum and maximum size thresholds for objects

- **Sophisticated Fuzzy Matching**: 
  - Objects are matched using a hybrid similarity metric that considers both:
    - Jaccard spatial overlap (intersection over union)
    - Centroid distance with physics-informed scaling
  - Matching is directional (reference → test) with configurable similarity thresholds
  - Produces a continuous similarity measure rather than binary matches

- **Physics-Based Distance Penalties**: 
  - Distance penalties are dynamically scaled by object size
  - Larger objects are allowed greater displacement before penalization
  - Accounts for the physical understanding that larger systems may have greater uncertainty in exact position

- **Fragmentation Pattern Detection**: 
  - Uniquely identifies cases where one reference object matches multiple test objects
  - Implements many-to-one matching algorithms to capture splitting/fragmentation
  - Allows for physically realistic representation of how extreme events evolve

- **Statistical Significance Testing**:
  - Implements rigorous permutation testing to determine result significance
  - Provides p-values for all key metrics
  - Uses optimization techniques for efficient computation of statistical tests

- **Comprehensive Metric Suite**: 
  - Event Coincidence Rate (ECR): Proportion of reference events detected
  - Local Intensity Ratio (LIR): Quantifies intensity correspondence
  - Intensity Bias (IB): Measures systematic over/under-prediction
  - Traditional contingency measures: (POD, FAR, CSI)
  - Object displacement vectors: Quantifies systematic spatial shifts

## Project Structure

```
vapor_fx/
├── data/                # Data directory
│   ├── processed/       # Processed outputs from each stage
│   │   ├── eobs_analysis/
│   │   ├── eobs_percentiles/
│   │   ├── era5_analysis/
│   │   ├── era5_percentiles/
│   │   └── stats/
│   ├── raw/             # Raw data from CDS API
│   │   ├── eobs/
│   │   ├── era5/
│   │   └── era5-land/
│   └── validated/       # Validation results
├── scripts/             # Processing scripts
│   ├── process_data/
│   │   ├── mask_creator.py
│   │   └── process_data.py
│   ├── process_thresholds/
│   │   └── process_thresholds.py
│   ├── process_validation/
│   │   └── process_validation.py
│   └── retrieve_data/
│       ├── retrieve_eobs_cdsapi.py
│       ├── retrieve_era5_cdsapi.py
│       └── retrieve_era5-land_cdsapi.py
├── src/                 # Source code modules
│   ├── data/
│   │   ├── loaders/
│   │   │   ├── load_eobs.py
│   │   │   └── load_era5.py
│   │   ├── processor/
│   │   │   └── processor.py
│   │   ├── data_base.py
│   │   ├── data_domains.py
│   │   ├── data_masking.py
│   │   └── data_transformations.py
│   ├── thresholds/
│   │   ├── threshold_base.py
│   │   ├── threshold_processors.py
│   │   └── threshold_utils.py
│   ├── validation/
│   │   ├── validation_utils.py
│   │   └── vapor_fx.py
│   ├── visualization/   # [Under development]
│   └── utils.py
├── tests/               # [Under development]
├── README.md            # Project documentation
├── LICENSE              # MIT License file
├── vapor-fx1.0.yml      # Conda environment specification
└── .gitignore           # Git ignore file
```

## Installation

### Prerequisites

- **Python 3.10** (required)
- **CDS API Access** (required for data retrieval)
  - You must [register for a Climate Data Store account](https://cds.climate.copernicus.eu/user/register)
  - Set up your [CDS API key](https://cds.climate.copernicus.eu/api-how-to) in `~/.cdsapirc`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/vapor_fx.git
cd vapor_fx
```

2. Create and activate the conda environment:
```bash
conda env create -f vapor-fx1.0.yml
conda activate vapor-fx1.0
```

## Usage

The repository includes raw data files for a case study period (1980-2023) using ERA5 (test data) and E-OBS (reference data) precipitation for the Western Mediterranean region (extended information below). You can use these files with the default settings or retrieve your own data for different periods.

### 1. Data Retrieval

> **Note**: This step is not necessary if using the included case study data (1980-2023).

Download ERA5 precipitation data:
```bash
python scripts/retrieve_data/retrieve_era5_cdsapi.py
```

Download E-OBS precipitation data:
```bash
python scripts/retrieve_data/retrieve_eobs_cdsapi.py
```

Create land-sea mask:
```bash
python scripts/retrieve_data/retrieve_era5-land_cdsapi.py
python scripts/process_data/mask_creator.py
```

### 2. Data Processing

Process ERA5 and E-OBS data (defaults to the case study period 1980-2023):
```bash
python scripts/process_data/process_data.py
```

This step:
- Processes raw precipitation data for analysis
- Creates 6-hourly aggregations from original ERA5 data
- Generates four different 24-hour aggregations with varying starting hours (00, 06, 12, 18 UTC)
- Regridds E-OBS data to match ERA5 grid spacing
- Optionally applies land-sea masking

Additional options:
```bash
python scripts/process_data/process_data.py --years 1990-2010 --mask --analysis-only --season SON
```

### 3. Threshold Processing

Calculate percentile thresholds (defaults to R95p for the case study period):
```bash
python scripts/process_thresholds/process_thresholds.py
```

Additional options:
```bash
python scripts/process_thresholds/process_thresholds.py --years 1990-2010 --dataset all --percentiles 90,95,98
```

### 4. Validation

Run the VAPOR_FX validation process:
```bash
python scripts/process_validation/process_validation.py
```

By default, this validates all four ERA5 24-hour aggregations (tp_24h_00, tp_24h_06, tp_24h_12, tp_24h_18) against the E-OBS reference dataset. This multi-temporal approach allows for robust evaluation of timing effects in precipitation forecasts.

Additional options:
```bash
# Validate specific ERA5 temporal aggregations
python scripts/process_validation/process_validation.py --variables tp_24h_00,tp_24h_12 --percentiles 95
```

## Extensibility and Scalability

The VAPOR_FX framework is designed with extensibility and scalability as core principles:

### Adaptable to Different Domains

While the current implementation focuses on extreme precipitation over Europe, the framework is readily adaptable to:

- **Different Geographical Regions**: By modifying domain parameters in configuration files
- **Alternative Variables**: The object-based methodology can be applied to other meteorological fields such as temperature extremes, wind patterns, or atmospheric pressure systems
- **Various Data Sources**: The loader architecture allows for integration of additional data sources beyond ERA5 and E-OBS

### Modular Design for Maintainability

The codebase follows a highly modular design with clear separation of concerns:

- **Layered Architecture**: 
  - Data retrieval layer (CDS API integration)
  - Processing layer (standardization and transformation)
  - Analysis layer (threshold calculation)
  - Validation layer (object-based comparison)

- **Abstract Base Classes**: Key components use abstraction to enable extension:
  - `DataLoader`: Extendable for new data sources
  - `ThresholdProcessor`: Adaptable to different threshold methodologies
  - `PrecipitationValidator`: Core validation logic with configurable parameters

- **Configuration Management**: Validation parameters are centralized in configuration classes

### Parallel Processing Support

The VAPOR_FX framework implements efficient parallel processing to handle large datasets and computationally intensive validation:

- **Dask Integration**: The validation process leverages Dask for distributed computation, automatically parallelizing the object identification and matching process across timesteps
- **Scalable Workflow**: Computationally intensive tasks like object matching and permutation testing are optimized for parallel execution
- **Progress Tracking**: Integrated progress bars provide visibility into parallel execution status
- **Resource Management**: The framework intelligently manages memory through chunked operations and LRU caching
- **Configurable Batch Processing**: Permutation tests are processed in configurable batch sizes to balance memory usage and computational efficiency

## Development Status

The VAPOR_FX framework is under active development:

- **Core Framework**: Complete implementation of the novel validation methodology
- **Documentation**: Full documentation of code modules and validation approach
- **Data Flow**: Complete end-to-end workflow from data retrieval to validation
- **Test Suite**: [Under development] Unit tests will be added to ensure code reliability
- **Visualization**: [Under development] Visualization modules for netCDF output data
- **Framework Extension**: Ongoing work to apply the methodology to additional meteorological variables

## Case Study Data

The repository includes raw data for a case study period (1980-2023) over Europe. This allows users to test the framework without needing to download large historical datasets.

### Spatial and Temporal Domain

The included case study focuses on:
- **Data Retrieval Domain**: European region (25°N-55°N, 15°W-22°E)
- **Analysis Domain**: Mediterranean region (35°N-45°N, 5°W-12°E)
- **Period**: 1980-2023
- **Season**: SON (September, October, November)
- **Variables**: Precipitation from ERA5 and E-OBS datasets

The framework uses two spatial domains:
1. **Retrieval Domain**: The broader European area from which raw data is retrieved
2. **Analysis Domain**: A focused Mediterranean region where the detailed validation is performed

This two-domain approach allows efficient analysis of specific regions while maintaining broader contextual data availability.

### Temporal Aggregation Rationale

A key design feature of VAPOR_FX is its analysis of multiple 24-hour aggregations with different starting times (00, 06, 12, and 18 UTC). This approach addresses a known limitation of E-OBS, for which the data sources for the precipitation are rain gauge data which do not have a uniform way of defining the 24-hour period over which precipitation measurements are made. Therefore, there is no uniform time period (for instance, 06 UTC previous day to 06 UTC today) which could be attached to the daily precipitation.

By comparing the observational E-OBS dataset with multiple temporal aggregations of ERA5 data, VAPOR_FX accounts for temporal uncertainty in the reference dataset (E-OBS). This multi-temporal approach enables a more robust comparison when the exact 24-hour measurement period in the observational data is not standardized across all stations.

## Requirements

The project requires Python 3.10 and the following key dependencies:
- xarray
- numpy
- pandas
- netcdf4
- scipy
- dask
- matplotlib
- cartopy
- cdsapi

A complete list of dependencies can be found in `vapor-fx1.0.yml`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [ERA5 Reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)
- [E-OBS Dataset](https://cds.climate.copernicus.eu/datasets/insitu-gridded-observations-europe?tab=overview)
- [Climate Data Store API](https://cds.climate.copernicus.eu/)

## Citation

If you use VAPOR_FX in your research, please cite it as:

```
[Citation details to be added after publication]
```

