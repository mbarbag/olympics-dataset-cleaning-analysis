# Olympics Dataset Cleaning & Analysis

## Overview
Comprehensive data cleaning pipeline for Olympic athletes biographical and results data (1896-2022).

## Dataset Sources
- olympedia.org data scraped by Keith Galli
- Covers summer & winter Olympics

## Key Transformations
- Birth/death information parsing
- Name standardization
- Physical measurements extraction
- Role-based filtering (Olympic competitors only)
- Cross-country competition analysis

## Key Findings
- 10 athletes competed for different countries than birth country
- 42 female athletes with official titles
- Gender distribution insights

## Files Generated
- `bios_new.csv`: Cleaned biographical data
- `results_new.csv`: Cleaned competition results

## Usage
Run cells sequentially to reproduce the cleaning pipeline.