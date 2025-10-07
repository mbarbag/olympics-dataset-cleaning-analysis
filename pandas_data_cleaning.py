# %%
import pandas as pd
import numpy as np

# %%
pd.__version__

# %% [markdown]
# ## About the Dataset 
#  
# Before cleaning the Olympics Dataset, it's important to understand its origin and structure.
#  
# **Data Source:** This data comes from [olympedia.org](https://olympedia.org) and was web scraped by [Keith Galli](https://github.com/KeithGalli).
#  
# **Dataset Structure:**
# - `bios.csv`: Contains raw biographical information on each athlete
# - `results.csv`: Contains a row-by-row breakdown of each event athletes competed in and their results
# 
# **Coverage:** This repository contains comprehensive data on summer & winter Olympic athletes and their results from 1896-2022.
# 

# %%
# Load the datasets directly from GitHub repository
bios = pd.read_csv('https://github.com/KeithGalli/Olympics-Dataset/raw/refs/heads/master/athletes/bios.csv')
results = pd.read_csv('https://github.com/KeithGalli/Olympics-Dataset/raw/refs/heads/master/results/results.csv')

# %% [markdown]
# ### Use Case
#  
# The objective of this analysis is to explore these datasets and extract meaningful insights about Olympic athletes, including:
# - Physical characteristics (height, weight)
# - Geographic origins (birth locations)
# - Career duration and competition patterns
# - Performance across different Olympic games

# %% [markdown]
# ## Dataset Exploration

# %% [markdown]
# ### `bios` Dataset Analysis

# %%
# Examine the structure of raw biographical data
bios.head()

# %%
# Identify data types, missing values, and memory usage
# This helps plan the cleaning strategy
bios.info()

# %%
# Find records of people who didn't compete in Olympic Games
# These may need to be filtered out or handled separately
bios.loc[~bios['Roles'].str.contains('Competed in Olympic Games')]

# %%
# Identify edge cases: people who didn't compete but also didn't fall into excluded categories
# This reveals data quality issues that need special handling
bios.loc[(
    ~bios['Roles'].str.contains('Competed in Olympic Games')) 
    & (~bios['Roles'].str.contains('Non-starter')) 
    & (~bios['Roles'].str.contains('Intercalated Games'))
    & (~bios['Roles'].str.contains('Youth Olympic Games'))
    ]

# %%
# Sample records with titles to understand data patterns
# Helps determine how to handle or preserve this information
bios.loc[bios['Title(s)'].notna()].sample(10)

# %%
# Examine nationality data distribution and format
# Important for deciding data retention strategy
bios.loc[bios['Nationality'].notna()].sample(10)

# %% [markdown]
# ### `results` Dataset Analysis

# %%
# Review the structure of competition results data
# Look for parsing needs and data consistency issues
results.head(15)

# %%
# Assess data quality: types, missing values, structure
results.info()

# %%
# Check nationality information in results
# Determine if this duplicates bios data or provides additional value
results.loc[results['Nationality'].notna()]

# %%
# Example of how athlete competition data is structured
# Shows the relationship between bios and results datasets
results.loc[results['athlete_id']==98904]

# %% [markdown]
# ## Data Cleaning Strategy
#  
# Based on the exploration above, here's the comprehensive cleaning plan:
#  
# ### `bios` Dataset Transformations:
#  
# 1. **Birth Information (`Born` column):**
#    - Split into separate columns: `born_date`, `born_city`, `born_region`, `born_country`
# 
# 2. **Death Information (`Died` column):**
#    - Extract `died_date` (other death information will be discarded for this use case)
# 
# 3. **Name Standardization:**
#    - Use only `Used name` column, rename to `name`
#    - Remove the "•" character separator between first and last names
#    - Drop redundant name columns: `Full name`, `Original name`, `Name order`, `Other names`
# 
# 4. **Physical Measurements (`Measurements` column):**
#    - Split into `height_cm` and `weight_kg` columns
#    - Convert to numeric format and standardize units
# 
# 5. **Role Classification:**
#    - Filter to include only athletes who "Competed in Olympic Games"
#    - Exclude: Non-starters, Intercalated Games participants, Youth Olympic Games participants
#    - Create `additional_roles` column for competitors with extra responsibilities
# 
# 6. **Column Cleanup:**
#    - Drop `Nick/petnames` (high percentage of NaNs, not relevant for analysis)
# 
# ### `results` Dataset Transformations:
#  
# 1. **Column Cleanup:**
#    - Drop `Unnamed: 7` (empty column)
#    - Drop `Nationality` (will use from `bios` dataset via `athlete_id` join)
#    - Drop `As` column (redundant with `Used name`)
# 
# 2. **Data Integration:**
#    - Use `athlete_id` as primary key for joining with cleaned `bios` dataset

# %% [markdown]
# ## Cleaning Bios Data

# %% [markdown]
# ### Establishing `name` column

# %%
# Create a working copy to preserve original data (prevents name binding issues)
bios_new = bios.copy()

# %%
# Remove redundant name-related columns
# Keep only the most commonly used name format
bios_new.drop(columns=['Original name','Name order','Other names','Nick/petnames','Full name'], inplace=True)
bios_new.head()

# %%
# Standardize column naming convention
bios_new.rename(columns={'Used name': 'name'}, inplace=True)
bios_new.head()

# %%
# Replace bullet separator with standard space
# This normalizes name formatting across all records
bios_new['name'] = bios_new['name'].str.replace('•',' ')
bios_new.head()

# %% [markdown]
# ### Filtering Olympic Competitors Based on `Roles` Column

# %%
# Filter to include only athletes who competed in official Olympic Games
# Excludes non-starters, intercalated games, and youth olympics
bios_new = bios_new.loc[bios_new['Roles'].str.contains('Competed in Olympic Games')]
bios_new

# %%
# Extract additional roles beyond Olympic competition
# Clean formatting and handle cases with only Olympic competition
bios_new['additional_roles'] = (bios_new['Roles']
    .str.replace('Competed in Olympic Games','')
    .str.replace(r'^\s*•\s*', '', regex=True) # Only remove • at the start
    .replace('',np.nan) # Convert empty strings to NaN
)
bios_new.head()

# %%
# Remove original roles column as information is now split appropriately
bios_new.drop(columns='Roles', inplace=True)
bios_new.head()

# %% [markdown]
# ### Creating `died_date` Column

# %%
# Extract death date from died column
# Split on 'in ' to separate date from location information
bios_new['died_date'] = bios_new['Died'].str.split('in ').str[0]
bios_new.head()

# %%
# Validate date extraction format
bios_new.loc[bios_new['died_date'].str.match('1967', na=False)]

# %%
# Check for inconsistent formatting patterns
bios_new.loc[bios_new['died_date'].str.contains('In', na=False)]

# %% [markdown]
# **Note**: Converting `died_date` column to datetime format isn't feasible due to inconsistent date formats throughout the dataset.
# 

# %%
# Remove original died column as relevant information has been extracted
bios_new.drop(columns='Died', inplace=True)
bios_new.head()

# %% [markdown]
# ### Splitting `Born` Column

# %%
# Extract birth date using same pattern as death date
# Split on 'in ' to separate date from location
bios_new['born_date'] = bios_new['Born'].str.split('in ').str[0]
bios_new.head()

# %%
# Check for date format variations that prevent datetime conversion
bios_new.loc[bios_new['born_date'].str.contains("c. 1929", regex=False, na=False)]

# %% [markdown]
# **Note**: Converting `born_date` column to datetime format isn't feasible due to various date formats including approximate dates (e.g., "c. 1929").
# 

# %%
# Extract country information from parentheses
# Use regex to find content within parentheses, take the last match
bios_new['born_country'] = (bios_new['Born']
    .str.findall(r'\(([^)]+)\)') # Find all text in parentheses
    .str[-1])  # Get last match (in case of multiple parentheses)
bios_new.sample(10)

# %%
# Extract city name between 'in ' and first comma
bios_new['born_city'] = bios_new['Born'].str.extract(r'in ([^,]+),') # Extract text between 'in ' and first comma
bios_new.sample(10)

# %%
# Handle special cases like unknown cities marked with '?'
bios_new.loc[bios_new['Born'] == '7 May 1983 in ? (KOR)']

# %%
# Extract region/state information between first comma and parentheses
bios_new['born_region'] = bios_new['Born'].str.extract(r',\s*([^(]+)') #Extract text between first comma and opening parenthesis
bios_new.sample(10)

# %%
# Remove original born column as information has been properly split
bios_new.drop(columns='Born', inplace=True)
bios_new.head()

# %% [markdown]
# ### Splitting `Measurements` Column

# %%
# Extract height and weight from measurements string
# Split by ' / ' separator and extract numeric values
bios_new['height_cm'] = bios_new['Measurements'].str.split(' / ').str[0].str.split(' ').str[0]
bios_new['weight_kg'] = bios_new['Measurements'].str.split(' / ').str[1].str.split(' ').str[0]
bios_new.sample(10)

# %%
# Remove original measurements column as data has been split into specific columns
bios_new.drop(columns='Measurements', inplace=True)
bios_new.head()

# %% [markdown]
# ### Formatting columns

# %%
# Reorder columns for logical data organization
# Group related information together (identity, birth, death, physical, roles)
bios_new = bios_new.loc[:,['athlete_id','Sex','name','born_date','born_city','born_region','born_country','NOC','Nationality','died_date','height_cm','weight_kg','Title(s)','additional_roles','Affiliations']]
bios_new.head()

# %%
# Standardize column names to lowercase for consistency
bios_new.rename(columns={'Sex':'sex','Nationality':'nationality','Title(s)':'title(s)','Affiliations':'affiliations'}, inplace=True)
bios_new.head()

# %%
# Review final dataset structure
bios_new.info()

# %%
# Check for malformed height entries that contain commas or other formatting issues
# This identifies data quality problems from the original measurements parsing
# Example: "74," indicates incomplete or incorrectly formatted measurement data
bios_new.loc[bios_new['height_cm']=='74,']

# %%
# Convert height and weight to numeric format for mathematical operations
# errors='coerce' converts invalid/non-numeric values to NaN instead of raising errors
# This handles cases like missing data, text entries, or malformed measurements
bios_new['height_cm'] = pd.to_numeric(bios_new['height_cm'], errors='coerce')
bios_new['weight_kg'] = pd.to_numeric(bios_new['weight_kg'], errors='coerce')
bios_new.info()

# %%
# Export cleaned dataset for future use
bios_new.to_csv('bios_new.csv', index=False)

# %% [markdown]
# ## Cleaning `results` Dataset

# %% [markdown]
# ### Dropping Unnecessary Columns

# %%
# Create working copy to preserve original results data (and avoid name binding issues)
results_new = results.copy()
results_new.head()

# %%
# Remove columns that are empty, redundant, or will be sourced from bios dataset
# Unnamed: 7 is completely empty, Nationality will come from bios via athlete_id join
# As column is redundant since name on `bios` dataset exists
results_new.drop(columns=['Unnamed: 7','Nationality','As'], inplace=True)
results_new.head()

# %% [markdown]
# ### Formatting columns

# %%
# Standardize column names to lowercase for consistency with bios dataset
# Improve readability by using more descriptive names where appropriate
results_new.rename(columns = {'Games':'games','Event':'event','Team':'team','Pos':'position','Medal':'medal','Discipline':'discipline'}, inplace=True)
results_new.head()

# %%
# Reorder columns for logical organization
# Place athlete_id first as primary key, followed by competition details
results_new = results_new.loc[:,['athlete_id','NOC','games','event','team','position','medal','discipline']]
results_new.head()

# %%
# Export cleaned results dataset for analysis and potential joining with bios data
results_new.to_csv('results_new.csv', index=False)

# %% [markdown]
# ## Exploring the Cleaned Data

# %%
# Analyze gender distribution among Olympic competitors
# This provides baseline demographics for the dataset
bios_new.groupby(bios_new['sex']).count()

# %%
# Identify female athletes with official titles or honors
# Explores intersection of gender, achievement, and recognition
bios_female_title = bios_new.loc[(bios_new['sex']=='Female') & (bios_new['title(s)'].notna())].copy()
bios_female_title

# %%
# Cross-reference titled female athletes with their medal achievements
# Determines if formal recognition correlates with Olympic success
bios_female_title_results = pd.merge(left=bios_female_title, right=results_new, how='left', on='athlete_id')

bios_female_title_results.loc[bios_female_title_results['medal'].notna(),['athlete_id','name','born_date','born_country','NOC_x','nationality','NOC_y','title(s)','games','discipline','medal']]

# %%
# Investigate athletes who competed for different countries than their birth country
# This reveals patterns of athletic migration and dual citizenship
bios_results = pd.merge(left=bios_new,right=results_new, how='inner',on='athlete_id')

bios_results.loc[
    (bios_results['NOC_y'].notna()) &
    (bios_results['born_country'].notna()) &
    (bios_results['born_country'] != bios_results['NOC_y']),
    ['athlete_id','sex','name','born_date','nationality','born_country','NOC_y','NOC_x','position','discipline']].sample(10)

# %% [markdown]
# ## Data Cleaning Summary
# 
# ### Records Impact
# - **Original bios dataset**: 145,500 athletes
# - **After filtering for Olympic competitors**: 142,941 athletes retained
# - **Records removed**: 2,559 athletes (non-starters, intercalated games, youth olympics)
# - **Retention rate**: 98.2%

# %%
print(f"Original bios records: {len(bios):,}")
print(f"Cleaned bios records: {len(bios_new):,}")
print(f"Records removed: {len(bios) - len(bios_new):,}")
print(f"Retention rate: {(len(bios_new)/len(bios)*100):.1f}%")

# %% [markdown]
# ### New Columns Created
# **From `bios` dataset:**
# - `name` (standardized from `Used name`)
# - `born_date`, `born_city`, `born_region`, `born_country` (parsed from `Born`)
# - `died_date` (extracted from `Died`)
# - `height_cm`, `weight_kg` (split from `Measurements`)
# - `additional_roles` (extracted non-Olympic roles from `Roles`)
# 
# **Column transformations:**
# - Standardized column names to lowercase
# - Removed 5 redundant name columns
# - Removed 3 empty/unnecessary columns from results dataset
# 
# ### Data Quality Improvements
# - **Name standardization**: Removed bullet separators (•) for consistent formatting
# - **Role classification**: Clear separation between Olympic competitors and other participants
# - **Geographic parsing**: Structured birth location data into separate, analyzable fields
# - **Physical measurements**: Converted combined measurements into separate numeric-ready columns
# - **Data deduplication**: Removed redundant nationality information between datasets
# 
# 
# ## Limitations and Considerations
# 
# ### Date Parsing Challenges
# - **Inconsistent date formats**: Birth and death dates contain various formats including:
#   - Approximate dates (e.g., "c. 1929")
#   - Incomplete dates (year only, month/year only)
#   - Non-standard date expressions
# - **No datetime conversion**: Due to format inconsistencies, dates remain as strings
# - **Future work**: Custom date parsing functions could standardize more entries
# 
# ### Data Quality Issues Identified
# - **Incomplete measurements**: Some height/weight entries contain formatting artifacts (e.g., "74,")
# - **Missing geographic data**: Birth locations sometimes marked as "?" or incomplete
# - **Inconsistent country codes**: Athletes may have different NOC codes between datasets
# - **Special characters**: Original data contains various separators and formatting inconsistencies
# 
# ### Assumptions Made During Cleaning
# 1. **Olympic focus**: Excluded non-starters, intercalated games, and youth olympics for analysis consistency
# 2. **Name preference**: Chose `Used name` over `Full name` or `Original name` as primary identifier
# 3. **Role prioritization**: Assumed Olympic competition is primary role, others are additional
# 4. **Geographic hierarchy**: Assumed country (in parentheses) is most reliable location identifier
# 
# ### Known Data Limitations
# - **Time period coverage**: Dataset spans 1896-2022, with varying data completeness across eras
# - **Source dependency**: Data quality reflects original olympedia.org content and web scraping accuracy
# - **Missing values**: Significant gaps in biographical information, especially for early Olympic periods
# - **Country representation**: Athletes competing for different countries than birth country may indicate citizenship changes, dual nationality, or historical political changes
# 
# ### Recommendations for Further Analysis
# - Implement custom date parsing for better temporal analysis
# - Cross-validate country codes with historical Olympic participation records
# - Consider imputation strategies for missing physical measurements
# - Investigate patterns in cross-country athlete representation


