# Research Template

The goal of the project is to analyze the existing data of the MITICA experiment and to develop data-driven models 
for discharge prediction. Electrical discharges in vacuum/low-pressure environments are one of the most critical 
restrictions in the efficiency/reliability of Neutral Beam Injectors. The data includes voltage, 
current, pressure, residual gas, X-ray dose, and energy spectra provided by our collaboration partner
Consorzio RFX. 

### Data

Download data from pernkopf@rfxssh.igi.cnr.it (/mnt/N41a/scratch/rigoni/hvptf)

### Project Organization
The following project skeleton is used for the project.

    ├── LICENSE
    │
    ├── README.md			<- The top-level README for using this project.
    │
    ├── environment.yml		<- The file for reproducing the Python environment necessary for
    │  	  	  	 		   	   running the code in this project. Generated with `conda <env> export > environment.yml`
    │
    ├── data
    │   ├── clean			<- Cleaned data sets, e.g. remove invalid/missing data,...
    │   ├── processed		<- The final, canonical data sets for modeling after any preprocessing steps.
    │   └── raw			<- The original, immutable data dump.
    │
    ├── models				<- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks			<- Jupyter notebooks (data exploration, tutorials, visualization of results, ...).
    │						   Sensible naming! No endless notebooks! Use Markdown for documentation!
    │
    ├── references			<- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── documentation       <- Folder containing documents and presentations related to the project..
    │
    ├── src					<- Source code for use in this project.
    │   ├── __init__.py		<- Makes src a Python module
    │   ├── data			<- Code related to data, e.g. for generating, loading, preprocessing...
    │   ├── models			<- Model classes or algorithms, e.g. custom NN implementations.
    │   ├── scripts			<- Scripts for training and evaluation of your models/methods.
    │   └── visualization	<- Code to create exploratory and results oriented visualizations.
    │
