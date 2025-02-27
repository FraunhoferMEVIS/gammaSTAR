Fraunhofer MEVIS does not ensure compliance to medical product regulations for the gammaSTAR sequences and tools. Hence, gammaSTAR sequences and tools are not certified as medical device and are not qualified for clinical use, but strictly for research purposes only.

Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

# About

This repository includes the sequence database used in the gammaSTAR frontend at https://gamma-star.mevis.fraunhofer.de/

# Usage of scripts

The sequence database consists of a single JSON file with all blueprints. For better versioning we publish the blueprints as separate files.

## Export
From a single database file you can export all blueprints to separate files by running 
```bash
python ./export_blueprints.py -i "all_sequences.json" -o "exported_blueprints"
```
Replace ```"all_sequences.json"``` with your database file and ```"exported_blueprints"``` with your preferred output folder.

## Import
To create a single database file from a folder of separate blueprint files, run:
```bash
python ./import_blueprints.py -i "exported_blueprints" -o "all_sequences.json"
```
Replace ```"exported_blueprints"``` with the folder you want to import and ```"all_sequences.json"``` with the single database file you want to create.