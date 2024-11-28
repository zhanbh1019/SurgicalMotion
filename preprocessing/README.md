# Data processing

This README file contains instructions to compute and process RAFT optical flows and LoFTR for optimizing SurgicalMotion.

## Data format
The input video data should be organized in the following format:
```
├──sequence_name/
    ├──color/
        ├──00000.jpg
        ├──00001.jpg
        .....
    ├──mask
        ├──00000.png
        ├──00001.png
        ..... 
```

## Preparation
The command below moves files to the correct locations and download pretrained models (this only needs to be run once).
```
cd preprocessing/  

mv exhaustive_raft.py filter_raft.py chain_raft.py RAFT/;
cd RAFT; ./download_models.sh; cd ../

mv extract_dino_features.py dino/
```

## Computing and processing flow

Run the following command to process the input video sequence. Please use absolute path for the sequence directory.
```
conda activate surgicalmotion
python main_processing.py --data_dir <sequence directory> --chain
```
The processing contains several steps:
- computing all pairwise optical flows using `exhaustive_raft.py`
- computing dino features for each frame using `extract_dino_features.py`
- filtering flows using cycle consistency and appearance consistency check using`filter_raft.py`
- (optional) chaining only cycle consistent flows to create denser correspondences using `chain_raft.py`. 
  We found this to be helpful for handling sequences with rapid motion and large displacements. 
  For simple motion, this may be skipped by omitting `--chain` to save processing time. 


## Computing LoFTR
Run the following command to compute LoFTR.
```
cd LoFTR
python LoFTR_demo.py --data_dir {sequence_directory} 
```

After processing the folder should look like the following:
```
├──sequence_name/
    ├──color/
    ├──mask/ (optional; only used for visualization purposes)
    ├──count_maps/
    ├──features/
    ├──raft_exhaustive/
    ├──raft_masks/
    ├──flow_stats.json
    ├──LoFTR_exhaustive
```
