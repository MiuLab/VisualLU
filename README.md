# Visually-Enhanced Phrase Understanding
![image](https://github.com/MiuLab/VisualLU/assets/57943718/33271fab-d976-42ed-9de2-c7c87aea8471)

## Requirements
* Python >= 3.8
* `pip install -r requirements.txt`

## Data
Please refer to [here](https://drive.google.com/drive/folders/14pvnY02bVgr_X_rCRbxv1oU_eW9AY7jx?usp=drive_link).\
**Note:** the data is extracted from https://github.com/JiachengLi1995/UCTopic#datasets.

## Run
1. Generate the prompts from the raw data.
```
python3 generate_prompt.py -d ./data -t <task> -p <task>.json
```
2. Generate the images from the given prompts.
```
python3 text_to_image.py -p <task>.json -o <task>_images/ --data_json_path <task>_data.json
```
3. Run the clustering experiments.
```
python3 clustering_clip.py --data_json_path <task>_data.json -t <task>
```

## Reference
```
@inproceedings{hsu-etal-2023-visually,
    title = "Visually-Enhanced Phrase Understanding",
    author = "Hsu, Tsu-Yuan  and
      Li, Chen-An  and
      Huang, Chao-Wei  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```