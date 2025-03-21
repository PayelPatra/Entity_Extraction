# Medical Entity Extraction of Cancer Data:
Our Paper describing about The transformer model how they are extracting medical entities related to Cancer Data. Here both Pre-Trained and Fine-Tuned models are used for extraction medical entities.
1. BioBERT
2. BioClinicalBERT
3. PubMedBERT
4. BlueBERT
5. RoBERTa
   
# Dataset Availability:
The data is collected from the paper details below:
*Paper Title :* "Coral: expert-curated oncology reports to advance language model inference"
*Dataset Citation:* Sushil, M., Kennedy, V., Mandair, D., Miao, B., Zack, T., & Butte, A. (2024). CORAL: expert-Curated medical Oncology Reports to Advance Language model inference (version 1.0). PhysioNet. https://doi.org/10.13026/v69y-xa45.
*GitHub resource:* https://github.com/MadhumitaSushil/OncLLMExtraction/blob/main/README.md

# PhysioNet Citation:
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

# License:
The code and annotation schema is shared under Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) Further details can be found on this page. Additionally, the dataset derived from this schema is shared under the PhysioNet Credentialed Health Data License 1.5.0, which is intended to be used only within non-commercial, sharealike setups similar to the CC BY-NC-SA license.

For more information on PhysioNet, visit: http://mimic.physionet.org/



## How to Add Data 
1. Place your input `.txt` files inside a `data/` folder. 
2. Run any model script inside `scripts/` to extract entities. 
3. The output CSV files will be saved inside `output/`. 
