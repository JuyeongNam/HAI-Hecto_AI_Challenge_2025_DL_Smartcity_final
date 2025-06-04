# HAI-Hecto_AI_Challenge_2025_DL_Smartcity_final
## **Introduction to the competition**
### *Overview*
- Competition Name: HAI! - Hecto AI Challenge (1st Half of 2025 Recruitment AI Competition)
- Hosted by: Hecto, in collaboration with DACON (Korea’s AI competition platform)
- Purpose:
    - Solve real-world AI problems using image data
    - Identify and recruit top AI talent (prize winners get interview opportunities and can join Hecto’s talent pool)
      
### *Main Task*
> Car Type Classification from Used Car Images
Participants are asked to build an AI model that classifies the **car type (class)** from an image of a used vehicle.
- This is a **multi-class image classification task**
- There are **396 unique car types (classes)**
- Practical applications: used car marketplaces, parking systems, vehicle security, etc.

### *Dataset Information*

#### 📁 `train/` Folder
- Contains 396 subfolders, one for each class
- Each folder includes labeled `.jpg` images
- A total of **33,137 training images**
- Some train data contain noise data

#### 📁 `test/` Folder
- Contains **8,258 unlabeled images** used for evaluation

#### 📄 `test.csv`
- Metadata file for the test set
- Columns:
    - `ID`: Unique image ID
    - `img_path`: Path to the image file
    - 
### *Understanding tasks*
- Frameworks: Pytorch, Tensorflow
- Pre-trained models are available, but we have to check the licence for the model and dataset before using them.
- Result submission: Submit one labelling result with Softmax applied to the prediction result.
- Data handling: Noise in the data can be removed. However, it is up to the participants to determine whether this will weaken the model's robustness.

### *Competition Schedule*
|Phase	| Date |
|---|---|
|Competition Start|May 19, 2025|
|Team Merge Deadline|June 9, 2025|
|Competition End	|June 16, 2025|
|Final Code & Slides Submission	|June 19, 2025|
|Code Review & Evaluation	|June 27, 2025|
|Awards Ceremony	|July 4, 2025|
