# Shadow-Robust Crosswalk Segmentation

## 1. What we have done

- Built a three-class semantic segmentation model for **background, crosswalk, and shadow** using TorchVision's DeepLabV3 with a ResNet-50 backbone.
- Collected and annotated 210 road images in COCO format. The notebook uses 168 images for training, 21 for validation, and 21 for testing.
- Compared an untrained-on-this-dataset baseline with a model trained using cross-entropy plus Dice loss. We tuned learning rate and weight decay with six random-search trials, then trained the selected configuration for 20 epochs.
- Included the dataset, notebooks, [project report](report/DSCI6011_Final_Project_Report.pdf), and a small Gradio app for trying the trained model.

## 2. Outcomes / results

- Baseline validation mean Intersection over Union (mIoU): **0.0528**.
- Best validation mIoU in the final 20-epoch run: **0.8289** at epoch 20.
- Selected learning rate: **0.0002709**; weight decay: **0.0001**.
- The app overlays predicted crosswalk pixels in **green** and shadow pixels in **blue**.

These numbers are from the saved output in `DSCI6011 Final Project.ipynb`. The notebook does not record a final test-set score. Its train/validation/test shuffle has no fixed seed, so retraining can produce different numbers.

## 3. How to run the project

Use Python 3.10 from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio and upload a road image. The included `best_deeplab_crosswalk_shadow_fp16.pth` checkpoint is a smaller copy of the trained weights; the app loads it into a regular float32 model. A GPU is optional for inference.

To inspect or rerun training, start Jupyter with `jupyter lab` and open `DSCI6011 Final Project.ipynb`. Run it from this folder so it can find `dataset/`. Training downloads TorchVision pretrained weights and is much faster with a GPU. `DSCI6011 Project.ipynb` is the earlier baseline notebook.

The dataset's COCO metadata lists a [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
