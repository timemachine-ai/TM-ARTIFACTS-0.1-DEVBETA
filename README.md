<h1> TimeMachine ARTIFACTS V0.1 DEV BETA </h1>

</div>



<div align="center">
  <video src="https://github.com/user-attachments/assets/b1d6dddf-4185-492d-b804-47d3d949adb5" width="70%"> </video>
</div>

## 🎨 Qualitative Performance

![Qualitative Results](assets/Demo.png)





## 📊 Quantitative Performance
![Quantitative Results](assets/quantitative.png)


## 🎮 Model Zoo


| Resolution | Parameter| Text Encoder | VAE | Download URL  |
| ---------- | ----------------------- | ------------ | -----------|-------------- |
| 1024       | 2.6B             |    [Gemma-2-2B](https://huggingface.co/google/gemma-2-2b)  |   [FLUX-VAE-16CH](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main/vae) | [hugging face](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) |

## 💻 Finetuning Code
### 1. Create a conda environment and install PyTorch
```bash
conda create -n Lumina2 -y
conda activate Lumina2
conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```
### 2.Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Install flash-attn
```bash
pip install flash-attn --no-build-isolation
```
### 4. Prepare data
You can place the links to your data files in `./configs/data.yaml`. Your image-text pair training data format should adhere to the following:
```json
{
    "image_path": "path/to/your/image",
    "prompt": "a description of the image"
}
```
### 5. Start finetuning
```bash
bash scripts/run_1024_finetune.sh
```
## 🚀 Inference Code
We support multiple solvers including Midpoint Solver, Euler Solver, and **DPM Solver** for inference.
> [!Note]
> Both the Gradio demo and the direct inference method use the .pth format weight file, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LQLh9CJwN3GOkS3unrqI0K_q9nbmqwBh?usp=drive_link).

> [!Note]
> You can also directly download from [huggingface](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0/tree/main). We have uploaded the .pth weight files, and you can simply specify the `--ckpt` argument as the download directory.
- Gradio Demo
```python   
python demo.py \
    --ckpt /path/to/your/ckpt \
    --res 1024 \
    --port 12123
``` 


- Direct Batch Inference
```bash
bash scripts/sample.sh
```

## Citation

If you find the provided code or models useful for your research, consider citing them as:

```bib
@misc{lumina2,
    author={Qi Qin and Le Zhuo and Yi Xin and Ruoyi Du and Zhen Li and Bin Fu and Yiting Lu and Xinyue Li and Dongyang Liu and Xiangyang Zhu and Will Beddow and Erwann Millon and Victor Perez,Wenhai Wang and Yu Qiao and Bo Zhang and Xiaohong Liu and Hongsheng Li and Chang Xu and Peng Gao},
    title={Lumina-Image 2.0: A Unified and Efficient Image Generative Framework},
    year={2025},
    eprint={2503.21758},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/pdf/2503.21758}, 
}
```


