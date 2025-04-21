# FedTPS: Traffic Pattern Sharing for Federated Traffic Flow Prediction with Personalization

PyTorch implementation of "Traffic Pattern Sharing for Federated Traffic Flow Prediction with Personalization" (ICDM 2024). 


## Requirement
- Python 3.8.8 
- PyTorch 1.9.1
- Pytorch_wavelets 1.3.0
- METIS (only for graph partition) 


## Datasets
Download from the Google Drive (https://drive.google.com/file/d/1v4zeNmUXbahzle7CnFwH7YSr2T1buLo-/view?usp=sharing) and then unzip it.

Place the `datasets` folder in the same path as `README.md`

## Run Experiments
```bash
python main.py --dataset PEMS03 --batch_size 128 --mode fedtps --num_client 4
```

## Hyperparameters Configuration
| Hyperparameters               | Values |
|-------------------------------|--------|
| hidden feature dimension        | 64     |
| learning rate                 | 0.01  |
| batch size                              | 128     |
| number of global epochs        | 200    |



## Citation
If you found our code or our paper useful in your work, please cite our work. Thank you very much!

```
@INPROCEEDINGS{10884295,
  author={Zhou, Hang and Yu, Wentao and Wan, Sheng and Tong, Yongxin and Gu, Tianlong and Gong, Chen},
  booktitle={2024 IEEE International Conference on Data Mining (ICDM)}, 
  title={Traffic Pattern Sharing for Federated Traffic Flow Prediction with Personalization}, 
  year={2024},
  volume={},
  number={},
  pages={639-648},
  keywords={spatial-temporal data;traffic flow prediction;personalized federated learning},
  doi={10.1109/ICDM59182.2024.00071}}
```
