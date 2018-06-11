# Data-Augmentation-For-Wearable-Sensor-Data

This is a sample code of data augmentation methods for wearable sensor data (time-series data). The example code writen in Jupyter notebook can be found [here](https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb) or [here](https://nbviewer.jupyter.org/github/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb). For more details, please refer to the the paper below.

T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks,” in Proceedings of the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 2017. New York, NY, USA: ACM, 2017, pp. 216–220. [[arXiv]](https://arxiv.org/abs/1706.00527)

## Motivation

Data augmentation is consider as a standard preprocessing in various recognition problems (e.g. image recognition), which gives additional performance improvement by providing more data. Data augmentation can be also interpreted as injecting human's prior knowledge about label-preserving transformation and giving regularization by data. This code provides a simple approach to augment time-series data, e.g., wearable sensor data, by applying various distortions to the data. 

## Data augmentation examples
Please see the jupyter notebook file ([here](https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb) or [here](https://nbviewer.jupyter.org/github/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb)) for more examples

![DA_examples](DA_examples.png)

## Dependency
You need to pre-install numpy, matplotlib, scipy, and transforms3d for running the code. (You can install them using `pip`)

## License
You can freely modify this code for your own purpose. However, please leave the citation information untouched when you redistributed the code to others. If this code helps your research, please cite the paper.

```
@inproceedings{TerryUm_ICMI2017,
 author = {Um, Terry T. and Pfister, Franz M. J. and Pichler, Daniel and Endo, Satoshi and Lang, Muriel and Hirche, Sandra and Fietzek, Urban and Kuli\'{c}, Dana},
 title = {Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks},
 booktitle = {Proceedings of the 19th ACM International Conference on Multimodal Interaction},
 series = {ICMI 2017},
 year = {2017},
 isbn = {978-1-4503-5543-8},
 location = {Glasgow, UK},
 pages = {216--220},
 numpages = {5},
 doi = {10.1145/3136755.3136817},
 acmid = {3136817},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Parkinson\&\#39;s disease, convolutional neural networks, data augmentation, health monitoring, motor state detection, wearable sensor},
} 
```
