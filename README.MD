__cRoss-dOmain Product rEcognition dataset (COPE)___ covers a wide range of product categories and contains over 180,000 products, corresponding to millions of short videos and live streams. It is the first dataset to cover product pages, short videos, and live streams simultaneously, providing the basis for establishing a unified product representation across different media domains.

<p align="center">
  <img width="800" height="500" src="./_images/product_pages.jpg">
</p>

<p align="center">
  <img width="800" height="500" src="./_images/short_videos.jpg">
</p>

<p align="center">
  <img width="800" height="500" src="./_images/live_clips.jpg">
</p>

__Statistics__

<p align="center">
  <img width="800" height="500" src="./_images/statistics.jpg">
</p>
  
The ROPE dataset comprises 3,056,624 product pages, 5,867,526 short videos, and 3,495,097 live streams associated with 189,958 products. We divided the ROPE dataset into train and test sets. 

The train set has 187,431 products with 3,025,236 product pages, 5,837,716 short videos, and 3,464,116 live streams. While the test set contains 2,527 products, with 31,388 product pages, 29,810 short videos, and 30,981 live streams.


## Download 
We organize the url data of the dataset into nine *txt* files for training set and *txt* files for test set. 
- training_product_pages_pair.txt : the product pages and correspond products.
- training_product_videos_pair.txt : the short videos and correspond products.
- training_product_lives_pair.txt : the live clips and correspond products.
- training_page_titles.txt : the title text of product pages.
- training_video_asrs.txt: the ASR info of short videos.
- training_live_asrs.txt: the ASR info of live clips.
- training_page_images.txt : the main image url of product pages.
- training_video_frames.txt: the frame url of short video frames.
- training_live_clips.txt : the frmae url of live clip frames. 


#### url example
```bash
https://js-ad.a.yximgs.com/bs2/ad-material-video/dfc3f0ca-2a9a-11ee-afd0-507c6f3c3e56.jpg
```
An example of image url, where *https://js-ad.a.yximgs.com/bs2/ad-material-video/* is the prefix, 

*dfc3f0ca-2a9a-11ee-afd0-507c6f3c3e56* is the name of image.





### Download the url files
To download the COPE data, please first xxx.
We will email you the download link of the four url files. These *txt* file links are valid for 7 days, please download the *txt* files in time before they expire.


 
## Citation

```bibtex
@article{bai2023crossdomain,
  title={Cross-Domain Product Representation Learning for Rich-Content E-Commerce},
  author={Xuehan Bai and Yan Li and Yanhua Cheng and Wenjie Yang and Quan Chen and Han Li},
  journal={arXiv preprint arXiv:2308.05550},
  year={2023}
}
```