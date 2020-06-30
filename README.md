## ATPFlash

Flash object detection of ATP imaging continuous frames.

## Algorithm

Based on the gradient change in time domain.
1. Computing the gradient;
2. Divide the target according to the gradient value;
3. Denoising;
4. Convert to TXT format and save.

## Requirements

- opencv-python~=3.4.8.29
- tensorflow-gpu~=2.1.0
- numpy~=1.17.2
- matplotlib~=3.0.3
- Pillow~=6.2.1
- scikit-image~=0.16.2

## Usage

1, Download the demo data 'lps12.tif.frames.zip' from http://119.90.33.35:3557/sharing/tfC4Ibndr and extract it.

2, Install the requirements
```shell script
pip install -r requirements.txt
```

3, Example code
```python
from main import OD2

od = OD2(
        src_img_dir='lps12.tif.frames_dir',
        der_param=9,
        der_th=21,
        smooth_ker=7,
    )
od.run()
```
Parameters interpretation:
- der_param: int, (default 9), the span used to calculate the gradient;
- der_th: int, (default 21), gradient threshold, used to distinguish between signal and background;
- smooth_ker: int, (default 7), kernel of smooth operation to der imsge.

After run, the result is saved in current dir named as 'res.txt'.


## Next Plan

Speed optimization

