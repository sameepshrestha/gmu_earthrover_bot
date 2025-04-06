# Installation

```
conda create -n erc python=3.10
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### current progress:

- tested different depth models and output the depth results as scales
- tested given values of the depth models and compared with the real-sense depth image.