# challengerai-mlsv2018

The docker has to have following structure:

```
.
├── Dockerfile
├── README.md
├── __init__.py
├── infer
│   ├── __init__.py
│   └── infer.py
└── requirements.txt
```

## Steps to build

Build docker

```
nvidia-docker build -t demo_pytorch -f Dockerfile .
```

Run docker

```
nvidia-docker run -it demo_pytorch /bin/bash
```

You also can try to build docker with GPU video decoding

```
nvidia-docker build -t docker_steroids -f Dockerfile.steroids .
```

Run inference

```
/usr/bin/python /data/data/run.py
```

## Requirements

* docker
* nvidia-docker