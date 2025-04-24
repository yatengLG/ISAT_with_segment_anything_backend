<h1 align='center'>ISAT_with_segment_anything_backend</h1>
<h2 align='center'>Support <a href="https://github.com/yatengLG/ISAT_with_segment_anything">ISAT</a> to use remote server for SAM encoding</h2>
    <p align="center">Deploy <b>isat-sam-backend</b> to a remote server. </p>
<p align="center">Run <b>isat-sam</b> on the local machine and connect to the remote server to perform sam encoding. </p>

# Use

## install
- Create a conda environment(recommended, optional)
```shell
# create environment
conda create -n isat_backend_env python=3.8

# activate environment
conda activate isat_backend_env
```

- Install
```shell
pip install isat-sam-backend
```

- Run
```shell
# default model: mobile_sam.pt
# default host: 127.0.0.1 
# default port: 8000

isat-sam-backend --checkpoint [model_name] --host [ip] --port [port]
```

## Model manage
- list model
```shell
isat-sam-backend model --list
```

- download model
```shell
isat-sam-backend model --download [model_name]
```

- remove model
```shell
isat-sam-backend model --remove [model_name]
```

