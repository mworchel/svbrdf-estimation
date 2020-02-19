@echo off

set tub_user=%1
set out_dir="./../multiImage_pytorch/remote_model"

mkdir %out_dir%
scp %tub_user%@gateway.hpc.tu-berlin.de:~/svbrdf-estimation/development/scripts/SVBRDF*.* %out_dir%
scp -rp %tub_user%@gateway.hpc.tu-berlin.de:~/svbrdf-estimation/development/multiImage_pytorch/models/* %out_dir%
scp -r -d %tub_user%@gateway.hpc.tu-berlin.de:~/svbrdf-estimation/development/multiImage_pytorch/logs %out_dir%
