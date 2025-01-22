# hl
wip

1) run on host in case of hsa out of resources error
```
sudo sysctl -w vm.max_map_count=262144
```
2) `pci=assign-busses` kernel parameter breaks amdgpu driver on some motherboards
