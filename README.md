# DeepDive: An Integrative Algorithm/Architecture Co-Design for Deep Separable Convolutional Neural Networks
![POWERED BY TeCSAR](https://raw.githubusercontent.com/TeCSAR-UNCC/Deep_RACE/master/logo/tecsarPowerBy.png)

DeepDive is a fully functional framwork for agile power efficient execution of DSCNNs. It has both algorithmic and architecture
optimization for and synthesis on the edge.

## Installation
Clone the repo
```bash
git clone https://github.com/TeCSAR-UNCC/DeepDive
```
## Prerequisites
Make sure you have the sysroot mounted on your system and update the path accordingly in the synth.sh file

### SYSROOT installation

Follow the instructions on the Xilinx wiki for SYSROOT generation 
```bash
https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18841937/Zynq+UltraScale+MPSoC+Ubuntu+part+2+-+Building+and+Running+the+Ubuntu+Desktop+From+Sources
```
Install Vivado 2018.3

### prog_fpga Tool Installation
For prog_fpga tool to program the FPGA using the teminal of peta-linux save the below file as prog_fpga in /usr/bin/
```bash
https://raw.githubusercontent.com/stanford-ppl/spatial-doc/46c40413dfae0dcbf61aaf1ed68abe10645b560d/docs/site/targets/zcu/prog_fpga
```

### Mount the SYSROOT

Before starting the synthesis we need to define the SYSROOT path inside the script.sh file. 

```bash
DIR=/mnt/500GB/shared_directory/SYSROOT/
```

There are 2 ways to mount the SYSROOT. If the ZCU102 is in the same network as the system used for sythesis. You can mount the root of the ZCU102 on the system and use that as a SYSROOT path. 

```bash
sshfs xilinx@<ip_of_ZCU102>:/ /<dir>
```
Where ip_of_ZCU102 is the IP of ZCU102 and dir is the directory where you want to mount the SYSROOT.

You can also mount the ZCU102 sdcard on the system directly.

```bash
sudo mount /dev/sdX1 /<dir> 
```

## Synthesis
Follow the below commands for running the synthesis
```bash
cd DeepDive
./synth.sh
```
## Running on ZCU102
To succesfully run the network on the ZCU102. Copy the mobileNetV2.bit (bitstream), (mobileNetV2)Executable from build/ZCU102-MobileNetV2 and the data and image folder to ZCU102

On ZCU102
```bash
sudo prog_fpga MobileNet.bit
sudo ./mobilenet image/bird.jpg
```
## Citing DeepDive
Please cite the DeepDive if it helps your research work.
```
@misc{baharani2020deepdive,
    title={DeepDive: An Integrative Algorithm/Architecture Co-Design for Deep Separable Convolutional Neural Networks},
    author={Mohammadreza Baharani, Ushma Sunil, Kaustubh Manohar, Steven Furgurson, and Hamed Tabkhi},
    year={2020},
    eprint={2007.09490},
    archivePrefix={arXiv},
    primaryClass={cs.AR}
}
```
## License
Copyright (c) 2018, the University of North Carolina at Charlotte All rights reserved. - see the [LICENSE](https://raw.githubusercontent.com/TeCSAR-UNCC/DeepDive/master/LICENSE.txt) file for details.
