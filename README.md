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
## Citing Deep RACE
Please cite the Deep RACE if it helps your research work.
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
