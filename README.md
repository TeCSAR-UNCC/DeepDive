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

Follow the instructions on the Xilinx github for SYSROOT generation 
```bash
https://github.com/Xilinx/Vitis_Embedded_Platform_Source/tree/master/Xilinx_Official_Platforms/zcu102_base
```
Install Vivado 2018.3

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
## License
Copyright (c) 2018, the University of North Carolina at Charlotte All rights reserved. - see the [LICENSE](https://raw.githubusercontent.com/TeCSAR-UNCC/DeepDive/master/LICENSE.txt) file for details.
