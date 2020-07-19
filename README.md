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

# SYSROOT installation

Follow the instructions on the Xilinx github for SYSROOT generation 

https://github.com/Xilinx/Vitis_Embedded_Platform_Source/tree/master/Xilinx_Official_Platforms/zcu102_base

Install Vivado 2018.3

## Synthesis
Follow the below commands for running the synthesis
```bash
cd DeepDive
./synth.sh
```
