<h1 aligh="center">Physics-Informed Neural Networks for the Harmonic Oscillator</h1>
<p align="center">Physics-Informed Neural Networks for the Harmonic Oscillator</p>

This repository contains hte code implementation of the PINN experiments performed inn the paper Physics-Informed Neural Networks for the Harmonic Oscillator.

## Contents
The repository consists of scripts made in python to solve differential equations using PINNs. Armonic Oscillator has been solved in different use cases. The different scripts are used to solve different type of oscillators. Each script is properly commented to follow the steps to solve the equations and to get started in the PINNs world. The main scripts are armonic_oscillator_main and armonic_oscillator_advanced. In the first program, which is commented in depth, changing the conditions of the problems, that defines the type of oscillator you want to solve, you can obtain the expected result and save the image. In physics the conditions of the problems determine the problem itself. In the armonic_oscillator_advanced is only taken into account the forced oscillator. The other programms are POCs of other problems which can be examinated and proved too.

## Requirements
To run this project, you need the following dependencies (as well as an IDE like VsCode):

- Python ≥ 3.8  
- [PyTorch](https://pytorch.org/) ≥ 1.10  
- NumPy ≥ 1.21  
- SciPy ≥ 1.7  
- Matplotlib ≥ 3.4  
- tqdm (for progress bars)  
- Optionally: CUDA-compatible GPU for faster training

## Installation and use
You can install them via `pip` (see below).

Follow these steps to set up and run the project locally.

### 1. Clone the repository

```
git clone https://github.com/yourusername/pinns-harmonic-oscillator.git
cd pinns-harmonic-oscillator
```
### 2. Create a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate         # On Windows: venv\\Scripts\\activate
```

### 3. Install the required packages
Install the updated versions of the packages to avoid depency problems.
```
pip install torch numpy scipy matplotlib tqdm scipy
```

## Data
The models in this project are trained on synthetic data generated from analytical expressions of harmonic oscillators.
These include initial conditions and uniformly sampled time values in a 1D domain.
No external datasets are required.

## Acknowledgements
This research has been performed for the research project <a href="https://aia.ua.es/en/proyectos/federated-serverless-architectures-for-heterogeneous-high-performance-computing-in-smart-manufacturing.html" target="_blank">Federated Serverless Architectures for Heterogeneous High Performance Computing in Smart Manufacturing</a>, at the Applied Intelligent Architectures Research Group of the University of Alicante (Spain).

Grant <b>Serverless4HPC PID2023-152804OB-I00</b> funded by MICIU/AEI/10.13039/501100011033 and by ERDF/EU.

## Citation
```bibtex
@article{munoz_david_PINN_2025,
	title = {Physics-Informed Neural Networks for the Harmonic Oscillator,
	issn = {},
	journal = {},
	author = {e},
	year = {2025},
	pages = {},
	note = {in press},
}
```

## License Information
This project is licensed under the <a href="LICENSE.txt">GPL-3 license</a>.
