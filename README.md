# Physics-Informed Neural Networks for the Harmonic Oscillator

This repository contains hte code implementation of the PINN experiments performed inn the paper Physics-Informed Neural Networks for the Harmonic Oscillator.

---

## 📑 Contents

- [Requirements](#requirements)  
- [Installation and Use](#installation-and-use)  
- [Data](#data)  
- [Acknowledgments](#acknowledgments)  
- [Citation](#citation)  
- [License Information](#license-information)  

---

## ⚙️ Requirements

To run this project, you need the following dependencies:

- Python ≥ 3.8  
- [PyTorch](https://pytorch.org/) ≥ 1.10  
- NumPy ≥ 1.21  
- SciPy ≥ 1.7  
- Matplotlib ≥ 3.4  
- tqdm (for progress bars)  
- Optionally: CUDA-compatible GPU for faster training

You can install them via `pip` (see below).

---

## 🚀 Installation and Use

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

The main programas ar armonic_oscillator_main and armonic_oscillator_advanced. In the first program, which is commented in depth, changing the conditions of the problems, that defines the type of oscillator you want to solve, you can obtain the expected result and save the image. In physics the conditions of the problems determine the problem itself. In the armonic_oscillator_advanced is only taken into account the forced oscillator. The other programms are POCs of other problems which can be examinated and proved too.

---

📊 Data
The models in this project are trained on synthetic data generated from analytical expressions of harmonic oscillators.
These include initial conditions and uniformly sampled time values in a 1D domain.
No external datasets are required.

📁 If needed, pre-generated .npy files or .csv datasets can be included under a /data/ folder.

---

🙌 Acknowledgments
This research is associated with the following project:

Federated Serverless Architectures for Heterogeneous High-Performance Computing in Smart Manufacturing
🔗 View project on AIA Group website

---

📄 License Information
This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the LICENSE file for details.

🔗 Learn more about the GPL-3.0 License

---

Made with ❤️ by David Muñoz Hernández.
