# **Optimal Transport for Multivariate Median Definition**

## 📄 **Overview**
This repository contains the results of a project focused on defining a multivariate median for datasets where the median is not explicitly known. The methodology leverages **optimal transport** in discrete and semi-discrete settings, where a known distribution (e.g., a uniformly distributed spherical distribution) is transported to the target distribution (e.g., the ANSUR II dataset).  
The results showcase detailed visualizations of transport processes, quantile contours, and algorithmic convergence.

---

## 🎯 **Objective**
The main goals of this project are:
1. **Define a robust multivariate median** for datasets lacking a direct median representation.
2. Utilize **optimal transport theory** to compute solutions for discrete and semi-discrete cases.
3. Analyze and visualize the resulting transports and their implications.

---

## 🌍 **Context**
This work was conducted during a two-month internship (June–July 2024) within the **Image Optimisation et Probabilities Team** at the **Institut de Mathématiques de Bordeaux**, supervised by **Professor Jérémie BIGOT**.

---

## 📁 **Repository Structure**
```plaintext
PY-Optimal-Transport-Median
├── README.md : Overview and usage instructions.
├── docs/ : Reference materials (papers, reports, etc.).
├── data/ : Dataset used (ANSUR II), accompanied by descriptive analysis.
├── src/ : Jupyter notebooks and Python scripts.
│   ├── utils.py : Reusable utility functions.
│   └── notebooks/ : Four Jupyter notebooks illustrating key processes.
├── results/ : Outputs (e.g., figures, graphs).
└── reports/ : Final documents.
    ├── internship_report.pdf
    ├── summary_note.pdf
    └── presentation_slides.pptx
```

---

## 🛠️ **Installation**
Ensure you have Python 3.8 or newer installed.

### Steps:
1. Clone this repository:
   ```bash
   git clone https://github.com/moranenzo/PY-Optimal-Transport-Median.git
   cd PY-Optimal-Transport-Median
   ```
2. Install dependencies:
   - Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # or venv\Scripts\activate on Windows
     ```
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
   > If `requirements.txt` is not available, run the following command in your notebooks:
   ```python
   !pip install numpy pandas matplotlib pot
   ```

---

## 📊 **Dataset**
The **ANSUR II** dataset, located in the `data/` folder, serves as the primary resource for this project.  
- **Source**: [ANSUR II Dataset](https://www.openicpsr.org/openicpsr/project/120028/version/V1/view).  
- **Description**: Anthropometric data from military populations. A descriptive analysis of the dataset is provided in `data/README.md`.

---

## 📈 **How to Use**
Navigate to the `src/notebooks/` directory to explore:
- **Detailed visualizations**:
  - Distributions of the data.
  - Optimal transport processes between measures.
  - Quantile contours of target distributions.
- **Step-by-step guides** for:
  - Multivariate median computation.
  - Transport map visualizations.

To run a notebook:
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open the desired notebook from `src/notebooks`.

---

## 💡 **Future Enhancements**
1. **Optimize computations**:
   - Implement parallelized algorithms or GPU-based libraries (e.g., **OT-GPU**).
2. **Expand applications**:
   - Test the method on more complex distributions or larger datasets.
   - Add interactive tools for experimenting with different source and target distributions.
3. **Dynamic documentation**:
   - Introduce an interactive dashboard using **Streamlit** or **Plotly Dash**.
