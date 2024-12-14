<a id="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/moranenzo/PY-Optimal-Transport-Median">
    <img src="social_preview.png">
  </a>

<h1 align="center">Optimal Transport for Multivariate Median Definition</h3>

  <p>
    Define a multivariate median for datasets where the median is not explicitly known.
    <br />
    <a href="https://github.com/moranenzo/PY-Optimal-Transport-Median"><strong>Explore the docs »</strong></a>
    <br />
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">📄 About The Project</a>
      <ul>
	<li><a href="#objective">🎯 Objective</a></li>
	<li><a href="#context">🌍 Context</a></li>
        <li><a href="#repository-structure">📁 Repository Structure</a></li>
        <li><a href="#dataset">📊 Dataset</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">🚀 Getting Started</a>
      <ul>
        <li><a href="#installation">🛠️ Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">📈 Usage</a></li>
    <li><a href="#contact">📞 Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
<h1 id="about-the-project">📄 About the project</h1>

<p>
This repository contains the results of a project focused on defining a multivariate median for datasets where the median is not explicitly known. The methodology leverages optimal transport in discrete and semi-discrete settings, where a known distribution (e.g., a uniformly distributed spherical distribution) is transported to the target distribution (e.g., the ANSUR II dataset).
The results showcase detailed visualizations of transport processes, quantile contours, and algorithmic convergence.
</p>


<!-- OBJECTIVE -->
<h2 id="objective">🎯 Objective</h2>
<p>The main goals of this project are:</p>
<ol>
  <li><strong>Define a robust multivariate median</strong> for datasets lacking a direct median representation.</li>
  <li>Utilize <strong>optimal transport theory</strong> to compute solutions for discrete and semi-discrete cases.</li>
  <li>Analyze and visualize the resulting transports and their implications.</li>
</ol>


<!-- CONTEXT -->
<h2 id="context">🌍 Context</h2>
<p>
This work was conducted during a two-month internship (June–July 2024) within the <strong>Image Optimisation et Probabilities Team</strong> at the <strong>Institut de Mathématiques de Bordeaux</strong>, supervised by <strong>Professor Jérémie BIGOT</strong>.
</p>


<!-- REPOSITORY STRUCTURE -->
<h2 id="repository-structure">📁 Repository Structure</h2>
<pre>
PY-Optimal-Transport-Median
├── README.md        # Overview and usage instructions.
├── social_preview.png        # Social Preview of the repo.
├── docs/        # Reference materials (papers, reports, etc.).
├── data/        # Dataset and its detailed analysis.
│   ├── analysis/        # Descriptive analysis of the database variables.
│   └── raw/        # Database used (ANSUR II Male and Female)
├── src/        # Jupyter notebooks and Python scripts.
│   ├── utils.py        # Reusable utility functions.
│   └── notebooks        # Four Jupyter notebooks illustrating key processes.
├── results/        # Outputs (e.g., figures, graphs).
└── reports/        # Final documents.
    ├── internship_report.pdf
    ├── summary_note.pdf
    └── presentation_slides.pptx
</pre>


<!-- DATASET -->
<h2 id="dataset">📊 Dataset</h2>
<p>The <strong>ANSUR II</strong> dataset, located in the <code>data/raw</code> folder, serves as the primary resource for this project.</p>

<ul>
  <li><strong>Source</strong>: <a href="https://www.openicpsr.org/openicpsr/project/120028/version/V1/view">ANSUR II Dataset</a></li>
  <li><strong>Description</strong>: Anthropometric data from military populations. A descriptive analysis of the dataset is provided in <code>data/README.md</code>.</li>
</ul>



<!-- GETTING STARTED -->
<h1 id="getting-started">🚀 Getting Started</h1>


<!-- INSTALLATION -->
<h3 id="installation">🛠️ Installation</h3>
<ol>
  <li>Clone the repo:
    <pre><code>git clone https://github.com/moranenzo/PY-Optimal-Transport-Median.git</code></pre>
  </li>
  <li>Navigate to the project directory:
    <pre><code>cd PY-Optimal-Transport-Median</code></pre>
  </li>
</ol>



<!-- USAGE -->
<h1 id="usage">📈 Usage</h1>
<p>Navigate to the <code>src</code> directory to explore:</p>
<ul>
  <li><strong>Detailed visualizations</strong>:
    <ul>
      <li>Distributions of the data.</li>
      <li>Optimal transport processes between measures.</li>
      <li>Quantile contours of target distributions.</li>
    </ul>
  </li>
  <li><strong>Step-by-step guides</strong> for:
    <ul>
      <li>Multivariate median computation.</li>
      <li>Transport map visualizations.</li>
    </ul>
  </li>
</ul>

<p>To run a notebook:</p>
<ol>
  <li>Start Jupyter Notebook:</li>
</ol>

<pre><code>jupyter notebook
</code></pre>

<ol start="2">
  <li>Open the desired notebook from <code>src</code>.</li>
</ol>



<!-- CONTACT -->
<h1 id="contact">📞 Contact</h1>
<p>
  <strong>Enzo MORAN</strong> - <a href="https://www.linkedin.com/in/moranenzo/" target="_blank">LinkedIn</a> - <a href="mailto:enzo.moran@ensae.fr">enzo.moran@ensae.fr</a></li>
</p>

<p>Project Link: <a href="https://github.com/moranenzo/PY-Music-Genre-Classifier" target="_blank">https://github.com/moranenzo/PY-Music-Genre-Classifier</a></p>



<p align="right">(<a href="#readme-top">back to top</a>)</p>
