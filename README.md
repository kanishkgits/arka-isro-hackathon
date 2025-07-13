![BAH_2_logo](images/BAH_logo.png)

**This project aims to solve the problem for [Identifying Halo CME Events Based on Particle Data from SWIS-ASPEX Payload onboard Aditya-L1](https://vision.hack2skill.com/event/bah2025?psNumber=ps10&scrollY=0&utm_source=hack2skill&utm_medium=homepage) stated under the [Bharitya Antariksh Hackathon 2025](https://vision.hack2skill.com/event/bah2025) Organised by INDIAN SPACE RESEARCH ORGANISATION.**

---

# 🛰️ ARKA – Halo CME Detection

![Arka_logo](images/Arka_logo.png)

**Arka** is a scientific data analysis project focused on detecting **Halo Coronal Mass Ejection (CME)** events using **solar wind particle data** collected by the **SWIS-ASPEX payload onboard the Aditya-L1 mission**. Built entirely on data collected by Indian satellites, Arka aims to contribute to an early warning system that can predict space weather events before they impact Earth and its orbiting satellites.

## 🌌 Project Motivation

The Sun plays a crucial role in driving particle flux throughout the interplanetary medium, significantly influencing our near-Earth environment. Sudden variations in this particle flux—such as those caused by **Coronal Mass Ejections (CMEs)**—can lead to severe disturbances in the upper atmosphere, potentially damaging satellites and other space-based assets. 

Arka is motivated by the need for an early warning system that can detect such events in advance. By analyzing solar wind particle data collected from a distant vantage point in space (via Indian satellites like Aditya-L1), we aim to identify early signatures of CME events—enabling timely precautions to mitigate the risk of catastrophic space weather impacts.

---

### How to Use the Project

1. **Explore the Project Overview (No Setup Required)**
   - Open the notebook **`Arka - CME Detection System`** directly on GitHub to get a complete overview of the project.
   - This is ideal if you want to quickly review the approach without setting up the environment locally.

2. **Run the Project Locally**
   - Clone the repository and ensure your environment meets the dependencies listed in `requirements.txt`.
   - Organize your data in the following directory structure:

     ```
     /data
     ├── Training_data/
     ├── Untrained_data/
     └── cmemar2025.txt  (or similar CACTus CME catalogue files)
     ```

   - Make sure to place all `.cdf` files appropriately in the `Training_data` and `Untrained_data` folders.
   - Run the notebooks or scripts to process the data and reproduce the CME detection results.

> 💡 **Tip:** If you're using a Jupyter environment like Google Colab or Jupyter Lab, make sure to adjust the data paths accordingly.


### 🛠️ Tech Stack

The following Python libraries were used to build and run the Arka CME detection system:

- **Data Handling & Processing**
  - `numpy==2.3.1`
  - `pandas==2.3.1`
  - `scipy==1.16.0`

- **Machine Learning & Model Evaluation**
  - `scikit_learn==1.7.0`
  - `imbalanced_learn==0.13.0`
  - `imblearn==0.0`
  - `joblib==1.5.1`

- **Visualization**
  - `matplotlib==3.10.3`
  - `seaborn==0.13.2`

- **Space Science Data Processing**
  - `spacepy==0.7.0` — used to handle and analyze CDF (Common Data Format) files from the Aditya-L1 SWIS-ASPEX payload.

> ⚠️ Make sure to install the dependencies using `pip install -r requirements.txt` before running the project locally.

---

## 📁 Data

The primary data source is:
- Format: `.cdf` (Common Data Format)
- Source: **SWIS-ASPEX Level-2** datasets from Aditya-L1

## 🔬 Resources

- [CDF Documentation](https://cdf.gsfc.nasa.gov/)  
  Official site for Common Data Format (CDF) documentation, libraries, and tools.

- [Particle Data – ISSDC](https://www.issdc.gov.in/)  
  Indian Space Science Data Centre – provides access to Aditya-L1 solar wind data and other space mission datasets.

- [CACTus CME Catalogue](https://www.sidc.be/cactus/catalog.php)  
  Catalogue of Coronal Mass Ejections detected by the CACTus system using LASCO data.

- [Aditya-L1 Payload Details (PDF)](https://www.issdc.gov.in/docs/Aditya/Al1_Payload.pdf)  
  Official document describing Aditya-L1's payload instruments, including SWIS-ASPEX.
