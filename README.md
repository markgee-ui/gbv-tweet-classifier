
## 🧠 GBV Tweet Classifier App

This project aims to support the fight against Gender-Based Violence (GBV) by classifying tweets into specific categories of abuse. By leveraging machine learning, this tool can help highlight patterns and raise awareness, while also informing policy makers and law enforcement.

---

## 📌 Problem Statement

One of the greatest challenges in combating GBV is the **culture of silence**, where victims fear reporting incidents due to shame, threats, or lack of trust in authorities.

This project offers a way to:

- Automatically classify tweets about GBV into five categories.
- Present trends over time and support data-driven interventions.

---

## 🗂️ Categories of GBV

Tweets are classified into the following types:

- `sexual_violence`
- `emotional_violence`
- `physical_violence`
- `economic_violence`
- `harmful_traditional_practices`

---

## 🚀 Technologies Used


| Technology     | Purpose                            |
|----------------|------------------------------------|
| Python         | Core programming language          |
| Pandas         | Data handling                      |
| scikit-learn   | Machine learning pipeline          |
| NLTK           | Natural language preprocessing     |
| Streamlit      | Frontend web app                   |
| VS Code / Jupyter | Development environment         |

---

## 📦 Installation & Setup

> You can run this on **Anaconda** or any Python 3.10+ environment.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/gbv-tweet-classifier.git
cd gbv-tweet-classifier
````

### 2️⃣ (Optional) Create a Virtual Environment

```bash
conda create -n gbv_env python=3.10
conda activate gbv_env
```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pandas scikit-learn nltk
```

> Also run this once to download stopwords:

```python
import nltk
nltk.download('stopwords')
```

---

## ▶️ Running the App

After setting up, run the Streamlit app:

```bash
streamlit run gbv_classifier_app.py
```

This will open a browser at `http://localhost:8501`.

---

## 📁 Files

| File                     | Description                   |
| ------------------------ | ----------------------------- |
| `gbv_classifier_app.py`  | Main Streamlit app script     |
| `Train.csv` & `Test.csv` | Datasets for training/testing |
| `requirements.txt`       | Dependencies list             |
| `README.md`              | This instruction file         |

---

## 🧠 Features of the Streamlit App

* Input a tweet and classify it into GBV categories
* Display a sample of predictions from test data
* Download the predictions as a CSV

---

## 📊 Output Format

The output will be a CSV like:

```
Tweet_ID       type
ID_0095QL4S    emotional_violence
ID_00E9F5X9    sexual_violence
...
```

---

## 📌 Future Enhancements

* Use transformer models like BERT for better accuracy
* Visualize trends (e.g., GBV mentions over time)
* Integrate location data (if available) to create GBV heatmaps

---

## 🧑‍💻 Author

**Ngugi Mark**
[GitHub](https://github.com/markgee-ui) • [Email](mailto:ngugimark93@email.com)

---


