# Bharatanatyam Mudra Recognition

![Mudra Demo](mudra.gif)

## A computer vision pipeline to recognize  **50** Bharatanatyam hand gestures (mudras) using MediaPipe and Machine Learning.

---

> [!warning]
> Important: This project requires Python 3.10
It will NOT work with the latest Python versions (like 3.13 / 3.14) due to MediaPipe compatibility issues.

## Quick Start

### 1. Setup

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\\Scripts\\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Train the model from csv provided(wont take too much time;)

```bash
python train.py
```


### 3. Run Real-Time Detection

```bash
python live.py
```
---

## Available Mudras

| #  | Asamyuta Hastas      | #  | Samyuta Hastas     |
|----|----------------------|----|--------------------|
| 1  | Pathaka              | 1  | Anjali             |
| 2  | Tripathaka           | 2  | Kapota             |
| 3  | Ardhapathaka         | 3  | Karkata            |
| 4  | Kartarimukha         | 4  | Swastika           |
| 5  | Mayura               | 5  | Pushpaputa         |
| 6  | Ardhachandra         | 6  | Shivalinga         |
| 7  | Arala                | 7  | Katakavardhana     |
| 8  | Shukatunda           | 8  | Kartariswastika    |
| 9  | Mushti               | 9  | Shakata            |
| 10 | Shikhara             | 10 | Shankha            |
| 11 | Kapittha             | 11 | Chakra             |
| 12 | Katakamukha (1,2,3)  | 12 | Samputa            |
| 13 | Suchi                | 13 | Pasha              |
| 14 | Chandrakala          | 14 | Kilaka             |
| 15 | Padmakosha           | 15 | Matsya             |
| 16 | Sarpasirsha          | 16 | Kurma              |
| 17 | Mrigasirsha          | 17 | Varaha             |
| 18 | Simhamukha           | 18 | Garuda             |
| 19 | Kangula              | 19 | Nagabandha         |
| 20 | Alapadma             | 20 | Khatva             |
| 21 | Chatura              | 21 | Bherunda           |
| 22 | Bhramara             |    |                    |
| 23 | Hamsasya             |    |                    |
| 24 | Hamsapaksha          |    |                    |
| 25 | Mukula               |    |                    |
| 26 | Tamrachuda           |    |                    |
| 27 | Trishula             |    |                    |

---

## If you want to build from scratch

### 1. Download Dataset

[https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset](https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset)

Place it inside project root.

### 2. Create Feature Dataset (takes like 20 mins)

```bash
python create_data.py
```

### 3. Train Model

```bash
python train.py
```
### 4. Run Real-Time Detection

```bash
python live.py
```





---

## Project Structure

```
.
├── create_data.py
├── train.py
├── live.py
├── mudra_model.pkl
├── label_encoder.pkl
├── mudra_keypoints_2hands.csv
├── requirements.txt
└── README.md
```

---

## To learn and understand how the system works, please refer to working.md

---

## Troubleshooting

Install missing modules:

```bash
pip install -r requirements.txt
```

If webcam fails:

* Check permissions
* Change camera index

If model not found:

* Run train.py first

---

## Credits

This project uses the Bharatanatyam Mudra Dataset by Jisha Raj R.

Dataset Repository:
[https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset](https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset)

Dataset License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)

Changes made in this project:

* Extracted MediaPipe hand keypoints from images
* Converted keypoints into normalized feature vectors
* Trained a machine learning model on the processed dataset

This project complies with the terms of the CC BY-SA 4.0 license by providing attribution and distributing derived work under the same license.

---

## License

This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).

You are free to:

* Share — copy and redistribute the material
* Adapt — remix, transform, and build upon the material

Under the following terms:

* Attribution — You must give appropriate credit
* ShareAlike — You must distribute your contributions under the same license

Full license text:
[https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)

