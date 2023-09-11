This repository holds the files related to the kaggle competition https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data

# Sleep Onset and Wake Detection using Wrist-Worn Accelerometer Data

## Overview

This project aims to enhance the analysis of accelerometer data for sleep monitoring. It focuses on large-scale studies on sleep to better understand the environmental effects on sleep, mood, and behavior in children, particularly those with mood and behavior difficulties.

## Dataset Description

The dataset consists of around 500 multi-day recordings of wrist-worn accelerometer data. We have to detect two main event types: the onset of sleep and wake up events.

**Sleep Logbook Guidance**:

- The sleep period must be a minimum of 30 minutes.
- The sleep period can be interrupted by activity not exceeding 30 consecutive minutes.
- The sleep window should be when the watch is worn.
- Only the longest sleep window during the night is recorded.
- No more than one window should be assigned per night.
- Approximately as many nights are recorded for a series as 24-hour periods.
- Some series might have times when the device was removed; no event predictions should be made during these times.

## Repository Structure

```plaintext
.
├── data/
│   ├── raw
│   │	├── train_series.parquet
│   │	├── train_series.parquet
│   │	├── test_series.parquet
│   │	├── train_events.csv
│   │	└── sample_submission.csv
│   ├── processed
├── src/
│   ├── preprocessing/
│   ├── modeling/
│   └── evaluation/
├── notebooks/
├── results/
└── README.md
