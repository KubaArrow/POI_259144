# Obsługa ćwiczenia 2

## Analiza chmur punktów algorytmem RANSAC oraz DBSCAN:

### a) płaszczyzna pozioma
```bash
python main.py data/plane.xyz --eps 0.3 --min_samples 10 --plane_thresh 0.02

```

### b) płaszczyzna pionowa 
```bash
python main.py data/wall.xyz --eps 0.3 --min_samples 10 --plane_thresh 0.02

```

## c) powierzchnia cylindra
```bash
python main.py data/cyl.xyz --eps 0.3 --min_samples 10 --plane_thresh 0.02
```

