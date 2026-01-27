แก้หน้าขาวแล้่ว

## fix wk08.html 
- ใน html folder แก้ currentSlide = 20 เป็น currentSlide = 28 [html/wk08.html](./html/wk08.html)

## ผลลัพธ์การ run pycaretflow

| Code | Model | Accuracy | AUC | Recall | Prec. | F1 | Kappa | MCC | TT (Sec) |
|------|-------|----------|-----|--------|-------|-----|-------|-----|----------|
| lr | Logistic Regression | 0.7689 | 0.8047 | 0.5602 | 0.7208 | 0.6279 | 0.4641 | 0.4736 | 0.757 |
| ridge | Ridge Classifier | 0.7670 | 0.8060 | 0.5497 | 0.7235 | 0.6221 | 0.4581 | 0.4690 | 0.011 |
| lda | Linear Discriminant Analysis | 0.7670 | 0.8055 | 0.5550 | 0.7202 | 0.6243 | 0.4594 | 0.4695 | 0.008 |
| rf | Random Forest Classifier | 0.7485 | 0.7911 | 0.5284 | 0.6811 | 0.5924 | 0.4150 | 0.4238 | 0.040 |
| nb | Naive Bayes | 0.7427 | 0.7955 | 0.5702 | 0.6543 | 0.6043 | 0.4156 | 0.4215 | 0.008 |
| catboost | CatBoost Classifier | 0.7410 | 0.7993 | 0.5278 | 0.6630 | 0.5851 | 0.4005 | 0.4078 | 0.517 |
| gbc | Gradient Boosting Classifier | 0.7373 | 0.7914 | 0.5550 | 0.6445 | 0.5931 | 0.4013 | 0.4059 | 0.030 |
| ada | Ada Boost Classifier | 0.7372 | 0.7799 | 0.5275 | 0.6585 | 0.5796 | 0.3926 | 0.4017 | 0.022 |
| et | Extra Trees Classifier | 0.7299 | 0.7788 | 0.4965 | 0.6516 | 0.5596 | 0.3706 | 0.3802 | 0.036 |
| qda | Quadratic Discriminant Analysis | 0.7282 | 0.7894 | 0.5281 | 0.6558 | 0.5736 | 0.3785 | 0.3910 | 0.009 |
| lightgbm | Light Gradient Boosting Machine | 0.7133 | 0.7645 | 0.5398 | 0.6036 | 0.5650 | 0.3534 | 0.3580 | 0.049 |
| knn | K Neighbors Classifier | 0.7001 | 0.7164 | 0.5020 | 0.5982 | 0.5413 | 0.3209 | 0.3271 | 0.364 |
| dt | Decision Tree Classifier | 0.6928 | 0.6512 | 0.5137 | 0.5636 | 0.5328 | 0.3070 | 0.3098 | 0.008 |
| xgboost | Extreme Gradient Boosting | 0.6928 | 0.7571 | 0.5070 | 0.5779 | 0.5335 | 0.3068 | 0.3131 | 0.020 |
| dummy | Dummy Classifier | 0.6518 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.009 |
| svm | SVM - Linear Kernel | 0.5954 | 0.5914 | 0.3395 | 0.4090 | 0.2671 | 0.0720 | 0.0912 | 0.008 |