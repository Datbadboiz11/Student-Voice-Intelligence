# Baseline Model Report

## Input

- Data: `data\processed\student_voice_enriched_reviewed.csv`
- Rows: `49141`

## Models

- TF-IDF word unigram+bigram + Logistic Regression
- TF-IDF word unigram+bigram + Linear SVM

## Best Test Results By Task

| task_name        | model_name       |   accuracy |   macro_f1 |   weighted_f1 | class_weight   |
|:-----------------|:-----------------|-----------:|-----------:|--------------:|:---------------|
| sentiment_3class | tfidf_linear_svm |   0.819409 |   0.811936 |      0.818725 | None           |
| topic_group      | tfidf_linear_svm |   0.815421 |   0.658308 |      0.815736 | balanced       |
| toxic_binary     | tfidf_linear_svm |   0.991717 |   0.901225 |      0.991321 | balanced       |
| urgency_final    | tfidf_linear_svm |   0.996421 |   0.751368 |      0.996136 | balanced       |

## All Results

| task_name        | model_name       | split      |   accuracy |   precision_macro |   recall_macro |   macro_f1 |   weighted_f1 |
|:-----------------|:-----------------|:-----------|-----------:|------------------:|---------------:|-----------:|--------------:|
| sentiment_3class | tfidf_linear_svm | test       |   0.819409 |          0.822789 |       0.803536 |   0.811936 |      0.818725 |
| sentiment_3class | tfidf_logreg     | test       |   0.815421 |          0.828076 |       0.789948 |   0.804914 |      0.813599 |
| sentiment_3class | tfidf_linear_svm | validation |   0.833674 |          0.837535 |       0.818844 |   0.827097 |      0.832919 |
| sentiment_3class | tfidf_logreg     | validation |   0.833061 |          0.847305 |       0.808152 |   0.823596 |      0.83123  |
| topic_group      | tfidf_linear_svm | test       |   0.815421 |          0.658953 |       0.658603 |   0.658308 |      0.815736 |
| topic_group      | tfidf_logreg     | test       |   0.686266 |          0.523575 |       0.704522 |   0.567046 |      0.715777 |
| topic_group      | tfidf_linear_svm | validation |   0.820376 |          0.684707 |       0.694904 |   0.689106 |      0.820116 |
| topic_group      | tfidf_logreg     | validation |   0.688625 |          0.514827 |       0.730929 |   0.564252 |      0.716685 |
| toxic_binary     | tfidf_linear_svm | test       |   0.991717 |          0.94391  |       0.866344 |   0.901225 |      0.991321 |
| toxic_binary     | tfidf_logreg     | test       |   0.754985 |          0.541971 |       0.857569 |   0.50651  |      0.840209 |
| toxic_binary     | tfidf_linear_svm | validation |   0.990794 |          0.941407 |       0.84809  |   0.888953 |      0.99025  |
| toxic_binary     | tfidf_logreg     | validation |   0.767185 |          0.543461 |       0.855529 |   0.513408 |      0.848135 |
| urgency_final    | tfidf_linear_svm | test       |   0.996421 |          0.806698 |       0.713127 |   0.751368 |      0.996136 |
| urgency_final    | tfidf_logreg     | test       |   0.70089  |          0.513579 |       0.776002 |   0.469058 |      0.82195  |
| urgency_final    | tfidf_linear_svm | validation |   0.995908 |          0.739177 |       0.63834  |   0.674528 |      0.99571  |
| urgency_final    | tfidf_logreg     | validation |   0.712357 |          0.468835 |       0.780736 |   0.443434 |      0.829223 |

## Confusion Matrices

- `outputs\figures\baseline_confusion_matrix_sentiment_3class_tfidf_logreg.png`
- `outputs\figures\baseline_confusion_matrix_sentiment_3class_tfidf_linear_svm.png`
- `outputs\figures\baseline_confusion_matrix_topic_group_tfidf_logreg.png`
- `outputs\figures\baseline_confusion_matrix_topic_group_tfidf_linear_svm.png`
- `outputs\figures\baseline_confusion_matrix_toxic_binary_tfidf_logreg.png`
- `outputs\figures\baseline_confusion_matrix_toxic_binary_tfidf_linear_svm.png`
- `outputs\figures\baseline_confusion_matrix_urgency_final_tfidf_logreg.png`
- `outputs\figures\baseline_confusion_matrix_urgency_final_tfidf_linear_svm.png`

## Classification Reports

### sentiment_3class | tfidf_logreg | validation

```text
              precision    recall  f1-score   support

    negative       0.84      0.74      0.79      1314
     neutral       0.81      0.92      0.86      2352
    positive       0.89      0.76      0.82      1222

    accuracy                           0.83      4888
   macro avg       0.85      0.81      0.82      4888
weighted avg       0.84      0.83      0.83      4888

```

### sentiment_3class | tfidf_logreg | test

```text
              precision    recall  f1-score   support

    negative       0.80      0.73      0.76      2630
     neutral       0.79      0.90      0.85      4725
    positive       0.89      0.74      0.81      2424

    accuracy                           0.82      9779
   macro avg       0.83      0.79      0.80      9779
weighted avg       0.82      0.82      0.81      9779

```

### sentiment_3class | tfidf_linear_svm | validation

```text
              precision    recall  f1-score   support

    negative       0.82      0.77      0.79      1314
     neutral       0.83      0.89      0.86      2352
    positive       0.87      0.80      0.83      1222

    accuracy                           0.83      4888
   macro avg       0.84      0.82      0.83      4888
weighted avg       0.83      0.83      0.83      4888

```

### sentiment_3class | tfidf_linear_svm | test

```text
              precision    recall  f1-score   support

    negative       0.79      0.76      0.77      2630
     neutral       0.82      0.87      0.84      4725
    positive       0.86      0.78      0.82      2424

    accuracy                           0.82      9779
   macro avg       0.82      0.80      0.81      9779
weighted avg       0.82      0.82      0.82      9779

```

### topic_group | tfidf_logreg | validation

```text
                   precision    recall  f1-score   support

      career_jobs       0.35      0.80      0.49        81
      events_news       0.50      0.56      0.53       158
       facilities       0.67      0.87      0.76        70
           others       0.74      0.73      0.73      1537
  personal_social       0.37      0.67      0.48       226
             spam       0.14      0.80      0.24        40
 student_services       0.38      0.76      0.50       305
teaching_learning       0.97      0.66      0.78      2471

         accuracy                           0.69      4888
        macro avg       0.51      0.73      0.56      4888
     weighted avg       0.79      0.69      0.72      4888

```

### topic_group | tfidf_logreg | test

```text
                   precision    recall  f1-score   support

      career_jobs       0.40      0.76      0.52       163
      events_news       0.48      0.60      0.54       316
       facilities       0.78      0.90      0.84       145
           others       0.73      0.72      0.72      3041
  personal_social       0.35      0.60      0.44       453
             spam       0.12      0.65      0.21        84
 student_services       0.36      0.74      0.48       611
teaching_learning       0.97      0.66      0.79      4966

         accuracy                           0.69      9779
        macro avg       0.52      0.70      0.57      9779
     weighted avg       0.79      0.69      0.72      9779

```

### topic_group | tfidf_linear_svm | validation

```text
                   precision    recall  f1-score   support

      career_jobs       0.56      0.63      0.59        81
      events_news       0.58      0.53      0.55       158
       facilities       0.84      0.87      0.85        70
           others       0.81      0.83      0.82      1537
  personal_social       0.55      0.50      0.53       226
             spam       0.63      0.68      0.65        40
 student_services       0.62      0.63      0.62       305
teaching_learning       0.90      0.89      0.90      2471

         accuracy                           0.82      4888
        macro avg       0.68      0.69      0.69      4888
     weighted avg       0.82      0.82      0.82      4888

```

### topic_group | tfidf_linear_svm | test

```text
                   precision    recall  f1-score   support

      career_jobs       0.61      0.63      0.62       163
      events_news       0.54      0.56      0.55       316
       facilities       0.86      0.87      0.86       145
           others       0.80      0.83      0.82      3041
  personal_social       0.51      0.46      0.48       453
             spam       0.47      0.43      0.45        84
 student_services       0.57      0.61      0.59       611
teaching_learning       0.91      0.89      0.90      4966

         accuracy                           0.82      9779
        macro avg       0.66      0.66      0.66      9779
     weighted avg       0.82      0.82      0.82      9779

```

### toxic_binary | tfidf_logreg | validation

```text
              precision    recall  f1-score   support

           0       1.00      0.76      0.86      4772
           1       0.09      0.95      0.16       116

    accuracy                           0.77      4888
   macro avg       0.54      0.86      0.51      4888
weighted avg       0.98      0.77      0.85      4888

```

### toxic_binary | tfidf_logreg | test

```text
              precision    recall  f1-score   support

           0       1.00      0.75      0.86      9549
           1       0.09      0.97      0.16       230

    accuracy                           0.75      9779
   macro avg       0.54      0.86      0.51      9779
weighted avg       0.98      0.75      0.84      9779

```

### toxic_binary | tfidf_linear_svm | validation

```text
              precision    recall  f1-score   support

           0       0.99      1.00      1.00      4772
           1       0.89      0.70      0.78       116

    accuracy                           0.99      4888
   macro avg       0.94      0.85      0.89      4888
weighted avg       0.99      0.99      0.99      4888

```

### toxic_binary | tfidf_linear_svm | test

```text
              precision    recall  f1-score   support

           0       0.99      1.00      1.00      9549
           1       0.89      0.73      0.81       230

    accuracy                           0.99      9779
   macro avg       0.94      0.87      0.90      9779
weighted avg       0.99      0.99      0.99      9779

```

### urgency_final | tfidf_logreg | validation

```text
              precision    recall  f1-score   support

        high       0.00      1.00      0.01         4
         low       1.00      0.71      0.83      4857
      medium       0.40      0.63      0.49        27

    accuracy                           0.71      4888
   macro avg       0.47      0.78      0.44      4888
weighted avg       0.99      0.71      0.83      4888

```

### urgency_final | tfidf_logreg | test

```text
              precision    recall  f1-score   support

        high       0.00      1.00      0.00         4
         low       1.00      0.70      0.82      9700
      medium       0.54      0.63      0.58        75

    accuracy                           0.70      9779
   macro avg       0.51      0.78      0.47      9779
weighted avg       1.00      0.70      0.82      9779

```

### urgency_final | tfidf_linear_svm | validation

```text
              precision    recall  f1-score   support

        high       0.50      0.25      0.33         4
         low       1.00      1.00      1.00      4857
      medium       0.72      0.67      0.69        27

    accuracy                           1.00      4888
   macro avg       0.74      0.64      0.67      4888
weighted avg       1.00      1.00      1.00      4888

```

### urgency_final | tfidf_linear_svm | test

```text
              precision    recall  f1-score   support

        high       0.50      0.50      0.50         4
         low       1.00      1.00      1.00      9700
      medium       0.92      0.64      0.76        75

    accuracy                           1.00      9779
   macro avg       0.81      0.71      0.75      9779
weighted avg       1.00      1.00      1.00      9779

```

## Notes

- Sentiment la task sach nhat va nen la baseline chinh.
- Topic dung `topic_group` de train chung NEU+UIT; neu dung `topic_std`, nen train rieng theo source.
- Toxic va urgency lech lop manh, nen doc macro-F1/recall tung lop thay vi accuracy.
- Duplicate conflict trong train duoc drop theo tung target truoc khi train.