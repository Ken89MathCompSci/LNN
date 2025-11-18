# Cross-House Comparison Report for NILM Models

Generated on: 2025-03-22 19:22:18

## Overview

This report compares the performance of different neural network architectures for NILM across multiple houses in the UK-DALE dataset.

## Comparative Analysis

### MAE Comparison

| House | LSTM | GRU | TCN | LIQUID | ADVANCED_LIQUID | RESNET | TRANSFORMER |
|-------|------|------|------|------|------|------|------|
| House 1 | 24.4313 | 25.4103 | 21.9769 | 17.9555 | 16.9105 | 23.3890 | 20.4465 |
| House 2 | 25.7339 | 26.6470 | 23.0381 | 19.2572 | 17.5097 | 24.7265 | 21.2727 |
| House 5 | 25.7993 | 27.0160 | 23.3080 | 18.6285 | 17.5221 | 23.8807 | 20.9624 |

![MAE Comparison Heatmap]('mae_house_comparison_heatmap.png')

### RMSE Comparison

| House | LSTM | GRU | TCN | LIQUID | ADVANCED_LIQUID | RESNET | TRANSFORMER |
|-------|------|------|------|------|------|------|------|
| House 1 | 29.0767 | 30.6808 | 26.6202 | 21.5615 | 20.0696 | 27.5779 | 23.8577 |
| House 2 | 30.7664 | 31.6137 | 28.0514 | 23.0882 | 21.2183 | 29.1140 | 25.7003 |
| House 5 | 31.2371 | 31.8614 | 28.6236 | 22.4784 | 21.3733 | 29.3101 | 25.6950 |

![RMSE Comparison Heatmap]('rmse_house_comparison_heatmap.png')

### F1 Comparison

| House | LSTM | GRU | TCN | LIQUID | ADVANCED_LIQUID | RESNET | TRANSFORMER |
|-------|------|------|------|------|------|------|------|
| House 1 | 0.8579 | 0.8430 | 0.8722 | 0.9034 | 0.9227 | 0.8607 | 0.8930 |
| House 2 | 0.8783 | 0.8690 | 0.8919 | 0.9190 | 0.9329 | 0.8883 | 0.9006 |
| House 5 | 0.8825 | 0.8611 | 0.8918 | 0.9225 | 0.9358 | 0.8844 | 0.9112 |

![F1 Comparison Heatmap]('f1_house_comparison_heatmap.png')

### PRECISION Comparison

| House | LSTM | GRU | TCN | LIQUID | ADVANCED_LIQUID | RESNET | TRANSFORMER |
|-------|------|------|------|------|------|------|------|
| House 1 | 0.8097 | 0.7994 | 0.8321 | 0.8574 | 0.8655 | 0.8123 | 0.8404 |
| House 2 | 0.8241 | 0.8276 | 0.8407 | 0.8707 | 0.8953 | 0.8408 | 0.8548 |
| House 5 | 0.8372 | 0.8099 | 0.8549 | 0.8813 | 0.8890 | 0.8408 | 0.8839 |

![PRECISION Comparison Heatmap]('precision_house_comparison_heatmap.png')

### RECALL Comparison

| House | LSTM | GRU | TCN | LIQUID | ADVANCED_LIQUID | RESNET | TRANSFORMER |
|-------|------|------|------|------|------|------|------|
| House 1 | 0.9061 | 0.8867 | 0.9070 | 0.9384 | 0.9655 | 0.9053 | 0.9410 |
| House 2 | 0.9307 | 0.9104 | 0.9386 | 0.9587 | 0.9632 | 0.9349 | 0.9437 |
| House 5 | 0.9279 | 0.9095 | 0.9252 | 0.9546 | 0.9696 | 0.9179 | 0.9384 |

![RECALL Comparison Heatmap]('recall_house_comparison_heatmap.png')

## Findings

Based on the cross-house comparison, the following models show the best overall performance:

- **MAE**: ADVANCED_LIQUID
- **RMSE**: ADVANCED_LIQUID
- **F1**: ADVANCED_LIQUID
- **PRECISION**: ADVANCED_LIQUID
- **RECALL**: ADVANCED_LIQUID

## Liquid Neural Network Performance

The analysis across multiple houses confirms the strong performance of Liquid Neural Networks:

- **ADVANCED_LIQUID** performs best for MAE, RMSE, F1, PRECISION, RECALL

This multi-house evaluation reinforces the conclusion that Liquid Neural Networks are particularly well-suited for NILM tasks, demonstrating consistent performance advantages across different household environments and appliance usage patterns.

## Conclusion

The cross-house comparison validates the findings from individual house analyses, showing that model performance patterns are consistent across different households in the UK-DALE dataset. This strengthens the reliability of the model comparison results and confirms that the observed advantages of certain architectures are not specific to particular household characteristics.
