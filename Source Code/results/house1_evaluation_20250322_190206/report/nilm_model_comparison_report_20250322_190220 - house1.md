# NILM Model Comparison Report

Generated on: 2025-03-22 19:02:20

## Introduction

This report compares the performance of various neural network architectures for Non-Intrusive Load Monitoring (NILM) on the UK-DALE dataset. The models evaluated include:

- **LSTM**: Long Short-Term Memory networks with bidirectional layers
- **GRU**: Gated Recurrent Unit networks
- **TCN**: Temporal Convolutional Networks with dilated convolutions
- **LNN**: Liquid Neural Networks with adaptive time constants
- **Advanced LNN**: Enhanced Liquid Neural Networks with sophisticated gating mechanisms
- **ResNet**: Residual Networks with skip connections
- **Transformer**: Transformer models with self-attention mechanisms

## Summary of Findings

The following table shows the average performance of each model across all appliances:

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 24.43 | 29.08 | 0.858 | 0.810 | 0.906 |
| GRU | 25.41 | 30.68 | 0.843 | 0.799 | 0.887 |
| TCN | 21.98 | 26.62 | 0.872 | 0.832 | 0.907 |
| LIQUID | 17.96 | 21.56 | 0.903 | 0.857 | 0.938 |
| ADVANCED_LIQUID | 16.91 | 20.07 | 0.923 | 0.865 | 0.966 |
| RESNET | 23.39 | 27.58 | 0.861 | 0.812 | 0.905 |
| TRANSFORMER | 20.45 | 23.86 | 0.893 | 0.840 | 0.941 |

### Best Performing Models

- **Mean Absolute Error (MAE)**: ADVANCED_LIQUID (16.91)
- **Root Mean Square Error (RMSE)**: ADVANCED_LIQUID (20.07)
- **F1 Score**: ADVANCED_LIQUID (0.923)
- **Precision**: ADVANCED_LIQUID (0.865)
- **Recall**: ADVANCED_LIQUID (0.966)

## Performance by Appliance

### Fridge

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 21.87 | 25.64 | 0.800 | 0.741 | 0.859 |
| GRU | 22.63 | 26.64 | 0.804 | 0.745 | 0.864 |
| TCN | 19.81 | 23.96 | 0.842 | 0.793 | 0.891 |
| LIQUID | 17.09 | 21.38 | 0.861 | 0.828 | 0.895 |
| ADVANCED_LIQUID | 15.49 | 19.31 | 0.885 | 0.832 | 0.938 |
| RESNET | 19.91 | 22.86 | 0.806 | 0.762 | 0.850 |
| TRANSFORMER | 18.89 | 21.29 | 0.867 | 0.823 | 0.912 |

### Dishwasher

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 24.69 | 31.40 | 0.855 | 0.797 | 0.913 |
| GRU | 25.82 | 31.10 | 0.828 | 0.798 | 0.858 |
| TCN | 22.45 | 25.05 | 0.866 | 0.835 | 0.896 |
| LIQUID | 18.29 | 21.50 | 0.911 | 0.846 | 0.976 |
| ADVANCED_LIQUID | 17.41 | 20.34 | 0.926 | 0.850 | 1.000 |
| RESNET | 24.43 | 27.88 | 0.854 | 0.811 | 0.898 |
| TRANSFORMER | 20.44 | 24.50 | 0.895 | 0.861 | 0.929 |

### Microwave

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 26.33 | 31.19 | 0.923 | 0.874 | 0.973 |
| GRU | 27.37 | 33.93 | 0.890 | 0.843 | 0.937 |
| TCN | 24.22 | 29.22 | 0.911 | 0.885 | 0.936 |
| LIQUID | 18.84 | 23.55 | 0.954 | 0.879 | 1.000 |
| ADVANCED_LIQUID | 18.12 | 19.25 | 0.962 | 0.910 | 1.000 |
| RESNET | 25.84 | 33.31 | 0.904 | 0.865 | 0.943 |
| TRANSFORMER | 23.40 | 26.00 | 0.925 | 0.862 | 0.987 |

### Washer_Dryer

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 29.93 | 34.57 | 0.940 | 0.902 | 0.978 |
| GRU | 30.13 | 35.60 | 0.928 | 0.867 | 0.989 |
| TCN | 26.14 | 32.06 | 0.959 | 0.892 | 1.000 |
| LIQUID | 21.86 | 25.88 | 0.982 | 0.938 | 1.000 |
| ADVANCED_LIQUID | 20.43 | 25.71 | 0.987 | 0.918 | 1.000 |
| RESNET | 27.74 | 31.08 | 0.954 | 0.890 | 1.000 |
| TRANSFORMER | 22.45 | 25.51 | 0.972 | 0.922 | 1.000 |

### Kettle

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 19.34 | 22.59 | 0.771 | 0.735 | 0.807 |
| GRU | 21.10 | 26.14 | 0.765 | 0.743 | 0.786 |
| TCN | 17.27 | 22.81 | 0.783 | 0.754 | 0.812 |
| LIQUID | 13.70 | 15.49 | 0.809 | 0.796 | 0.822 |
| ADVANCED_LIQUID | 13.10 | 15.74 | 0.853 | 0.817 | 0.889 |
| RESNET | 19.03 | 22.76 | 0.785 | 0.734 | 0.836 |
| TRANSFORMER | 17.06 | 21.99 | 0.806 | 0.734 | 0.878 |

## Focus on Liquid Neural Networks

The Liquid Neural Network architectures demonstrated several advantages over traditional neural networks for NILM tasks:

1. **Superior Overall Performance**: Liquid Neural Networks achieved the best results across most metrics, with the Advanced LNN variant showing particularly strong performance.

2. **Handling of Temporal Dynamics**: The adaptive time constants in LNNs appear well-suited for capturing the complex temporal patterns of different appliances.

3. **Consistent Performance Across Appliances**: While other models showed variable performance depending on the appliance, LNNs maintained strong results across all appliance types.

4. **Effectiveness for Both On/Off and Variable-Power Appliances**: LNNs performed well for both appliances with simple on/off patterns (like kettle) and those with variable power modes (like washing machines).

### Comparison with Traditional Models

#### LIQUID

**MAE**: LIQUID outperforms LSTM by 26.5%, GRU by 29.3%, TCN by 18.3%, RESNET by 23.2% and TRANSFORMER by 12.2%.

**RMSE**: LIQUID outperforms LSTM by 25.8%, GRU by 29.7%, TCN by 19.0%, RESNET by 21.8% and TRANSFORMER by 9.6%.

**F1**: LIQUID improves upon LSTM by 5.3%, GRU by 7.2%, TCN by 3.6%, RESNET by 5.0% and TRANSFORMER by 1.2%.

**PRECISION**: LIQUID improves upon LSTM by 5.9%, GRU by 7.3%, TCN by 3.0%, RESNET by 5.6% and TRANSFORMER by 2.0%.

**RECALL**: LIQUID improves upon LSTM by 3.6%, GRU by 5.8%, TCN by 3.5% and RESNET by 3.7%.

#### ADVANCED_LIQUID

**MAE**: ADVANCED_LIQUID outperforms LSTM by 30.8%, GRU by 33.5%, TCN by 23.1%, RESNET by 27.7% and TRANSFORMER by 17.3%.

**RMSE**: ADVANCED_LIQUID outperforms LSTM by 31.0%, GRU by 34.6%, TCN by 24.6%, RESNET by 27.2% and TRANSFORMER by 15.9%.

**F1**: ADVANCED_LIQUID improves upon LSTM by 7.5%, GRU by 9.5%, TCN by 5.8%, RESNET by 7.2% and TRANSFORMER by 3.3%.

**PRECISION**: ADVANCED_LIQUID improves upon LSTM by 6.9%, GRU by 8.3%, TCN by 4.0%, RESNET by 6.5% and TRANSFORMER by 3.0%.

**RECALL**: ADVANCED_LIQUID improves upon LSTM by 6.6%, GRU by 8.9%, TCN by 6.5%, RESNET by 6.7% and TRANSFORMER by 2.6%.

## Conclusions

Based on the comprehensive evaluation across multiple appliance types and performance metrics, **Liquid Neural Networks** demonstrate superior effectiveness for the NILM task. Their ability to model continuous-time dynamics and adapt to varying temporal patterns makes them particularly well-suited for energy disaggregation.

For applications requiring the highest accuracy, the **Advanced Liquid Neural Network** variant is recommended, although it comes with increased computational requirements. For a balance of performance and efficiency, the standard **Liquid Neural Network** architecture offers strong results while maintaining reasonable training times.

## Future Work

Potential areas for future investigation include:

1. **Hybrid Architectures**: Combining Liquid Neural Networks with attention mechanisms or convolutional layers could further enhance performance.

2. **Transfer Learning**: Exploring how well Liquid Neural Network models trained on one household transfer to other households.

3. **Hyperparameter Optimization**: Systematic tuning of hyperparameters could further improve the already strong performance of LNNs.

4. **Real-time Implementation**: Adapting LNNs for real-time energy disaggregation applications, potentially with optimized, lighter variants.

5. **High-Frequency Features**: Investigating whether incorporating high-frequency electrical measurements could further enhance LNN performance for difficult appliances.

