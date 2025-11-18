# NILM Model Comparison Report

Generated on: 2025-03-22 19:12:23

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
| LSTM | 25.73 | 30.77 | 0.878 | 0.824 | 0.931 |
| GRU | 26.65 | 31.61 | 0.869 | 0.828 | 0.910 |
| TCN | 23.04 | 28.05 | 0.892 | 0.841 | 0.939 |
| LIQUID | 19.26 | 23.09 | 0.919 | 0.871 | 0.959 |
| ADVANCED_LIQUID | 17.51 | 21.22 | 0.933 | 0.895 | 0.963 |
| RESNET | 24.73 | 29.11 | 0.888 | 0.841 | 0.935 |
| TRANSFORMER | 21.27 | 25.70 | 0.901 | 0.855 | 0.944 |

### Best Performing Models

- **Mean Absolute Error (MAE)**: ADVANCED_LIQUID (17.51)
- **Root Mean Square Error (RMSE)**: ADVANCED_LIQUID (21.22)
- **F1 Score**: ADVANCED_LIQUID (0.933)
- **Precision**: ADVANCED_LIQUID (0.895)
- **Recall**: ADVANCED_LIQUID (0.963)

## Performance by Appliance

### Fridge

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 23.82 | 27.67 | 0.846 | 0.793 | 0.899 |
| GRU | 23.20 | 28.40 | 0.820 | 0.779 | 0.862 |
| TCN | 20.91 | 26.17 | 0.856 | 0.806 | 0.906 |
| LIQUID | 16.48 | 18.84 | 0.894 | 0.844 | 0.944 |
| ADVANCED_LIQUID | 15.49 | 20.02 | 0.924 | 0.920 | 0.928 |
| RESNET | 21.88 | 27.41 | 0.830 | 0.790 | 0.869 |
| TRANSFORMER | 18.64 | 23.23 | 0.862 | 0.816 | 0.908 |

### Dishwasher

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 25.57 | 29.17 | 0.875 | 0.806 | 0.944 |
| GRU | 26.23 | 31.90 | 0.880 | 0.837 | 0.923 |
| TCN | 22.92 | 27.03 | 0.885 | 0.828 | 0.943 |
| LIQUID | 19.67 | 24.01 | 0.926 | 0.888 | 0.965 |
| ADVANCED_LIQUID | 17.24 | 20.71 | 0.956 | 0.904 | 1.000 |
| RESNET | 24.21 | 25.83 | 0.905 | 0.875 | 0.935 |
| TRANSFORMER | 22.26 | 26.54 | 0.888 | 0.829 | 0.947 |

### Microwave

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 26.90 | 32.41 | 0.915 | 0.859 | 0.970 |
| GRU | 29.83 | 34.37 | 0.915 | 0.878 | 0.952 |
| TCN | 24.92 | 29.01 | 0.940 | 0.900 | 0.979 |
| LIQUID | 21.11 | 26.09 | 0.962 | 0.910 | 1.000 |
| ADVANCED_LIQUID | 18.91 | 22.29 | 0.958 | 0.912 | 1.000 |
| RESNET | 27.54 | 31.93 | 0.925 | 0.865 | 0.985 |
| TRANSFORMER | 22.72 | 26.31 | 0.961 | 0.909 | 1.000 |

### Washer_Dryer

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 31.52 | 39.77 | 0.971 | 0.933 | 1.000 |
| GRU | 32.08 | 36.64 | 0.954 | 0.911 | 0.998 |
| TCN | 27.86 | 34.80 | 0.961 | 0.899 | 1.000 |
| LIQUID | 23.48 | 28.47 | 0.979 | 0.927 | 1.000 |
| ADVANCED_LIQUID | 21.78 | 27.13 | 0.975 | 0.927 | 1.000 |
| RESNET | 30.03 | 38.33 | 0.969 | 0.933 | 1.000 |
| TRANSFORMER | 25.32 | 32.34 | 0.967 | 0.934 | 1.000 |

### Kettle

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 20.86 | 24.80 | 0.785 | 0.729 | 0.841 |
| GRU | 21.89 | 26.75 | 0.776 | 0.734 | 0.817 |
| TCN | 18.59 | 23.24 | 0.817 | 0.770 | 0.865 |
| LIQUID | 15.55 | 18.04 | 0.834 | 0.784 | 0.884 |
| ADVANCED_LIQUID | 14.13 | 15.95 | 0.851 | 0.814 | 0.888 |
| RESNET | 19.96 | 22.06 | 0.813 | 0.740 | 0.885 |
| TRANSFORMER | 17.42 | 20.08 | 0.825 | 0.787 | 0.863 |

## Focus on Liquid Neural Networks

The Liquid Neural Network architectures demonstrated several advantages over traditional neural networks for NILM tasks:

1. **Superior Overall Performance**: Liquid Neural Networks achieved the best results across most metrics, with the Advanced LNN variant showing particularly strong performance.

2. **Handling of Temporal Dynamics**: The adaptive time constants in LNNs appear well-suited for capturing the complex temporal patterns of different appliances.

3. **Consistent Performance Across Appliances**: While other models showed variable performance depending on the appliance, LNNs maintained strong results across all appliance types.

4. **Effectiveness for Both On/Off and Variable-Power Appliances**: LNNs performed well for both appliances with simple on/off patterns (like kettle) and those with variable power modes (like washing machines).

### Comparison with Traditional Models

#### LIQUID

**MAE**: LIQUID outperforms LSTM by 25.2%, GRU by 27.7%, TCN by 16.4%, RESNET by 22.1% and TRANSFORMER by 9.5%.

**RMSE**: LIQUID outperforms LSTM by 25.0%, GRU by 27.0%, TCN by 17.7%, RESNET by 20.7% and TRANSFORMER by 10.2%.

**F1**: LIQUID improves upon LSTM by 4.6%, GRU by 5.8%, TCN by 3.0%, RESNET by 3.5% and TRANSFORMER by 2.0%.

**PRECISION**: LIQUID improves upon LSTM by 5.7%, GRU by 5.2%, TCN by 3.6%, RESNET by 3.6% and TRANSFORMER by 1.9%.

**RECALL**: LIQUID improves upon LSTM by 3.0%, GRU by 5.3%, TCN by 2.1%, RESNET by 2.5% and TRANSFORMER by 1.6%.

#### ADVANCED_LIQUID

**MAE**: ADVANCED_LIQUID outperforms LSTM by 32.0%, GRU by 34.3%, TCN by 24.0%, RESNET by 29.2% and TRANSFORMER by 17.7%.

**RMSE**: ADVANCED_LIQUID outperforms LSTM by 31.0%, GRU by 32.9%, TCN by 24.4%, RESNET by 27.1% and TRANSFORMER by 17.4%.

**F1**: ADVANCED_LIQUID improves upon LSTM by 6.2%, GRU by 7.3%, TCN by 4.6%, RESNET by 5.0% and TRANSFORMER by 3.6%.

**PRECISION**: ADVANCED_LIQUID improves upon LSTM by 8.6%, GRU by 8.2%, TCN by 6.5%, RESNET by 6.5% and TRANSFORMER by 4.7%.

**RECALL**: ADVANCED_LIQUID improves upon LSTM by 3.5%, GRU by 5.8%, TCN by 2.6%, RESNET by 3.0% and TRANSFORMER by 2.1%.

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

