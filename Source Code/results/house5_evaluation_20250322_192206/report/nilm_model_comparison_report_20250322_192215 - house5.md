# NILM Model Comparison Report

Generated on: 2025-03-22 19:22:15

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
| LSTM | 25.80 | 31.24 | 0.883 | 0.837 | 0.928 |
| GRU | 27.02 | 31.86 | 0.861 | 0.810 | 0.910 |
| TCN | 23.31 | 28.62 | 0.892 | 0.855 | 0.925 |
| LIQUID | 18.63 | 22.48 | 0.923 | 0.881 | 0.955 |
| ADVANCED_LIQUID | 17.52 | 21.37 | 0.936 | 0.889 | 0.970 |
| RESNET | 23.88 | 29.31 | 0.884 | 0.841 | 0.918 |
| TRANSFORMER | 20.96 | 25.69 | 0.911 | 0.884 | 0.938 |

### Best Performing Models

- **Mean Absolute Error (MAE)**: ADVANCED_LIQUID (17.52)
- **Root Mean Square Error (RMSE)**: ADVANCED_LIQUID (21.37)
- **F1 Score**: ADVANCED_LIQUID (0.936)
- **Precision**: ADVANCED_LIQUID (0.889)
- **Recall**: ADVANCED_LIQUID (0.970)

## Performance by Appliance

### Fridge

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 23.93 | 28.31 | 0.830 | 0.771 | 0.888 |
| GRU | 24.59 | 29.59 | 0.811 | 0.747 | 0.874 |
| TCN | 20.99 | 26.11 | 0.847 | 0.815 | 0.879 |
| LIQUID | 17.71 | 22.26 | 0.890 | 0.851 | 0.929 |
| ADVANCED_LIQUID | 15.90 | 18.87 | 0.923 | 0.882 | 0.963 |
| RESNET | 21.22 | 26.24 | 0.838 | 0.795 | 0.880 |
| TRANSFORMER | 19.47 | 24.08 | 0.876 | 0.835 | 0.917 |

### Dishwasher

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 26.23 | 32.93 | 0.892 | 0.847 | 0.937 |
| GRU | 25.57 | 29.91 | 0.865 | 0.839 | 0.892 |
| TCN | 23.14 | 28.91 | 0.891 | 0.854 | 0.928 |
| LIQUID | 18.06 | 22.92 | 0.931 | 0.892 | 0.971 |
| ADVANCED_LIQUID | 17.41 | 22.11 | 0.946 | 0.898 | 0.993 |
| RESNET | 23.51 | 28.49 | 0.879 | 0.851 | 0.908 |
| TRANSFORMER | 21.20 | 24.64 | 0.923 | 0.902 | 0.944 |

### Microwave

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 28.80 | 33.74 | 0.929 | 0.878 | 0.979 |
| GRU | 29.99 | 36.44 | 0.908 | 0.831 | 0.985 |
| TCN | 26.43 | 31.64 | 0.950 | 0.929 | 0.971 |
| LIQUID | 20.06 | 24.24 | 0.982 | 0.933 | 1.000 |
| ADVANCED_LIQUID | 19.93 | 24.31 | 0.978 | 0.934 | 1.000 |
| RESNET | 25.67 | 31.24 | 0.933 | 0.899 | 0.967 |
| TRANSFORMER | 22.80 | 29.03 | 0.965 | 0.940 | 0.991 |

### Washer_Dryer

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 29.69 | 37.29 | 0.973 | 0.947 | 0.999 |
| GRU | 33.04 | 36.37 | 0.943 | 0.872 | 1.000 |
| TCN | 27.96 | 34.33 | 0.972 | 0.927 | 1.000 |
| LIQUID | 22.67 | 26.92 | 0.964 | 0.913 | 1.000 |
| ADVANCED_LIQUID | 20.49 | 25.07 | 0.974 | 0.905 | 1.000 |
| RESNET | 29.17 | 35.88 | 0.978 | 0.906 | 1.000 |
| TRANSFORMER | 24.81 | 30.57 | 0.970 | 0.946 | 0.993 |

### Kettle

| Model | MAE | RMSE | F1 Score | Precision | Recall |
|-------|-----|------|----------|-----------|--------|
| LSTM | 20.36 | 23.91 | 0.789 | 0.742 | 0.836 |
| GRU | 21.90 | 27.00 | 0.779 | 0.761 | 0.797 |
| TCN | 18.02 | 22.13 | 0.799 | 0.750 | 0.848 |
| LIQUID | 14.65 | 16.05 | 0.845 | 0.818 | 0.873 |
| ADVANCED_LIQUID | 13.89 | 16.50 | 0.859 | 0.826 | 0.892 |
| RESNET | 19.82 | 24.70 | 0.794 | 0.752 | 0.835 |
| TRANSFORMER | 16.52 | 20.16 | 0.822 | 0.798 | 0.846 |

## Focus on Liquid Neural Networks

The Liquid Neural Network architectures demonstrated several advantages over traditional neural networks for NILM tasks:

1. **Superior Overall Performance**: Liquid Neural Networks achieved the best results across most metrics, with the Advanced LNN variant showing particularly strong performance.

2. **Handling of Temporal Dynamics**: The adaptive time constants in LNNs appear well-suited for capturing the complex temporal patterns of different appliances.

3. **Consistent Performance Across Appliances**: While other models showed variable performance depending on the appliance, LNNs maintained strong results across all appliance types.

4. **Effectiveness for Both On/Off and Variable-Power Appliances**: LNNs performed well for both appliances with simple on/off patterns (like kettle) and those with variable power modes (like washing machines).

### Comparison with Traditional Models

#### LIQUID

**MAE**: LIQUID outperforms LSTM by 27.8%, GRU by 31.0%, TCN by 20.1%, RESNET by 22.0% and TRANSFORMER by 11.1%.

**RMSE**: LIQUID outperforms LSTM by 28.0%, GRU by 29.4%, TCN by 21.5%, RESNET by 23.3% and TRANSFORMER by 12.5%.

**F1**: LIQUID improves upon LSTM by 4.5%, GRU by 7.1%, TCN by 3.4%, RESNET by 4.3% and TRANSFORMER by 1.2%.

**PRECISION**: LIQUID improves upon LSTM by 5.3%, GRU by 8.8%, TCN by 3.1% and RESNET by 4.8%.

**RECALL**: LIQUID improves upon LSTM by 2.9%, GRU by 5.0%, TCN by 3.2%, RESNET by 4.0% and TRANSFORMER by 1.7%.

#### ADVANCED_LIQUID

**MAE**: ADVANCED_LIQUID outperforms LSTM by 32.1%, GRU by 35.1%, TCN by 24.8%, RESNET by 26.6% and TRANSFORMER by 16.4%.

**RMSE**: ADVANCED_LIQUID outperforms LSTM by 31.6%, GRU by 32.9%, TCN by 25.3%, RESNET by 27.1% and TRANSFORMER by 16.8%.

**F1**: ADVANCED_LIQUID improves upon LSTM by 6.0%, GRU by 8.7%, TCN by 4.9%, RESNET by 5.8% and TRANSFORMER by 2.7%.

**PRECISION**: ADVANCED_LIQUID improves upon LSTM by 6.2%, GRU by 9.8%, TCN by 4.0%, RESNET by 5.7% and TRANSFORMER by 0.6%.

**RECALL**: ADVANCED_LIQUID improves upon LSTM by 4.5%, GRU by 6.6%, TCN by 4.8%, RESNET by 5.6% and TRANSFORMER by 3.3%.

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

