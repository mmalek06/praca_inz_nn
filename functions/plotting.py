import matplotlib.pyplot as plt
import numpy as np


def plot_single_output_history(hist, outlier_threshold=None) -> None:
    train_loss = np.array(hist['loss'])
    val_loss = np.array(hist['val_loss'])
    train_acc = np.array(hist['accuracy'])
    val_acc = np.array(hist['val_accuracy'])

    if outlier_threshold is None:
        Q1 = np.percentile(train_loss, 25)
        Q3 = np.percentile(train_loss, 75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR

    train_loss_outliers = train_loss > outlier_threshold
    val_loss_outliers = val_loss > outlier_threshold
    train_loss_line = np.where(train_loss_outliers, np.nan, train_loss)
    val_loss_line = np.where(val_loss_outliers, np.nan, val_loss)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_line, label='Train Loss')
    plt.plot(val_loss_line, label='Validation Loss')

    plt.plot(np.where(train_loss_outliers)[0], train_loss[train_loss_outliers], 'ro', label='Outliers')

    for i, loss in zip(np.where(train_loss_outliers)[0], train_loss[train_loss_outliers]):
        plt.text(i, loss, f'{loss:.2f}', color='red')

    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist['accuracy'], label='Train accuracy')
    plt.plot(hist['val_accuracy'], label='Validation accuracy')
    plt.title('Metric Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_multi_output_history(
        hist,
        loss_key='loss',
        val_loss_key='val_loss',
        metric_key='ciou_metric',
        val_metric_key='val_ciou_metric') -> None:
    # Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history1[loss_key], label='Train Loss')
    plt.plot(hist.history1[val_loss_key], label='Validation Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    # CIoU vals
    plt.subplot(1, 2, 2)
    plt.plot(hist.history1[metric_key], label='Train CIoU metric')
    plt.plot(hist.history1[val_metric_key], label='Validation CIoU metric')
    plt.title('Metric Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
