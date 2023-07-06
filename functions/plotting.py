import matplotlib.pyplot as plt


def plot_single_output_history(hist) -> None:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hist['loss'], label='Train Loss')
    plt.plot(hist['val_loss'], label='Validation Loss')
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
