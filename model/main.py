from network import CNN_v1, MobileNetV2_debit_card

import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import os


def plot_results(history, filename: str, epochs: int):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(filename, dpi=300)
    # plt.show()


if __name__ == "__main__":
    data_dir = pathlib.Path("../data/").with_suffix('')
    batch_size = 32
    img_size = 192
    epochs = 20
    cnn = CNN_v1(batch_size, img_size, data_dir)
    mobile_net = MobileNetV2_debit_card(batch_size, img_size, data_dir)
    cnn_hist = cnn.build_model(epochs)
    mobile_net_hist = mobile_net.build_model(epochs)
    plot_results(cnn_hist, "../plots/cnn_performance.png", epochs)
    plot_results(mobile_net_hist, "../plots/mobile_net_performance.png", epochs)
    cnn.save_model("../cnn.keras")
    mobile_net.save_model("../mobile_net_debit.keras")
    #mobile_net.store_misclassified("misclassified")
