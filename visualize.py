import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def visualize_PRF():
    with open("modelfiles/tta_msdrop_n/history.txt", "r", encoding="utf-8") as fr:
        history_n = fr.read()
        history_n = eval(history_n)

    with open("modelfiles/tta_msdrop_y_8/history.txt", "r", encoding="utf-8") as fr:
        history_y = fr.read()
        history_y = eval(history_y)

    with open("modelfiles/tta_msdrop_yy_8/history.txt", "r", encoding="utf-8") as fr:
        history_yy = fr.read()
        history_yy = eval(history_yy)

    gs = gridspec.GridSpec(2, 2)
    plt.subplot(gs[0, 0])
    plt.plot(history_n["loss"])
    plt.plot(history_y["loss"])
    plt.plot(history_yy["loss"])
    plt.grid()
    plt.title('loss')
    plt.xlabel('Epoch')
    plt.legend(['without MSDrop', 'with MSDropLogits','with MSDropLoss'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 0])
    plt.plot(history_n["val_loss"])
    plt.plot(history_y["val_loss"])
    plt.plot(history_yy["val_loss"])
    plt.grid()
    plt.title('val_loss')
    plt.xlabel('Epoch')
    plt.legend(['without MSDrop', 'with MSDropLogits','with MSDropLoss'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 1])
    plt.plot(history_n["acc"])
    plt.plot(history_y["acc"])
    plt.plot(history_yy["acc"])
    plt.grid()
    plt.title('acc')
    plt.xlabel('Epoch')
    plt.legend(['without MSDrop', 'with MSDropLogits','with MSDropLoss'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 1])
    plt.plot(history_n["val_acc"])
    plt.plot(history_y["val_acc"])
    plt.plot(history_yy["val_acc"])
    plt.grid()
    plt.title('val_acc')
    plt.xlabel('Epoch')
    plt.legend(['without MSDrop', 'with MSDropLogits','with MSDropLoss'], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig("msdrop_LossAcc.jpg", dpi=1000, bbox_inches="tight")

    gs = gridspec.GridSpec(2, 3)

    plt.subplot(gs[0, 0])
    plt.plot(history_n["precision"])
    plt.plot(history_y["precision"])
    plt.plot(history_yy["precision"])
    plt.grid()
    plt.title('precision')
    plt.xlabel('Epoch')
    plt.legend(['without MSDrop', 'with MSDropLogits','with MSDropLoss'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 0])
    plt.plot(history_n["val_precision"])
    plt.plot(history_y["val_precision"])
    plt.plot(history_yy["val_precision"])
    plt.grid()
    plt.title('val_precision')
    plt.xlabel('Epoch')
    plt.legend(['without MSDrop', 'with MSDropLogits','with MSDropLoss'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 1])
    plt.plot(history_n["recall"])
    plt.plot(history_y["recall"])
    plt.plot(history_yy["recall"])
    plt.grid()
    plt.title('recall')
    plt.xlabel('Epoch')
    plt.legend(['without MSDrop', 'with MSDropLogits','with MSDropLoss'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 1])
    plt.plot(history_n["val_recall"])
    plt.plot(history_y["val_recall"])
    plt.plot(history_yy["val_recall"])
    plt.grid()
    plt.title('val_recall')
    plt.xlabel('Epoch')
    plt.legend(['without MSDrop', 'with MSDropLogits','with MSDropLoss'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 2])
    plt.plot(history_n["f1"])
    plt.plot(history_y["f1"])
    plt.plot(history_yy["f1"])
    plt.grid()
    plt.title('f1')
    plt.xlabel('Epoch')
    plt.legend(['without MSDrop', 'with MSDropLogits','with MSDropLoss'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 2])
    plt.plot(history_n["val_f1"])
    plt.plot(history_y["val_f1"])
    plt.plot(history_yy["val_f1"])
    plt.grid()
    plt.title('val_f1')
    plt.xlabel('Epoch')
    plt.legend(['without MSDrop', 'with MSDropLogits','with MSDropLoss'], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig("msdrop_PRF.jpg", dpi=1000, bbox_inches="tight")


def visualize_PRF_alpha():
    with open("modelfiles/tta_msdrop_n/history.txt", "r", encoding="utf-8") as fr:
        history_n = fr.read()
        history_n = eval(history_n)

    with open("modelfiles/tta_msdrop_y_0.3/history.txt", "r", encoding="utf-8") as fr:
        history_y_1 = fr.read()
        history_y_1 = eval(history_y_1)

    with open("modelfiles/tta_msdrop_y_1.0/history.txt", "r", encoding="utf-8") as fr:
        history_y_2 = fr.read()
        history_y_2 = eval(history_y_2)

    with open("modelfiles/tta_msdrop_y_2.0/history.txt", "r", encoding="utf-8") as fr:
        history_y_3 = fr.read()
        history_y_3 = eval(history_y_3)

    with open("modelfiles/tta_msdrop_y_5.0/history.txt", "r", encoding="utf-8") as fr:
        history_y_4 = fr.read()
        history_y_4 = eval(history_y_4)

    gs = gridspec.GridSpec(2, 2)
    plt.subplot(gs[0, 0])
    plt.plot(history_n["loss"])
    plt.plot(history_y_1["loss"])
    plt.plot(history_y_2["loss"])
    plt.plot(history_y_3["loss"])
    plt.plot(history_y_4["loss"])
    plt.grid()
    plt.title('loss')
    plt.xlabel('Epoch')
    plt.legend(['0.0', '0.3', '1.0','2.0','5.0'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 0])
    plt.plot(history_n["val_loss"])
    plt.plot(history_y_1["val_loss"])
    plt.plot(history_y_2["val_loss"])
    plt.plot(history_y_3["val_loss"])
    plt.plot(history_y_4["val_loss"])
    plt.grid()
    plt.title('val_loss')
    plt.xlabel('Epoch')
    plt.legend(['0.0', '0.3', '1.0','2.0','5.0'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 1])
    plt.plot(history_n["acc"])
    plt.plot(history_y_1["acc"])
    plt.plot(history_y_2["acc"])
    plt.plot(history_y_3["acc"])
    plt.plot(history_y_4["acc"])
    plt.grid()
    plt.title('acc')
    plt.xlabel('Epoch')
    plt.legend(['0.0', '0.3', '1.0','2.0','5.0'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 1])
    plt.plot(history_n["val_acc"])
    plt.plot(history_y_1["val_acc"])
    plt.plot(history_y_2["val_acc"])
    plt.plot(history_y_3["val_acc"])
    plt.plot(history_y_4["val_acc"])
    plt.grid()
    plt.title('val_acc')
    plt.xlabel('Epoch')
    plt.legend(['0.0', '0.3', '1.0','2.0','5.0'], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig("msdrop_LossAcc.jpg", dpi=1000, bbox_inches="tight")

    gs = gridspec.GridSpec(2, 3)

    plt.subplot(gs[0, 0])
    plt.plot(history_n["precision"])
    plt.plot(history_y_1["precision"])
    plt.plot(history_y_2["precision"])
    plt.plot(history_y_3["precision"])
    plt.plot(history_y_4["precision"])
    plt.grid()
    plt.title('precision')
    plt.xlabel('Epoch')
    plt.legend(['0.0', '0.3', '1.0','2.0','5.0'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 0])
    plt.plot(history_n["val_precision"])
    plt.plot(history_y_1["val_precision"])
    plt.plot(history_y_2["val_precision"])
    plt.plot(history_y_3["val_precision"])
    plt.plot(history_y_4["val_precision"])
    plt.grid()
    plt.title('val_precision')
    plt.xlabel('Epoch')
    plt.legend(['0.0', '0.3', '1.0','2.0','5.0'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 1])
    plt.plot(history_n["recall"])
    plt.plot(history_y_1["recall"])
    plt.plot(history_y_2["recall"])
    plt.plot(history_y_3["recall"])
    plt.plot(history_y_4["recall"])
    plt.grid()
    plt.title('recall')
    plt.xlabel('Epoch')
    plt.legend(['0.0', '0.3', '1.0','2.0','5.0'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 1])
    plt.plot(history_n["val_recall"])
    plt.plot(history_y_1["val_recall"])
    plt.plot(history_y_2["val_recall"])
    plt.plot(history_y_3["val_recall"])
    plt.plot(history_y_4["val_recall"])
    plt.grid()
    plt.title('val_recall')
    plt.xlabel('Epoch')
    plt.legend(['0.0', '0.3', '1.0','2.0','5.0'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 2])
    plt.plot(history_n["f1"])
    plt.plot(history_y_1["f1"])
    plt.plot(history_y_2["f1"])
    plt.plot(history_y_3["f1"])
    plt.plot(history_y_4["f1"])
    plt.grid()
    plt.title('f1')
    plt.xlabel('Epoch')
    plt.legend(['0.0', '0.3', '1.0','2.0','5.0'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 2])
    plt.plot(history_n["val_f1"])
    plt.plot(history_y_1["val_f1"])
    plt.plot(history_y_2["val_f1"])
    plt.plot(history_y_3["val_f1"])
    plt.plot(history_y_4["val_f1"])
    plt.grid()
    plt.title('val_f1')
    plt.xlabel('Epoch')
    plt.legend(['0.0', '0.3', '1.0','2.0','5.0'], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig("msdrop_PRF.jpg", dpi=1000, bbox_inches="tight")


if __name__ == "__main__":
    visualize_PRF()
    # visualize_PRF_alpha()
