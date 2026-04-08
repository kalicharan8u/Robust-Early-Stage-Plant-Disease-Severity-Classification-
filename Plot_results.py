import numpy as np
import warnings
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics

warnings.filterwarnings("ignore")


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'SCO-APSPN', 'SAA-APSPN', 'ECO-APSPN', 'QSO-APSPN', 'ERRV-QSOA-APSPN']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(Fitness.shape[1])
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.plot(length, Fitness[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label=Algorithm[1])
    plt.plot(length, Fitness[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label=Algorithm[2])
    plt.plot(length, Fitness[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label=Algorithm[3])
    plt.plot(length, Fitness[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label=Algorithm[4])
    plt.plot(length, Fitness[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label=Algorithm[5])
    plt.xlabel('No. of Iteration', fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.ylabel('Cost Function', fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.legend(loc=1, prop={'weight':'bold', 'size':12})
    plt.savefig("./Results/Conv.png")
    plt.show()


def Plot_ROC_Curve():
    cls = ['CNN', 'VGG-16', 'Resnet', 'RAN', 'DResBiGRU']
    Actual = np.load('Target.npy', allow_pickle=True)
    lenper = round(Actual.shape[0] * 0.75)
    Actual = Actual[lenper:, :]
    fig = plt.figure()
    fig.canvas.manager.set_window_title('ROC Curve')
    colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
    for i, color in zip(range(len(cls)), colors):  # For all classifiers
        Predicted = np.load('Y_Score_1.npy', allow_pickle=True)[i]
        false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc * 100

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            lw=5,
            label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate",fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.ylabel("True Positive Rate",fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.title("ROC Curve")
    plt.legend(loc="lower right", prop={'weight':'bold', 'size':12})
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Table():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Algorithm = ['BatchSize', 'SCO', 'SAA', 'ECO', 'QSO', 'Proposed']
    Classifier = ['BatchSize', 'CNN', 'VGG-16', 'Resnet', 'RAN', 'DResBiGRU']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 4, 6, 9, 15]).astype(int)
    Table_Terms = [0, 2, 4, 6, 9, 15]
    table_terms = [Terms[i] for i in Table_Terms]
    Batchsize = ['4', '8', '16', '32', '48']
    for k in range(len(Table_Terms)):
        value = eval[:, :, 4:]

        Table = PrettyTable()
        Table.add_column(Classifier[0], Batchsize)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, j, Graph_Terms[k]])
        print('---------------------------------------', table_terms[k], '  Classifier Comparison',
              '---------------------------------------')
        print(Table)


def Plots_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    bar_width = 0.15
    kfold = [1, 2, 3, 4]
    Classifier = ['CNN', 'VGG-16', 'Resnet', 'RAN', 'DResBiGRU']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Method Comparison of Epochs')
            X = np.arange(len(kfold))
            bars1 = plt.bar(X + 0.00, Graph[:, 0], color='#8f2d56', edgecolor='w', width=0.15, label=Classifier[0])
            bars2 = plt.bar(X + 0.15, Graph[:, 1], color='#b388eb', edgecolor='w', width=0.15, label=Classifier[1])
            bars3 = plt.bar(X + 0.30, Graph[:, 2], color='#219ebc', edgecolor='w', width=0.15, label=Classifier[2])
            bars4 = plt.bar(X + 0.45, Graph[:, 3], color='#f77f00', edgecolor='w', width=0.15, label=Classifier[3])
            bars5 = plt.bar(X + 0.60, Graph[:, 4], color='#8ac926', edgecolor='w', width=0.15, label=Classifier[4])
            for bars in [bars1, bars2, bars3, bars4, bars5]:
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, height - (0.05 * height),
                             f"{str(np.round(height, 1))}", ha='center', va='top', fontsize=10, color='w', rotation=90,
                             fontweight='bold')
            plt.xticks(X + bar_width * 2, ['20', '40', '60', '80'], fontname="Arial",
                       fontsize=17,
                       fontweight='bold', color='k')
            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=18, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=18, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=17, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['#8f2d56', '#b388eb', '#219ebc', '#f77f00', '#8ac926']]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=10,
                       frameon=False, ncol=len(Classifier), prop={'weight':'bold', 'size':12})
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            path = "./Results/%s_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def plot_seg_results():
    Eval_all = np.load('Evaluate_Seg_all.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['TERMS', 'SCO-APSPN', 'SAA-APSPN', 'ECO-APSPN', 'QSO-APSPN', 'ERRV-QSOA-APSPN']
    Methods = ['TERMS', 'Unet', 'Unet3+', 'YoloV3', 'PSPN', 'ERRV-QSOA-APSPN']
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Statistics[1::3])
            for k in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[k + 1], stats[i, k, 1::3])
            print('-------------------------------------------------- ', Terms[i - 4],
                  'Algorithm  for Segmentation', '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Methods[0], Statistics[1::3])
            Table.add_column(Methods[1], stats[i, 5, 1::3])
            Table.add_column(Methods[2], stats[i, 6, 1::3])
            Table.add_column(Methods[3], stats[i, 7, 1::3])
            Table.add_column(Methods[4], stats[i, 8, 1::3])
            Table.add_column(Methods[5], stats[i, 4, 1::3])
            print('-------------------------------------------------- ', Terms[i - 4],
                  'Comparison for Segmentation', '--------------------------------------------------')
            print(Table)

            X = np.arange(len(Statistics) - 3)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title(str(Terms[i - 4]) + 'Algorithm Comparison')
            ax.bar(X + 0.00, stats[i, 0, 0:3:2], color='#f77f00', edgecolor='w', width=0.15, label=Algorithm[1])
            ax.bar(X + 0.15, stats[i, 1, 0:3:2], color='#52796f', edgecolor='w', width=0.15, label=Algorithm[2])
            ax.bar(X + 0.30, stats[i, 2, 0:3:2], color='#4361ee', edgecolor='w', width=0.15, label=Algorithm[3])
            ax.bar(X + 0.45, stats[i, 3, 0:3:2], color='#ff0054', edgecolor='w', width=0.15, label=Algorithm[4])
            ax.bar(X + 0.60, stats[i, 4, 0:3:2], color='k', edgecolor='w', width=0.15, label=Algorithm[5])
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['#f77f00', '#52796f', '#4361ee', '#ff0054', 'k']]
            plt.legend(dot_markers, ['SCO-APSPN', 'SAA-APSPN', 'ECO-APSPN', 'QSO-APSPN', 'ERRV-QSOA-APSPN'],
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1.20), fontsize=10,
                       frameon=False, ncol=3, prop={'weight':'bold', 'size':12})

            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.xticks(X + 0.30, ('BEST', 'MEAN'),fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.xlabel('Statisticsal Analysis',fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[i - 4],fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
            path = "./Results/%s_Alg_bar.png" % (Terms[i - 4])
            plt.savefig(path)
            plt.show()

            X = np.arange(len(Statistics) - 3)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title(str(Terms[i - 4]) + 'Classification Comparison')
            ax.bar(X + 0.00, stats[i, 5, 0:3:2], color='#ff1654', edgecolor='w', width=0.15, label=Methods[1])
            ax.bar(X + 0.15, stats[i, 6, 0:3:2], color='#8ac926', edgecolor='w', width=0.15, label=Methods[2])
            ax.bar(X + 0.30, stats[i, 7, 0:3:2], color='#9e2a2b', edgecolor='w', width=0.15, label=Methods[3])
            ax.bar(X + 0.45, stats[i, 8, 0:3:2], color='#197278', edgecolor='w', width=0.15, label=Methods[4])
            ax.bar(X + 0.60, stats[i, 4, 0:3:2], color='k', edgecolor='w', width=0.15, label=Methods[5])
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['#ff1654', '#8ac926', '#9e2a2b', '#197278', 'k']]
            plt.legend(dot_markers, ['Unet', 'Unet3+', 'YoloV3', 'PSPN', 'ERRV-QSOA-APSPN'], loc='upper center',
                       bbox_to_anchor=(0.5, 1.20), fontsize=10,
                       frameon=False, ncol=3, prop={'weight':'bold', 'size':12})

            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.xticks(X + 0.30, ('BEST', 'MEAN'), fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.xlabel('Statisticsal Analysis', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[i - 4], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
            path = "./Results/%s_Mod_bar.png" % (Terms[i - 4])
            plt.savefig(path)
            plt.show()


def Proposed_PlotsResults():
    Eval_all = np.load('Evaluate_Seg_all.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['QSO-APSPN', 'ERRV-QSOA-APSPN']
    Methods = ['PSPN', 'ERRV-QSOA-APSPN']
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(len(Statistics) - 1)
            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.28, 0.8, 0.65])
            fig.canvas.manager.set_window_title(str(Terms[i - 4]) + 'Algorithm Comparison')
            ax.bar(X + 0.00, stats[i, 3, :4], color='#00bbf9', edgecolor='w', width=0.25, label=Algorithm[0])
            ax.bar(X + 0.30, stats[i, 4, :4], color='#f86624', edgecolor='w', width=0.25, label=Algorithm[1])
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['#00bbf9', '#f86624']]
            plt.legend(dot_markers, Algorithm, loc='upper center',
                       bbox_to_anchor=(0.5, -0.23), fontsize=10,
                       frameon=False, ncol=5, prop={'weight':'bold', 'size':12})

            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.xticks(X + 0.15, ('BEST', 'WORST', 'MEAN', 'MEDIAN'), fontname="Arial", fontsize=14, fontweight='bold',
                       color='k')
            plt.xlabel('Statisticsal Analysis', fontname="Arial", fontsize=15, fontweight='bold', color='b')
            plt.ylabel(Terms[i - 4], fontname="Arial", fontsize=15, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='b')
            path = "./Results/%s_Propo_Alg.png" % (Terms[i - 4])
            plt.savefig(path)
            plt.show()

            X = np.arange(len(Statistics) - 1)
            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.28, 0.8, 0.65])
            fig.canvas.manager.set_window_title(str(Terms[i - 4]) + 'Classification Comparison')
            ax.bar(X + 0.00, stats[i, 8, :4], color='#4f5d75', edgecolor='w', width=0.25, label=Methods[0])
            ax.bar(X + 0.30, stats[i, 4, :4], color='#c44900', edgecolor='w', width=0.25, label=Methods[1])
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['#4f5d75', '#c44900']]
            plt.legend(dot_markers, Methods, loc='upper center',
                       bbox_to_anchor=(0.5, -0.23), fontsize=10,
                       frameon=False, ncol=5, prop={'weight':'bold', 'size':12})

            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.xticks(X + 0.15, ('BEST', 'WORST', 'MEAN', 'MEDIAN'), fontname="Arial", fontsize=14, fontweight='bold',
                       color='k')
            plt.xlabel('Statisticsal Analysis', fontname="Arial", fontsize=15, fontweight='bold', color='b')
            plt.ylabel(Terms[i - 4], fontname="Arial", fontsize=15, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='b')
            path = "./Results/%s_Prop_mod.png" % (Terms[i - 4])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    Plots_Results()
    Plot_ROC_Curve()
    plot_seg_results()
    Table()
    Proposed_PlotsResults()
