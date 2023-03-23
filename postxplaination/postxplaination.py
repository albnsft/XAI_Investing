import pandas as pd
import numpy as np
import shap
from pylab import plt
import matplotlib.pyplot as pl


def shap_plots(clf):
    clf.columns = clf.X_val.columns.to_list()
    X_val = clf.model['scaler'].transform(clf.X_val.values)
    X_test = clf.model['scaler'].transform(clf.X_test.values)
    tree_explainer = shap.TreeExplainer(clf.model['clf'])

    val_shap_values = pd.DataFrame(tree_explainer.shap_values(X_val)[1], columns=clf.columns)
    val_shape_mean = val_shap_values.mean().sort_values()

    test_shap_values = pd.DataFrame(tree_explainer.shap_values(X_test)[1], columns=clf.columns)
    test_shape_mean = test_shap_values.mean().sort_values()

    fig = plt.figure(constrained_layout=True, figsize=(15, 10))
    ax_dict = fig.subplot_mosaic(
        """
        AB
        """
    )
    plt.suptitle('Long (green) vs Short (red) Indicators')
    val_shape_mean.plot(ax=ax_dict['A'], kind='barh', color=(val_shape_mean > 0).map({True: 'g', False: 'r'}),
                        title ='Validation - Contribution of features')
    test_shape_mean.plot(ax=ax_dict['B'], kind='barh', color=(test_shape_mean > 0).map({True: 'g', False: 'r'}),
                        title='Testing - Contribution of features')
    pdf_filename = f"{clf.path_results}\\{clf.method_type}_contrib_features.jpg"
    fig.savefig(pdf_filename)
    plt.close(fig)

    shap.summary_plot(val_shap_values.values, X_val, feature_names=clf.columns, plot_type='dot', show=False)
    pdf_filename = f"{clf.path_results}\\{clf.method_type}_summary_val.jpg"
    plt.savefig(pdf_filename)
    plt.close()

    shap.summary_plot(test_shap_values.values, X_test, feature_names=clf.columns, plot_type='dot', show=False)
    pdf_filename = f"{clf.path_results}\\{clf.method_type}_summary_test.jpg"
    plt.savefig(pdf_filename)
    plt.close()





