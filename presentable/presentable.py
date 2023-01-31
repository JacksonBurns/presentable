from typing import Union

from tabulate import tabulate
import numpy as np

from sklearn.metrics import confusion_matrix as sk_conmat
from sklearn.utils.multiclass import unique_labels


def confusion_matrix(
    truth: Union[list, np.array],
    prediction: Union[list, np.array],
    tabulate_args: dict = {},
    sklearn_args: dict = {},
) -> None:
    """Prints a nicely formatted confusion matrix.

    Args:
        truth (Union[list, np.array]): Ground truth, reference values.
        prediction (Union[list, np.array]): Model predictions. Must be same size as prediction.
        tabulate_args (dict, optional): Optional additional customization for tabulate. Defaults to empty dict.
        sklearn_args (dict, optional): Optional additional customization for sklearn's confusion_matrix. Defaults to empty dict.
    """
    if not tabulate_args:
        # use default arguments to tabulate
        tabulate_args["tablefmt"] = "outline"

    # convert list dtypes to numpy arrays
    if type(truth) is not np.array:
        truth = np.array(truth)
    if type(prediction) is not np.array:
        prediction = np.array(prediction)

    # get the confusion matrix from sklearn
    conmat = sk_conmat(
        truth,
        prediction,
        **sklearn_args,
    )
    # change the dtype to object to allow strings to be added later
    conmat = np.array(conmat, dtype=object)

    # get the unique labels also with sklearn
    labels = unique_labels(truth, prediction)

    # formulate it into the tabulate table
    presentable_conmat = tabulate(
        np.insert(conmat, 0, labels.T, axis=1),
        headers=["Truth\\Model"] + labels.tolist(),
        **tabulate_args,
    )

    # print the table
    print(presentable_conmat)
