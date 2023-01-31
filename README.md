<h1 align="center">presentable</h1> 
<h3 align="center">Providing a Prettier Confusion Matrix for your Command Line</h3>

<p align="center">  
  <img alt="presentablelogo" src="https://github.com/JacksonBurns/presentable/blob/main/presentable_logo.png">
</p> 
<p align="center">
  <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/JacksonBurns/presentable?style=social">
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/presentable">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/presentable">
  <img alt="PyPI - License" src="https://img.shields.io/github/license/JacksonBurns/presentable">
</p>

## Installation
`presentable` is available on PyPI. Install it with `pip install presentable`.

## Usage
`presentable` is a drop in replacement from `sklearn.metrics.confusion_matrix` that just includes some nice formatting and only ever prints to the terminal, helpful on remote hardware accessible only from the CLI.

```python
>>> confusion_matrix(gtr, pred, tabulate_args={"tablefmt":"github","floatfmt":".2f"},sklearn_args={"normalize":"all"})
| Truth\Model   |   cat |   dog |
|---------------|-------|-------|
| cat           |  0.17 |  0.17 |
| dog           |  0.33 |  0.33 |
```

`tabulate_args` and `sklearn_args` are optional dictionaries to specify additional configurations arguments for `tabulate` and sklearn's `confusion_matrix`. Check the [`tabulate` documentation](https://github.com/astanin/python-tabulate#table-format) and [`sklearn` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) for a list of which args are supported.

## Online Documentation
[Click here to read the documentation](https://JacksonBurns.github.io/presentable/)

