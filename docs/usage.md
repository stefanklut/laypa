# Usage

(installation)=

## Installation

To use loghi, you need to install the package.

## Creating recipes

To retrieve a list of random ingredients,
you can use the `train.setup_training()` function:

```{eval-rst}
.. autofunction:: train.setup_training
```

{py:func}`train.setup_training` will train a model
will raise an exception.

```{eval-rst}
.. autoexception:: laypa.InvalidKindError
```

For example:

```pycon
>>> import laypa
>>> laypa.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']
```
