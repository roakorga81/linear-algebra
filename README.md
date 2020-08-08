# Linear Algebra Workbook

This repository is not intended to be used as a linear algebra library. Rather, the purpose is to be used as a way to study linear algebra and the common operations defined in linear algebra. You can think of this repository as a self guided workbook for linear algebra.

This repository can also be seen as a Python workbook, since you would get familiar with building a Python library as you go through. This is the reason to not use Numpy (the most used numeric Python library). If we were to use that library, we would end up calling already made functions and objects instead of building our own and understanding the operations used in linear algebra. 

## Your Task

What you should do is that you should implement the library. The library consist of the class vector and matrix, and common operations. You will notice that you are given the test cases for each of the implementaions required. Please report an issue if you find that a test case is wrong. To start, please do:

```
git clone https://github.com/open-workbooks/linear-algebra.git
cd linear-algebra
pip install -e ".[dev]"
```

Now you can run the tests and start implementing your own linear algebra library!


### Library desing

We have two simple classes, `Vector` and `Matrix`, which represent the mathematical objects with the same name. These classes offer some ways of building commonly used instances by using classmethods, for instance, the class `Vector` has `classmethod` named `make_random` to build a vector out of random entries. Both classes make some mathematical operator overloading allowed by python, so we can sum two vectors by simply using `+ `. We also implement some features that make the classes look like native Pyhton, by going through the implementation of such classes you could learn a lot about Python.

We reserve the use of the `property` decorator to attributes that are of the very nature of the object. For instance, every vector has a norm (here we use the euclidean one) and that is a property of each `Vector` instance. The same is true for the `determinant` and `inverse` of a matrix, even though they may be undefined, which is reported by raising an error with a well descriptive message. Additionally, there are some functions that operate on vectors, some on matrices and some on both. 

The tests cases for all implementations aim to hide implementation details, since otherwise the implementation can be derived from there. We make an effort to implement tests cases as the mathematical properties of whatever is being tested. Hopefully this will give the user yet another chance to see how mathematical relations can be implemented in Python code.

Finally, we favor type hints and self descriptive names over docstrings, but we make use of docstrings when we need to (when it is not redundant given the names and typehints used.)

## Where to get the answers

The workbook is splited into two repositories `open-workbooks/linear-algebra` and `open-workbooks/linear-algebra-answers`, which hold the task and the answers respectively. Since this worbook can be used as teaching material, the repo with the ansers is a private repository, but we can give you access to the private repository if you fill out our [collaborators form](https://forms.gle/atFNQEUxryN72L189). The repo with the answers has CI/CD implemented that so that all pushes to master update the repo without the answers, we can make sure that the task repo is always updated.

## Contribute

All forms of contributing are highly appreciated, please read the [contributing guide](./CONTRIBUTING.md)

## Collaborators

- Sebastián Rodríguez Colina
- [Add your name here!](./CONTRIBUTING.md)
