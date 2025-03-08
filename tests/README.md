# Testing

To run the tests type

    pytest -v # increase verbosity.
           -x # stop after first failure
           -s # shortcut for --capture=no.
           --capture=method per-test capturing method: one of fd|sys|no.

We’ll use [`pytest`](https://docs.pytest.org/en/latest/) since it allows for more pythonic test code as compared to the JUnit-inspired unittest module.

`pytest` will run all files of the form `test_*.py` or `*_test.py` in the current directory and its subdirectories.

## Some useful links

Anatomy of a test
https://docs.pytest.org/en/latest/explanation/anatomy.html

About fixtures
https://docs.pytest.org/en/latest/explanation/fixtures.html

How to parametrize fixtures and test functions
https://docs.pytest.org/en/latest/how-to/parametrize.html



