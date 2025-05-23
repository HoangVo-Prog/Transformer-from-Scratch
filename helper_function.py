import torch
import warnings


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if RUN_EXAMPLES:
        chart = fn(*args)
    if fn.__name__ != "run_tests":
        chart.save(f"examples/{fn.__name__}.html")
        print(f"Example saved to examples/{fn.__name__}.html")


def execute_example(fn, args=[]):
    if RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None
