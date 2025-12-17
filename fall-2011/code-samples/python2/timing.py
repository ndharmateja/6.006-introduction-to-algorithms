# timing_py3_fixed.py
# Converted to Python 3 and patched for big-int and Python 3.11+ limits.
# Original Author: Ronald L. Rivest
# Conversion & patches: ChatGPT

import math
import sys
import timeit
import scipy.optimize
import scipy.linalg  # kept in case you want to switch to lstsq

# Note: scipy must be installed for fit2 to work.

# Allow larger decimal-to-int conversions for these timing experiments (Python 3.11+)
# Increase this if you plan to convert even bigger strings to int.
try:
    sys.set_int_max_str_digits(200000)
except AttributeError:
    # Older Python versions don't have this function; ignore.
    pass

# Parameter generation routines


def lg(x):
    return math.log(x) / math.log(2.0)


def sqrt(x):
    return math.sqrt(x)


def make_param_list(spec_string, growth_factor):
    """
    Generate a list of dictionaries
    given maximum and minimum values for each range.
    Each min and max value is a *string* that can be evaluated;
    each string may depend on earlier variable values.
    Values increment by factor of growth_factor from min to max.

    Example:
       make_param_list("1<=n<=1000")
       make_param_list("1<=n<=1000;1<=m<=1000;min(n,m)<=k<=max(n,m)")
    """
    var_list = []
    spec_list = spec_string.split(";")
    D = {}
    D["lg"] = lg
    D["sqrt"] = sqrt
    D_list = [D]
    for spec in spec_list:
        spec_parts = spec.split("<=")
        assert len(spec_parts) == 3, "Each spec must have form LOWER<=var<=UPPER"
        lower_spec = spec_parts[0].strip()
        var_name = spec_parts[1].strip()
        assert len(var_name) == 1, "variable name should be a single character"
        var_list.append(var_name)
        upper_spec = spec_parts[2].strip()
        new_D_list = []
        for D in D_list:
            new_D = D.copy()
            val = eval(lower_spec, {}, D)
            # Use a safety guard: ensure val is a number
            while val <= eval(upper_spec, {}, D):
                new_D[var_name] = val
                new_D_list.append(new_D.copy())
                val = val * growth_factor
        D_list = new_D_list
    return (var_list, D_list)


# fit / fitting helpers


def fit(var_list, param_list, run_times, f_list):
    """
    Return matrix A needed for least-squares fit.
    Given:
        list of variable names
        list of sample dicts for various parameter sets
        list of corresponding run times (in microseconds)
        list of functions to be considered for fit
            these are *strings*, e.g. "n","n**2","min(n,m)",etc.
    Prints coefficients for each function in f_list.
    """
    print("var_list", var_list)
    print("Function list:", f_list)
    print("run times:")
    for i in range(len(param_list)):
        D = param_list[i]
        for v in var_list:
            print(f" {v} = {str(D[v]):6s}", end="")
        print(f" : {float(run_times[i]):8f} microseconds")
    print()

    rows = len(run_times)
    cols = len(f_list)
    # build matrix A
    A = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        D = param_list[i]
        for j in range(cols):
            # evaluate the j-th function in D's context
            A[i][j] = float(eval(f_list[j], {}, D))
    b = run_times

    # Solve using relative-error minimizer fit2:
    (x, resids, rank, s) = fit2(A, b)

    print("Coefficients as interpolated from data:")
    for j in range(cols):
        sign = ""
        if x[j] > 0 and j > 0:
            sign = "+"
        elif x[j] > 0:
            sign = " "
        print(f"{sign}{x[j]}*{f_list[j]}")
    print("(measuring time in microseconds)")
    print("Sum of squares of residuals:", resids)
    print("RMS error = %0.2g percent" % (math.sqrt(resids / len(A)) * 100.0))
    sys.stdout.flush()


def fit2(A, b):
    """Relative error minimizer using scipy.optimize.leastsq"""

    def f(x):
        assert len(x) == len(A[0])
        resids = []
        for i in range(len(A)):
            s = 0.0
            for j in range(len(A[0])):
                s += A[i][j] * x[j]
            # avoid division by zero — if b[i]==0 use absolute error fallback
            if b[i] != 0:
                relative_error = (s - b[i]) / b[i]
            else:
                relative_error = s - b[i]
            resids.append(relative_error)
        return resids

    initial = [0.0] * len(A[0])
    ans = scipy.optimize.leastsq(f, initial)
    # ans[0] is the parameter vector
    if len(A[0]) == 1:
        x = [ans[0][0]]
    else:
        x = ans[0].tolist()
    resids = sum([r * r for r in f(x)])
    return (x, resids, 0, 0)


# --------------------------
# Test suites (patched)
# --------------------------


def test_misc():
    print()
    print("Test Misc-1 -- running time should be n+2*m+7+3*n*lg(n)+17*n*m")
    spec_string = "1<=n<=100000;1<=m<=100000"
    growth_factor = 10
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    run_times = [eval("n+2*m+7+3*n*lg(n)+17*n*m", {}, D) for D in param_list]
    f_list = ("(n*m)", "n**2", "n*lg(n)", "n", "m", "1")
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Misc-2: pass")
    spec_string = "10000<=n<=1000000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("1",)
    run_times = []
    trials = 1000
    for D in param_list:
        t = timeit.Timer("pass")
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)


def test_number():
    print()
    print("Test Number-1 -- time to compute int('1'*n)")
    spec_string = "1000<=n<=10000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n**2",)
    run_times = []
    trials = 1000
    for D in param_list:
        # Using int(x) on a large string; sys.set_int_max_str_digits increased above
        t = timeit.Timer("int(x)", "x='1'*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Number-2 -- time to compute repr(2**n)")
    spec_string = "1000<=n<=10000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n**2",)
    run_times = []
    trials = 1000
    for D in param_list:
        t = timeit.Timer("repr(x)", "x=2**%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Number-3 -- time to convert (2**n) to hex")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 1000
    for D in param_list:
        t = timeit.Timer("'%x'%x", "x=2**%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Number-4 -- time to add 2**n to itself")
    spec_string = "1000<=n<=1000000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 10000
    for D in param_list:
        t = timeit.Timer("x+x", "x=2**%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Number-5 -- time to multiply (2**n/3) by itself")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n**1.585",)
    run_times = []
    trials = 1000
    for D in param_list:
        # Use integer division // to avoid float conversion overflow
        t = timeit.Timer("x*x", "x=(2**%(n)s)//3" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Number-6 -- time to divide (2**(2n) by (2**n))")
    spec_string = "1000<=n<=50000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n**2",)
    run_times = []
    trials = 1000
    for D in param_list:
        # Use integer division // to avoid float conversion overflow in timed code
        t = timeit.Timer("w//x", "w=(2**(2*%(n)s));x=(2**(%(n)s))" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Number-7 -- time to compute remainder of (2**(2n) by (2**n))")
    spec_string = "1000<=n<=50000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n**2",)
    run_times = []
    trials = 1000
    for D in param_list:
        t = timeit.Timer("w%x", "w=(2**(2*%(n)s));x=(2**(%(n)s))" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Number-8 -- time to compute pow(x,y,z)")
    spec_string = "1000<=n<=5000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n**3",)
    run_times = []
    trials = 10
    for D in param_list:
        t = timeit.Timer("pow(x,y,z)", "z=(2**%(n)s)+3;x=y=(2**%(n)s)+1" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Number-9 -- time to compute 2**n")
    spec_string = "1000<=n<=1000000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("1",)
    run_times = []
    trials = 10000
    for D in param_list:
        t = timeit.Timer("2**%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)


def test_string():
    print()
    print("Test String-1: extract a byte from a string")
    spec_string = "1000<=n<=1000000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("1",)
    run_times = []
    trials = 1000
    for D in param_list:
        t = timeit.Timer("s[500]", "s='0'*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test String-2: concatenate two string of length n")
    spec_string = "1000<=n<=500000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 1000
    for D in param_list:
        t = timeit.Timer("s+t", "s=t='0'*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test String-3: extract a string of length n/2")
    spec_string = "1000<=n<=500000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 1000
    for D in param_list:
        t = timeit.Timer("s[0:%(n)s//2]" % D, "s='0'*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test String-4: translate a string of length n")
    spec_string = "1000<=n<=500000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 1000
    for D in param_list:
        # Use built-in str.translate and str.maketrans
        setup = "s='0'*%(n)s;T=str.maketrans('1','2')" % D
        t = timeit.Timer("s.translate(T)", setup)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)


def test_list():
    print()
    print("Test List-1: create an empty list")
    spec_string = "1<=n<=10"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("1",)
    run_times = []
    trials = 1000
    for D in param_list:
        t = timeit.Timer("x = list()")
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test List-2: list (array) lookup")
    spec_string = "10000<=n<=1000000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("1",)
    run_times = []
    trials = 1000
    for D in param_list:
        t = timeit.Timer("x=L[5]", "L=[0]*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test List-3: appending to a list of length n")
    spec_string = "10000<=n<=1000000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("1",)
    run_times = []
    trials = 1000
    for D in param_list:
        t = timeit.Timer("L.append(0)", "L=[0]*%(n)s;L.append(0)" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test List-4: Pop")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("1",)
    run_times = []
    trials = 200
    for D in param_list:
        t = timeit.Timer("L.pop()", "L=[0]*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test List-5: concatenating two lists of length n")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 2000
    for D in param_list:
        t = timeit.Timer("L+L", "L=[0]*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test List-6: extracting a slice of length n/2")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 2000
    for D in param_list:
        t = timeit.Timer("L[0:%(n)s//2]" % D, "L=[0]*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test List-7: copy")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 2000
    for D in param_list:
        t = timeit.Timer("L[:]", "L=[0]*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test List-8: assigning a slice of length n/2")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 2000
    for D in param_list:
        t = timeit.Timer("L[0:%(n)s//2]=L[1:1+%(n)s//2]" % D, "L=[0]*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test List-9: Delete first")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 200
    for D in param_list:
        t = timeit.Timer("del L[0]", "L=[0]*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test List-10: Reverse")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 200
    for D in param_list:
        t = timeit.Timer("L.reverse()", "L=[0]*%(n)s" % D)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test List-11: Sort")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n*lg(n)",)
    run_times = []
    trials = 200
    for D in param_list:
        t = timeit.Timer(
            "L.sort()", "import random;L=[random.random() for i in range(%(n)s)]" % D
        )
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)


def test_dict():
    print()
    print("Test Dict-1: create an empty dictionary")
    spec_string = "1<=n<=1"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("1",)
    run_times = []
    trials = 1000
    for D in param_list:
        t = timeit.Timer("x = dict()")
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Dict-2: dictionary lookup")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("1",)
    run_times = []
    trials = 1000
    for D in param_list:
        setup = "d = dict([(i,i) for i in range(%(n)s)])" % D
        t = timeit.Timer("x = d[1]", setup)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Dict-3: dictionary copy")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n",)
    run_times = []
    trials = 1000
    for D in param_list:
        setup = "d = dict([(i,i) for i in range(%(n)s)])" % D
        t = timeit.Timer("d.copy()", setup)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)

    print()
    print("Test Dict-4: dictionary list items")
    spec_string = "1000<=n<=100000"
    growth_factor = 2
    print("Spec_string: ", spec_string, "by factors of", growth_factor)
    var_list, param_list = make_param_list(spec_string, growth_factor)
    f_list = ("n*lg(n)",)
    run_times = []
    trials = 1000
    for D in param_list:
        setup = "d = dict([(i,i) for i in range(%(n)s)])" % D
        t = timeit.Timer("d.items()", setup)
        run_times.append(t.timeit(trials) * 1e6 / float(trials))
    fit(var_list, param_list, run_times, f_list)


def main():
    test_misc()
    test_number()
    test_string()
    test_list()
    test_dict()


if __name__ == "__main__":
    main()
