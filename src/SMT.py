from pysmt.shortcuts import Symbol, And, GE, GT, LE, LT, Plus, Equals, Int, get_model, Real, NotEquals, Implies
from pysmt.shortcuts import Solver
from pysmt.typing import INT,REAL


def SMT_solver(rule,a,b,feature_num,con_num,data):
    letters = []
    letters1 = []
    for i in range(feature_num):
        letters.append(Symbol('x'+str(i), REAL))
        letters1.append(Symbol('x_'+str(i), INT))

    domains = And([And(GE(letters[i], Real(float(a[i]))), LE(letters[i], Real(float(b[i])))) for i in range(feature_num)])
    problem_rule = []

    for node in rule:
        if node[1] == ">":
            problem_rule.append(GT(letters[node[0]], Real(float(node[2]))))
        else:
            problem_rule.append(LE(letters[node[0]], Real(float(node[2]))))

    problem = And(problem_rule)

    constraint = And([And(Implies(Equals(letters[i],Real(float(data[i]))),Equals(letters1[i],Int(0))),Implies(NotEquals(letters[i],Real(float(data[i]))),Equals(letters1[i],Int(1)))) for i in range(feature_num)])
    sum_letters1 = Plus(letters1)
    problem1 = LE(sum_letters1,Int(con_num))
    formula = And([domains,problem,constraint,problem1])

    test_case = []
    with Solver(name='z3', random_seed = 23) as solver:
        solver.add_assertion(formula)
        if solver.solve():
            for l in letters:
                ans = solver.get_py_value(l)
                test_case.append(float(ans))
            print("find a solution")
        # else:
        #     print("No solution found")

    return test_case
