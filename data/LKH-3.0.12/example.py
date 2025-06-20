import requests
import lkh

problem_str = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n32-k5.vrp').text
problem = lkh.LKHProblem.parse(problem_str)
print(problem)


solver_path = '../LKH-3.0.12/LKH'
solution = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=10)
tour = [n - 1 for n in solution[0]]

print(f"Tour: {tour}")
print(f"Tour length: {solution[1]}")