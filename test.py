import cuopt_mps_parser
from cuopt.linear_programming import Solve, SolverSettings

data_model = cuopt_mps_parser.ParseMps("batch_instances/neos8.mps")

settings = SolverSettings()
settings.set_mip_batch_pdlp_strong_branching(True)

solution = Solve(data_model, settings)

print(solution.get_termination_reason())
print(solution.get_primal_objective())
