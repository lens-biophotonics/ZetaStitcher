from .mapper import Term, QPBuilder

def pretty_solver_debug(solver_matrices, variables):
    builder = QPBuilder()
    objective = builder.create_expression()
    for i, vi in enumerate(variables):
        for j, vj in enumerate(variables):
            objective.add_term(
                term=Term([vi, vj]),
                coeff=solver_matrices.P[i,j]
            )

    for i, vi in enumerate(variables):
        objective.add_term(
            term=Term([vi]),
            coeff=solver_matrices.q[i]
        )
    print('DEBUG MAT OBJ', str(objective))