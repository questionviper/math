from algorithm import ahp
import judgment_matrix
if ahp.check_consistency(judgment_matrix.a_judgment_matrix, 6)<=0.1:
    print(1)