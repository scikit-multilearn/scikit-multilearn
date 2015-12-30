def get_matrix_in_format(original_matrix, matrix_format):
    if original_matrix.getformat() == matrix_format:
        return original_matrix
    
    return original_matrix.asformat(matrix_format)